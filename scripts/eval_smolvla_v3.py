"""SmolVLA eval using lerobot's preprocessor pipeline."""

import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args([])
args_cli.headless = True
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np, torch, gymnasium as gym, os, math, imageio, time
import leisaac
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

CHECKPOINT = "/data/smolvla_orange/checkpoints/020000/pretrained_model"
VIDEO_DIR = "/data/smolvla_orange/eval_videos"
os.makedirs(VIDEO_DIR, exist_ok=True)
device = "cuda:0"
TASK_STRING = "Grab orange and place into plate"
NUM_EPISODES = 5
MAX_STEPS = 480
DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi

# Action unnormalization stats (from training data, in degrees)
ACTION_MEAN = torch.tensor([8.8597, 2.9652, 2.2492, 76.5756, 18.2819, 32.5675])
ACTION_STD = torch.tensor([19.5699, 39.6252, 41.9694, 13.6732, 9.6120, 12.3731])

print("=" * 60, flush=True)
print("  SmolVLA Eval v3 (with preprocessor pipeline)", flush=True)
print("=" * 60, flush=True)

# Load policy + create preprocessor/postprocessor
print("\n[1/3] Loading SmolVLA + preprocessor...", flush=True)
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

policy = SmolVLAPolicy.from_pretrained(CHECKPOINT)
policy.to(device)
# policy.half() -- use autocast instead
policy.eval()

preprocessor, postprocessor = make_pre_post_processors(policy.config)
print(f"  Policy loaded, preprocessor has {len(preprocessor.steps)} steps", flush=True)

# Create env
print("\n[2/3] Creating PickOrange env...", flush=True)
TASK = "LeIsaac-SO101-PickOrange-v0"
env_cfg = parse_env_cfg(TASK, num_envs=1, use_fabric=True)
env_cfg.use_teleop_device("so101leader")
env_cfg.recorders = {}
env = gym.make(TASK, cfg=env_cfg)
print("  Env created", flush=True)

# Run episodes
print(f"\n[3/3] Running {NUM_EPISODES} episodes...", flush=True)

all_episode_metrics = []

for ep in range(NUM_EPISODES):
    obs, info = env.reset()
    policy.reset()
    frames = []
    grip_actions = []
    metrics = {
        "orange1_grasped": False, "orange1_grasp_step": None,
        "orange2_grasped": False, "orange2_grasp_step": None,
        "orange3_grasped": False, "orange3_grasp_step": None,
        "orange1_placed": False, "orange1_place_step": None,
        "orange2_placed": False, "orange2_place_step": None,
        "orange3_placed": False, "orange3_place_step": None,
        "task_done": False, "task_done_step": None,
        "grip_min": float("inf"), "grip_max": float("-inf"),
        "total_steps": 0,
    }
    t0 = time.time()

    for step in range(MAX_STEPS):
        # Read subtask observations
        if "subtask_terms" in obs:
            st = obs["subtask_terms"]
            for i, oname in enumerate(["001", "002", "003"]):
                gk = f"pick_orange{oname}"
                pk = f"put_orange{oname}_to_plate"
                if gk in st and st[gk][0].item() and not metrics[f"orange{i+1}_grasped"]:
                    metrics[f"orange{i+1}_grasped"] = True
                    metrics[f"orange{i+1}_grasp_step"] = step
                if pk in st and st[pk][0].item() and not metrics[f"orange{i+1}_placed"]:
                    metrics[f"orange{i+1}_placed"] = True
                    metrics[f"orange{i+1}_place_step"] = step

        # Capture cameras
        front_cam = env.unwrapped.scene["front"]
        frame = None
        if hasattr(front_cam, "data") and hasattr(front_cam.data, "output"):
            rgb = front_cam.data.output.get("rgb", None)
            if rgb is not None:
                frame = rgb[0].cpu().numpy()
                if frame.dtype != np.uint8:
                    frame = (frame * 255).clip(0, 255).astype(np.uint8)
                if frame.shape[-1] == 4:
                    frame = frame[:, :, :3]
                frames.append(frame)

        policy_obs = obs["policy"]
        jp_rad = policy_obs["joint_pos"][0].cpu().numpy()
        jp_deg = jp_rad * RAD2DEG

        # Build raw obs for preprocessor (original key names, unbatched)
        if frame is not None:
            img_t = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1)  # (3,H,W)
        else:
            img_t = torch.zeros(3, 480, 640, dtype=torch.float32)

        wrist_cam = env.unwrapped.scene["wrist"]
        wrist_t = torch.zeros(3, 480, 640, dtype=torch.float32)
        if hasattr(wrist_cam, "data") and hasattr(wrist_cam.data, "output"):
            w_rgb = wrist_cam.data.output.get("rgb", None)
            if w_rgb is not None:
                wf = w_rgb[0].cpu().numpy()
                if wf.dtype != np.uint8:
                    wf = (wf * 255).clip(0, 255).astype(np.uint8)
                if wf.shape[-1] == 4:
                    wf = wf[:, :, :3]
                wrist_t = torch.tensor(wf, dtype=torch.float32).permute(2, 0, 1)

        raw_obs = {
            "observation.images.camera1": img_t,         # (3,H,W) float, CPU
            "observation.images.camera2": wrist_t,       # (3,H,W) float, CPU
            "observation.state": torch.tensor(jp_deg, dtype=torch.float32),  # (6,) degrees
            "task": TASK_STRING,
        }

        # Preprocess (rename, batch, tokenize, normalize, to device)
        batch = preprocessor(raw_obs)

        # Inference
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.float16):
                action = policy.select_action(batch)

        # Postprocess (unnormalize)
        action = postprocessor(action)

        if isinstance(action, dict):
            action = action.get("action", action)
        if hasattr(action, 'action'):
            action = action.action
        if action.dim() == 1:
            action = action.unsqueeze(0)

        action_normalized = action[0].cpu()
        action_deg = (action_normalized * ACTION_STD + ACTION_MEAN).numpy()
        action_rad = action_deg * DEG2RAD
        action_tensor = torch.tensor(action_rad, dtype=torch.float32, device=device).unsqueeze(0)

        grip_actions.append(action_rad[5])
        metrics["grip_min"] = min(metrics["grip_min"], action_rad[5])
        metrics["grip_max"] = max(metrics["grip_max"], action_rad[5])

        obs, reward, terminated, truncated, info = env.step(action_tensor)
        metrics["total_steps"] = step + 1

        if terminated[0].item():
            metrics["task_done"] = True
            metrics["task_done_step"] = step
            break
        if truncated[0].item():
            break

    dt = time.time() - t0
    grip_actions = np.array(grip_actions)

    print(f"\n  {'='*55}", flush=True)
    print(f"  Episode {ep} ({metrics['total_steps']} steps, {dt:.1f}s)", flush=True)
    print(f"  {'='*55}", flush=True)
    print(f"  Gripper: [{metrics['grip_min']:.4f}, {metrics['grip_max']:.4f}] rad "
          f"([{metrics['grip_min']*RAD2DEG:.1f}, {metrics['grip_max']*RAD2DEG:.1f}] deg)", flush=True)
    for i in range(1, 4):
        g = "YES" if metrics[f"orange{i}_grasped"] else "no"
        gs = f" (step {metrics[f'orange{i}_grasp_step']})" if metrics[f"orange{i}_grasped"] else ""
        p = "YES" if metrics[f"orange{i}_placed"] else "no"
        ps = f" (step {metrics[f'orange{i}_place_step']})" if metrics[f"orange{i}_placed"] else ""
        print(f"  Orange {i}: grasped={g}{gs}, placed={p}{ps}", flush=True)
    td = "YES" if metrics["task_done"] else "no"
    tds = f" (step {metrics['task_done_step']})" if metrics["task_done"] else ""
    print(f"  Task complete: {td}{tds}", flush=True)
    for s in [0, 100, 200, 300, 400]:
        if s < len(grip_actions):
            print(f"    Step {s:>4}: grip={grip_actions[s]:.4f} rad ({grip_actions[s]*RAD2DEG:.1f} deg)", flush=True)
    if frames:
        path = os.path.join(VIDEO_DIR, f"smolvla_ep{ep}.mp4")
        imageio.mimsave(path, frames, fps=30)
        print(f"  Video: {path}", flush=True)
    all_episode_metrics.append(metrics)

# Summary
print(f"\n{'='*60}", flush=True)
print(f"  SUMMARY ({NUM_EPISODES} episodes)", flush=True)
print(f"{'='*60}", flush=True)
ng = sum(1 for m in all_episode_metrics if any(m[f"orange{i}_grasped"] for i in range(1,4)))
np_ = sum(1 for m in all_episode_metrics if any(m[f"orange{i}_placed"] for i in range(1,4)))
nd = sum(1 for m in all_episode_metrics if m["task_done"])
print(f"  Grasp rate:  {ng}/{NUM_EPISODES} ({100*ng/NUM_EPISODES:.0f}%)", flush=True)
print(f"  Place rate:  {np_}/{NUM_EPISODES} ({100*np_/NUM_EPISODES:.0f}%)", flush=True)
print(f"  Complete:    {nd}/{NUM_EPISODES} ({100*nd/NUM_EPISODES:.0f}%)", flush=True)
print(f"{'='*60}", flush=True)

env.close()
os.system("pkill -f 'http.server 8888' 2>/dev/null")
os.system(f"cd {VIDEO_DIR} && nohup /workspace/isaaclab/_isaac_sim/kit/python/bin/python3 -m http.server 8888 > /dev/null 2>&1 &")
print(f"\n  Download: https://xg9eksc3dex7aq-8888.proxy.runpod.net/", flush=True)
simulation_app.close()
