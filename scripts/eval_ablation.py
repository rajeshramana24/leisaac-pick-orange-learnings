"""
SmolVLA eval with 3 ablation tests:
  Test A: Remove autocast, use bfloat16 weights (hypothesis #5)
  Test B: Increase denoising steps 10->50 (hypothesis #2)
  Test C: Scale gripper channel by 2x toward closing (hypothesis #1)
Run 2 episodes per test, compare gripper ranges and grasp rates.
"""

import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args([])
args_cli.headless = True
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np, torch, gymnasium as gym, os, math, imageio, time, copy
import leisaac
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

CHECKPOINT = "/data/smolvla_orange/checkpoints/020000/pretrained_model"
VIDEO_DIR = "/data/smolvla_orange/ablation_videos"
os.makedirs(VIDEO_DIR, exist_ok=True)
device = "cuda:0"
TASK_STRING = "Grab orange and place into plate"
MAX_STEPS = 480
DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi
ACTION_MEAN = torch.tensor([8.8597, 2.9652, 2.2492, 76.5756, 18.2819, 32.5675])
ACTION_STD = torch.tensor([19.5699, 39.6252, 41.9694, 13.6732, 9.6120, 12.3731])

print("=" * 60, flush=True)
print("  SmolVLA Ablation Tests", flush=True)
print("=" * 60, flush=True)

# Load policy + preprocessor
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

policy = SmolVLAPolicy.from_pretrained(CHECKPOINT)
policy.to(device)
policy.eval()
preprocessor, _ = make_pre_post_processors(policy.config)
print(f"  Loaded. num_steps={policy.config.num_steps}, n_action_steps={policy.config.n_action_steps}", flush=True)

# Create env
TASK = "LeIsaac-SO101-PickOrange-v0"
env_cfg = parse_env_cfg(TASK, num_envs=1, use_fabric=True)
env_cfg.use_teleop_device("so101leader")
env_cfg.recorders = {}
env = gym.make(TASK, cfg=env_cfg)
print("  Env created\n", flush=True)

def run_episode(policy, env, preprocessor, ep_name, use_autocast=True, gripper_scale=1.0):
    """Run one episode, return metrics."""
    obs, info = env.reset()
    policy.reset()
    frames, grip_actions = [], []
    metrics = {k: False for k in [f"orange{i}_{a}" for i in range(1,4) for a in ["grasped","placed"]]}
    metrics.update({k: None for k in [f"orange{i}_{a}_step" for i in range(1,4) for a in ["grasp","place"]]})
    metrics["task_done"] = False
    t0 = time.time()

    for step in range(MAX_STEPS):
        if "subtask_terms" in obs:
            st = obs["subtask_terms"]
            for i, on in enumerate(["001","002","003"]):
                if f"pick_orange{on}" in st and st[f"pick_orange{on}"][0].item() and not metrics[f"orange{i+1}_grasped"]:
                    metrics[f"orange{i+1}_grasped"] = True
                    metrics[f"orange{i+1}_grasp_step"] = step
                if f"put_orange{on}_to_plate" in st and st[f"put_orange{on}_to_plate"][0].item() and not metrics[f"orange{i+1}_placed"]:
                    metrics[f"orange{i+1}_placed"] = True
                    metrics[f"orange{i+1}_place_step"] = step

        # Capture frame
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
        jp_deg = policy_obs["joint_pos"][0].cpu().numpy() * RAD2DEG

        img_t = torch.tensor(frame, dtype=torch.float32).permute(2,0,1) if frame is not None else torch.zeros(3,480,640)
        wrist_cam = env.unwrapped.scene["wrist"]
        wrist_t = torch.zeros(3,480,640,dtype=torch.float32)
        if hasattr(wrist_cam, "data") and hasattr(wrist_cam.data, "output"):
            w_rgb = wrist_cam.data.output.get("rgb", None)
            if w_rgb is not None:
                wf = w_rgb[0].cpu().numpy()
                if wf.dtype != np.uint8: wf = (wf*255).clip(0,255).astype(np.uint8)
                if wf.shape[-1]==4: wf = wf[:,:,:3]
                wrist_t = torch.tensor(wf, dtype=torch.float32).permute(2,0,1)

        raw_obs = {
            "observation.images.camera1": img_t,
            "observation.images.camera2": wrist_t,
            "observation.state": torch.tensor(jp_deg, dtype=torch.float32),
            "task": TASK_STRING,
        }
        batch = preprocessor(raw_obs)

        with torch.no_grad():
            if use_autocast:
                with torch.autocast("cuda", dtype=torch.float16):
                    action = policy.select_action(batch)
            else:
                action = policy.select_action(batch)

        if isinstance(action, dict): action = action.get("action", action)
        if hasattr(action, 'action'): action = action.action
        if action.dim() == 1: action = action.unsqueeze(0)

        action_norm = action[0].cpu().float()
        action_deg = (action_norm * ACTION_STD + ACTION_MEAN).numpy()

        # Apply gripper scaling: push gripper toward closing
        if gripper_scale != 1.0:
            # Scale normalized gripper toward negative (closed)
            grip_norm = action_norm[5].item()
            grip_norm_scaled = grip_norm * gripper_scale  # amplify deviation from mean
            action_deg[5] = (grip_norm_scaled * ACTION_STD[5].item() + ACTION_MEAN[5].item())

        action_rad = action_deg * DEG2RAD
        action_tensor = torch.tensor(action_rad, dtype=torch.float32, device=device).unsqueeze(0)

        grip_actions.append(action_rad[5])
        obs, reward, terminated, truncated, info = env.step(action_tensor)

        if terminated[0].item():
            metrics["task_done"] = True
            break
        if truncated[0].item():
            break

    dt = time.time() - t0
    grip_actions = np.array(grip_actions)

    n_grasps = sum(1 for i in range(1,4) if metrics[f"orange{i}_grasped"])
    n_places = sum(1 for i in range(1,4) if metrics[f"orange{i}_placed"])

    print(f"    {ep_name}: {step+1} steps, {dt:.0f}s", flush=True)
    print(f"      Grip: [{grip_actions.min():.3f}, {grip_actions.max():.3f}] rad "
          f"([{grip_actions.min()*RAD2DEG:.1f}, {grip_actions.max()*RAD2DEG:.1f}] deg)", flush=True)
    print(f"      Grasps: {n_grasps}/3, Places: {n_places}/3, Done: {metrics['task_done']}", flush=True)

    if frames:
        path = os.path.join(VIDEO_DIR, f"{ep_name}.mp4")
        imageio.mimsave(path, frames, fps=30)
        print(f"      Video: {path}", flush=True)

    return {"grip_min": grip_actions.min(), "grip_max": grip_actions.max(),
            "grasps": n_grasps, "places": n_places, "done": metrics["task_done"]}

# ===== TEST A: No autocast (bfloat16 weights, full precision inference) =====
print("=" * 50, flush=True)
print("  TEST A: No autocast (full precision)", flush=True)
print("=" * 50, flush=True)
results_a = []
for i in range(2):
    r = run_episode(policy, env, preprocessor, f"testA_ep{i}", use_autocast=False)
    results_a.append(r)

# ===== TEST B: More denoising steps (50 instead of 10) =====
print("\n" + "=" * 50, flush=True)
print("  TEST B: 50 denoising steps (was 10)", flush=True)
print("=" * 50, flush=True)
orig_steps = policy.config.num_steps
policy.config.num_steps = 50
results_b = []
for i in range(2):
    r = run_episode(policy, env, preprocessor, f"testB_ep{i}", use_autocast=True)
    results_b.append(r)
policy.config.num_steps = orig_steps  # restore

# ===== TEST C: Gripper scaling 2x =====
print("\n" + "=" * 50, flush=True)
print("  TEST C: Gripper scale 2x (amplify close signal)", flush=True)
print("=" * 50, flush=True)
results_c = []
for i in range(2):
    r = run_episode(policy, env, preprocessor, f"testC_ep{i}", use_autocast=True, gripper_scale=2.0)
    results_c.append(r)

# ===== TEST D: Combined — no autocast + more steps + gripper scale =====
print("\n" + "=" * 50, flush=True)
print("  TEST D: Combined (no autocast + 50 steps + grip 2x)", flush=True)
print("=" * 50, flush=True)
policy.config.num_steps = 50
results_d = []
for i in range(2):
    r = run_episode(policy, env, preprocessor, f"testD_ep{i}", use_autocast=False, gripper_scale=2.0)
    results_d.append(r)
policy.config.num_steps = orig_steps

# ===== SUMMARY =====
print("\n" + "=" * 60, flush=True)
print("  ABLATION SUMMARY", flush=True)
print("=" * 60, flush=True)
for name, results in [("A: No autocast", results_a), ("B: 50 denoise steps", results_b),
                       ("C: Grip scale 2x", results_c), ("D: Combined", results_d)]:
    avg_gmin = np.mean([r["grip_min"] for r in results])
    avg_gmax = np.mean([r["grip_max"] for r in results])
    total_grasps = sum(r["grasps"] for r in results)
    total_places = sum(r["places"] for r in results)
    print(f"  {name:30s} | grip=[{avg_gmin*RAD2DEG:.1f}, {avg_gmax*RAD2DEG:.1f}]deg | "
          f"grasps={total_grasps}/6 | places={total_places}/6", flush=True)
print("=" * 60, flush=True)

env.close()
os.system("pkill -f 'http.server 8888' 2>/dev/null")
os.system(f"cd {VIDEO_DIR} && nohup /workspace/isaaclab/_isaac_sim/kit/python/bin/python3 -m http.server 8888 > /dev/null 2>&1 &")
print(f"\n  Videos: https://xg9eksc3dex7aq-8888.proxy.runpod.net/", flush=True)
simulation_app.close()
