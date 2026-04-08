"""GR00T N1.6 final eval — 3 episodes with video recording."""
import multiprocessing
if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args([])
args_cli.headless = True
args_cli.enable_cameras = True
args_cli.device = "cuda"
app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import os, torch, numpy as np
import gymnasium as gym
from PIL import Image
from isaaclab_tasks.utils import parse_env_cfg
import leisaac  # noqa: F401
from leisaac.utils.env_utils import get_task_type, dynamic_reset_gripper_effort_limit_sim
from leisaac.utils.robot_utils import convert_leisaac_action_to_lerobot, convert_lerobot_action_to_leisaac
from gr00t.policy.server_client import PolicyClient

SAVE_DIR = "/data/groot_eval_final"
NUM_EPISODES = 3
STEPS_PER_EPISODE = 900  # 15 seconds at 60Hz — longer to allow pick+place

task = "LeIsaac-SO101-PickOrange-v0"
env_cfg = parse_env_cfg(task, device="cuda", num_envs=1)
task_type = get_task_type(task)
env_cfg.use_teleop_device(task_type)
if hasattr(env_cfg.terminations, "time_out"):
    env_cfg.terminations.time_out = None
env_cfg.recorders = None
env = gym.make(task, cfg=env_cfg).unwrapped

client = PolicyClient(host="localhost", port=5555, timeout_ms=15000, strict=False)
assert client.ping(), "Server not reachable!"
print("Connected to GR00T server!")

all_sensor_keys = list(env.scene.sensors.keys())
camera_keys = [k for k in all_sensor_keys if k in ("front", "wrist")]
print(f"Cameras: {camera_keys}")

for ep in range(NUM_EPISODES):
    ep_dir = f"{SAVE_DIR}/episode_{ep}"
    os.makedirs(ep_dir, exist_ok=True)

    obs_dict, _ = env.reset()
    frame_idx = 0
    print(f"\n=== Episode {ep+1}/{NUM_EPISODES} ({STEPS_PER_EPISODE} steps) ===")

    for step in range(STEPS_PER_EPISODE // 16):
        obs = obs_dict["policy"]

        gr00t_obs = {}
        for key in camera_keys:
            img_data = obs[key].cpu().numpy()
            if img_data.dtype != np.uint8:
                img_data = (img_data * 255).clip(0, 255).astype(np.uint8)
            gr00t_obs[f"video.{key}"] = img_data[:, None, :, :, :]
        joint_pos = convert_leisaac_action_to_lerobot(obs["joint_pos"])
        gr00t_obs["state.single_arm"] = joint_pos[:, None, 0:5].astype(np.float32)
        gr00t_obs["state.gripper"] = joint_pos[:, None, 5:6].astype(np.float32)
        gr00t_obs["annotation.human.task_description"] = ["Pick the orange and place it on the plate"]

        action_dict, info = client._get_action(gr00t_obs)

        arm_action = np.array(action_dict["action.single_arm"])
        grip_action = np.array(action_dict["action.gripper"])
        if grip_action.ndim < arm_action.ndim:
            grip_action = grip_action[..., None]
        concat_action = np.concatenate([arm_action, grip_action], axis=-1)
        action_horizon = concat_action[0]
        action_horizon = convert_lerobot_action_to_leisaac(action_horizon)
        actions = torch.from_numpy(action_horizon).to(env.device)

        for i in range(min(16, actions.shape[0])):
            action = actions[i].unsqueeze(0)
            dynamic_reset_gripper_effort_limit_sim(env, task_type)
            obs_dict, _, _, _, _ = env.step(action)
            # Save every 3rd frame
            if frame_idx % 3 == 0:
                for cam_key in camera_keys:
                    img = obs_dict["policy"][cam_key][0].cpu().numpy()
                    if img.dtype != np.uint8:
                        img = (img * 255).clip(0, 255).astype(np.uint8)
                    Image.fromarray(img).save(f"{ep_dir}/{cam_key}_{frame_idx:05d}.png")
            frame_idx += 1
            if frame_idx % 120 == 0:
                print(f"  Step {frame_idx}/{STEPS_PER_EPISODE}")

    print(f"  Episode {ep+1} done — {frame_idx} steps")

    # Generate videos for this episode
    os.system(f"ffmpeg -y -framerate 20 -pattern_type glob -i '{ep_dir}/front_*.png' -c:v libx264 -pix_fmt yuv420p /data/groot_eval_final/episode_{ep}_front.mp4")
    os.system(f"ffmpeg -y -framerate 20 -pattern_type glob -i '{ep_dir}/wrist_*.png' -c:v libx264 -pix_fmt yuv420p /data/groot_eval_final/episode_{ep}_wrist.mp4")
    print(f"  Videos saved for episode {ep+1}")

print(f"\nAll {NUM_EPISODES} episodes complete!")
print(f"Videos at: {SAVE_DIR}/")
env.close()
simulation_app.close()
