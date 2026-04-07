"""Train BC-RNN-GMM with vision (front image + joint_pos + joint_vel)."""

import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args([])
args_cli.headless = True
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import h5py, os, time, torch, numpy as np, imageio
import robomimic, robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.config import config_factory
from robomimic.algo import algo_factory
import gymnasium as gym
import leisaac
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

DATASET = "/data/mimicgen/seed_demos/pick_orange_with_images.hdf5"
OUTPUT_DIR = "/data/mimicgen/bc_orange_vision"
VIDEO_DIR = "/data/mimicgen/bc_vision_videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)
device = "cuda:0"

print("=" * 60, flush=True)
print("  BC-RNN-GMM with VISION (front image + joint state)", flush=True)
print(f"  Dataset: {DATASET}", flush=True)
print("=" * 60, flush=True)

# Verify dataset
with h5py.File(DATASET, "r") as f:
    d0 = f["data/demo_0"]
    print(f"  demo_0 keys: {list(d0['obs'].keys())}", flush=True)
    print(f"  actions: {d0['actions'].shape}", flush=True)
    print(f"  front_image: {d0['obs/front_image'].shape}, dtype={d0['obs/front_image'].dtype}", flush=True)
    num_demos = len(f["data"].keys())
    print(f"  Total demos: {num_demos}", flush=True)

config = config_factory(algo_name="bc")
config.train.data = DATASET
config.train.output_dir = OUTPUT_DIR
config.train.hdf5_cache_mode = "low_dim"  # cache low_dim, load images on demand
config.train.num_data_workers = 2
config.train.batch_size = 64  # smaller batch for images
config.train.num_epochs = 300  # more epochs for vision
config.train.seed = 42

# Observations: low_dim (joint state) + rgb (front camera)
config.observation.modalities.obs.low_dim = ["joint_pos", "joint_vel"]
config.observation.modalities.obs.rgb = ["front_image"]
config.observation.modalities.obs.depth = []
config.observation.modalities.obs.scan = []
config.observation.modalities.goal.low_dim = []
config.observation.modalities.goal.rgb = []

# Image encoder config
config.observation.encoder.rgb.core_class = "VisualCore"
config.observation.encoder.rgb.core_kwargs.feature_dimension = 64
config.observation.encoder.rgb.core_kwargs.backbone_class = "ResNet18Conv"
config.observation.encoder.rgb.core_kwargs.backbone_kwargs.pretrained = False
config.observation.encoder.rgb.core_kwargs.backbone_kwargs.input_coord_conv = False
config.observation.encoder.rgb.core_kwargs.pool_class = "SpatialSoftmax"
config.observation.encoder.rgb.core_kwargs.pool_kwargs.num_kp = 32

# Policy network
config.algo.optim_params.policy.learning_rate.initial = 1e-4
config.algo.actor_layer_dims = (256, 256)  # slightly smaller for vision
config.algo.rnn.enabled = True
config.algo.rnn.horizon = 10
config.algo.rnn.hidden_dim = 400
config.algo.rnn.rnn_type = "LSTM"
config.algo.rnn.num_layers = 2
config.algo.gmm.enabled = True
config.algo.gmm.num_modes = 5
config.algo.gmm.min_std = 0.0001

config.experiment.name = "so101_orange_bc_vision"
config.experiment.validate = False
config.experiment.save.enabled = True
config.experiment.save.every_n_epochs = 50
config.experiment.epoch_every_n_steps = 200
config.experiment.rollout.enabled = False
config.lock()

ObsUtils.initialize_obs_utils_with_config(config)
torch.manual_seed(42)

trainset, _ = TrainUtils.load_data_for_training(config, obs_keys=config.all_obs_keys)
print(f"  Dataset: {len(trainset)} samples", flush=True)

train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=config.train.batch_size, shuffle=True,
    num_workers=config.train.num_data_workers, drop_last=True)

with h5py.File(DATASET, "r") as f:
    d0 = f["data/demo_0/obs"]
    obs_shapes = {}
    for k in config.all_obs_keys:
        if k in d0:
            shape = d0[k].shape[1:]
            # robomimic process_frame converts HWC->CHW, so register CHW shape for image keys
            if len(shape) == 3 and shape[-1] == 3:  # (H, W, 3) -> (3, H, W)
                shape = (shape[2], shape[0], shape[1])
            obs_shapes[k] = shape
    ac_dim = f["data/demo_0/actions"].shape[1]

print(f"  Obs shapes: {obs_shapes}", flush=True)
print(f"  Action dim: {ac_dim}", flush=True)

model = algo_factory(algo_name=config.algo_name, config=config,
                     obs_key_shapes=obs_shapes, ac_dim=ac_dim, device=device)

total_params = sum(p.numel() for p in model.nets.parameters())
trainable_params = sum(p.numel() for p in model.nets.parameters() if p.requires_grad)
print(f"  Model params: {total_params:,} total, {trainable_params:,} trainable", flush=True)

print(f"\n  Training {config.train.num_epochs} epochs...", flush=True)
best_loss = float("inf")
for epoch in range(1, config.train.num_epochs + 1):
    t0 = time.time()
    step_log = TrainUtils.run_epoch(model=model, data_loader=train_loader,
                                    epoch=epoch, validate=False, num_steps=200)
    dt = time.time() - t0
    loss = step_log.get("Loss", 0.0)
    if epoch % 10 == 0 or epoch == 1:
        print(f"  Epoch {epoch:>4d}: loss={loss:.4f} ({dt:.1f}s)", flush=True)
    if loss < best_loss:
        best_loss = loss
        TrainUtils.save_model(model=model, config=config, env_meta={},
                             shape_meta={"obs_key_shapes": obs_shapes, "ac_dim": ac_dim},
                             ckpt_path=os.path.join(OUTPUT_DIR, "model_best.pth"))

TrainUtils.save_model(model=model, config=config, env_meta={},
                     shape_meta={"obs_key_shapes": obs_shapes, "ac_dim": ac_dim},
                     ckpt_path=os.path.join(OUTPUT_DIR, "model_final.pth"))
print(f"\n  Training done! Best loss: {best_loss:.4f}", flush=True)

# ===== EVAL: Replay in PickOrange env =====
print(f"\n  Recording 3 episodes with vision BC model...", flush=True)

TASK = "LeIsaac-SO101-PickOrange-v0"
env_cfg = parse_env_cfg(TASK, num_envs=1, use_fabric=True)
env_cfg.use_teleop_device("so101leader")
env_cfg.recorders = {}
env = gym.make(TASK, cfg=env_cfg)

# Reload best model
ckpt = torch.load(os.path.join(OUTPUT_DIR, "model_best.pth"), map_location=device)
config2, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt)
ObsUtils.initialize_obs_utils_with_config(config2)
model2 = algo_factory(algo_name=config2.algo_name, config=config2,
                      obs_key_shapes=obs_shapes, ac_dim=ac_dim, device=device)
model2.deserialize(ckpt["model"])
model2.set_eval()

import cv2

for ep in range(3):
    obs, info = env.reset()
    model2.reset()
    frames = []
    grip_actions = []

    for step in range(480):
        # Capture front camera for video recording
        front_cam = env.unwrapped.scene["front"]
        if hasattr(front_cam, "data") and hasattr(front_cam.data, "output"):
            rgb = front_cam.data.output.get("rgb", None)
            if rgb is not None:
                frame = rgb[0].cpu().numpy()
                if frame.dtype != np.uint8:
                    frame = (frame * 255).clip(0, 255).astype(np.uint8)
                if frame.shape[-1] == 4:
                    frame = frame[:, :, :3]
                frames.append(frame)

                # Prepare image obs for BC model (84x84, uint8)
                img_resized = cv2.resize(frame, (84, 84))
                # robomimic expects HWC numpy for process_frame, but get_action handles it
                front_obs = torch.tensor(img_resized, dtype=torch.uint8, device=device).unsqueeze(0)
            else:
                front_obs = torch.zeros(1, 84, 84, 3, dtype=torch.uint8, device=device)
        else:
            front_obs = torch.zeros(1, 84, 84, 3, dtype=torch.uint8, device=device)

        policy_obs = obs["policy"]
        bc_obs = {
            "joint_pos": policy_obs["joint_pos"],
            "joint_vel": policy_obs["joint_vel"],
            "front_image": front_obs,
        }

        with torch.no_grad():
            action = model2.get_action(bc_obs)
            if isinstance(action, dict):
                action = action["action"]

        grip_actions.append(action[0, 5].item())
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated[0].item() or truncated[0].item():
            break

    grip_actions = np.array(grip_actions)
    print(f"  Episode {ep}: gripper=[{grip_actions.min():.4f}, {grip_actions.max():.4f}]", flush=True)
    for s in [0, 50, 100, 200, 300, 400]:
        if s < len(grip_actions):
            print(f"    Step {s}: grip_cmd={grip_actions[s]:.4f}", flush=True)

    if frames:
        path = os.path.join(VIDEO_DIR, f"bc_vision_ep{ep}.mp4")
        imageio.mimsave(path, frames, fps=30)
        print(f"    Video: {path}", flush=True)

env.close()

os.system("pkill -f 'http.server 8888' 2>/dev/null")
os.system(f"cd {VIDEO_DIR} && nohup /workspace/isaaclab/_isaac_sim/kit/python/bin/python3 -m http.server 8888 > /dev/null 2>&1 &")
print(f"\n  Download: https://xg9eksc3dex7aq-8888.proxy.runpod.net/", flush=True)

simulation_app.close()
