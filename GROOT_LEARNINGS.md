# GR00T N1.6 Fine-tuning Learnings

## Overview

Fine-tuned NVIDIA GR00T N1.6 (3B parameters) on 60 teleoperation demos for the LeIsaac PickOrange task. This is the official approach recommended by the LeIsaac framework.

## Training Configuration

```
Base model:       nvidia/GR00T-N1.6-3B
Dataset:          60 episodes, 36K frames, dual camera (front + wrist)
Total steps:      10,000 (across 3 training phases)
Learning rates:   1e-4 -> 5e-5 -> 2e-5
Batch size:       8
Action horizon:   16 steps
Frozen:           Diffusion decoder (--no-tune-diffusion-model)
GPU:              RTX 4090 (24GB)
VRAM usage:       ~24GB during training, ~10GB inference
```

## Why 3 Training Phases?

Training crashed during checkpoint saves at ~24GB VRAM. Rather than fight the OOM, we split into 3 phases:

1. **Phase 1** (steps 0-3000): `lr=1e-4`, loss 0.854 -> 0.050
2. **Phase 2** (steps 3001-7000): `lr=5e-5`, loaded from ckpt-3000 as base model, loss 0.050 -> 0.023  
3. **Phase 3** (steps 7001-10000): `lr=2e-5`, loaded from ckpt-7000 as base model, loss 0.023 -> 0.017

Each phase uses `--base-model-path` pointing to the previous checkpoint, since GR00T's `launch_finetune.py` doesn't support `--resume-from-checkpoint`.

## Setup Issues Resolved

### 1. N1.5 vs N1.6
The GR00T repo only has code for N1.6 (`gr00t_n1d6`). N1.5 requires `trust_remote_code=True` and isn't registered in AutoModel. **Use N1.6.**

### 2. Dataset format (v3.0 -> v2.1)
GR00T's `LeRobotEpisodeLoader` expects `meta/episodes.jsonl` (v2.1 format). The HuggingFace dataset uses v3.0 with a `meta/episodes/` directory. Had to point to the v2.1 copy on disk.

### 3. modality.json
Must have 4 sections: `state`, `action`, `video`, AND `annotation`. Missing any causes KeyError.

```json
{
  "state": {"single_arm": {"start": 0, "end": 5}, "gripper": {"start": 5, "end": 6}},
  "action": {"single_arm": {"start": 0, "end": 5}, "gripper": {"start": 5, "end": 6}},
  "video": {"front": {"original_key": "observation.images.front"}, "wrist": {"original_key": "observation.images.wrist"}},
  "annotation": {"human.task_description": {"original_key": "annotation.human.task_description"}}
}
```

### 4. Annotation column must be integer task_index
The loader maps through `tasks_map` which expects integers, not text strings. Had to update all 60 episode parquet files.

### 5. Video codec: AV1 -> H.264
Decord doesn't support AV1. Re-encoded all 120 videos:
```bash
ffmpeg -y -i input.mp4 -c:v libx264 -crf 23 -preset fast -pix_fmt yuv420p output.mp4
```

### 6. Install GR00T with --no-deps
Full install replaces Isaac Lab's `torch 2.7.0+cu126` with generic torch, breaking everything.

### 7. nvjitlink version mismatch
`nvidia-nvjitlink-cu12==12.8.61` must be installed AND copied to Isaac Sim's pip_prebundle:
```bash
cp /isaac-sim/kit/python/lib/python3.11/site-packages/nvidia/nvjitlink/lib/libnvJitLink.so.12 \
   /isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/nvidia/nvjitlink/lib/
```

### 8. deepspeed causes CUDA_HOME errors
Uninstall it — not needed for 1-GPU training and nvcc isn't in the Isaac Lab container.

### 9. Disk quota management
RunPod network storage has per-pod quotas. Each checkpoint is ~14GB. Must aggressively clean with `--save-total-limit 2`.

## Eval Pipeline

GR00T uses a client-server architecture. Critical details:

### Server
```bash
python gr00t/eval/run_gr00t_server.py \
  --model_path /path/to/checkpoint \
  --embodiment_tag NEW_EMBODIMENT \
  --port 5555 \
  --use_sim_policy_wrapper   # REQUIRED for flat observation format
```

### Client
Must use GR00T's native `PolicyClient` (msgpack serialization), NOT LeIsaac's `Gr00tServicePolicyClient` (torch pickle). They are **incompatible**.

### Observation format
```python
obs = {
    "video.front": np.uint8, shape (B, T=1, H, W, C),
    "video.wrist": np.uint8, shape (B, T=1, H, W, C),
    "state.single_arm": np.float32, shape (B, T=1, 5),
    "state.gripper": np.float32, shape (B, T=1, 1),
    "annotation.human.task_description": ["Pick the orange and place it on the plate"],
}
```

### Action format (from server)
```python
action_dict = {
    "action.single_arm": np.array, shape (B, H=16, 5),  # action horizon = 16
    "action.gripper": np.array, shape (B, H=16, 1),
}
```

### Action conversion
```python
# Concatenate arm + gripper
concat = np.concatenate([arm, grip], axis=-1)  # (B, 16, 6)
# Take first batch, iterate over horizon
for i in range(16):
    action = convert_lerobot_action_to_leisaac(concat[0, i])  # degrees -> radians + joint mapping
    env.step(action.unsqueeze(0))
```

## Heartbeat Monitor

Used a background bash script to detect crashes within 30 seconds:

```bash
# Checks process alive, logs step/loss/VRAM every 30s
# Detects: crash (process gone), stall (same step for 5 min)
# Writes to /data/groot_heartbeat.log
```

## Results Comparison

| Model | Params | Steps | Loss | Grasp | Place | Notes |
|-------|--------|-------|------|-------|-------|-------|
| BC-RNN-GMM | ~1M | 200 ep | -13.84 | 0% | 0% | No vision, gripper collapse |
| BC-RNN + ResNet18 | ~12M | 200 ep | -14.30 | 0% | 0% | Wrong object attention |
| SmolVLA | 450M | 20K | 0.022 | 60% | 0% | Gripper never fully closes |
| **GR00T N1.6** | **3B** | **10K** | **0.017** | **100% (1/3)** | **100% (1/3)** | **Picks + places 1 orange, heading for 2nd at timeout** |
