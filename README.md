# LeIsaac PickOrange: BC, VLA & GR00T Training Learnings

Training a SO-101 robot arm to pick oranges and place them on a plate in NVIDIA Isaac Lab simulation, using the [LightwheelAI/leisaac](https://github.com/LightwheelAI/leisaac) framework.

## Task

**LeIsaac-SO101-PickOrange-v0**: A kitchen scene with 3 oranges on a counter. The SO-101 5-DOF arm + gripper must pick each orange and place it on a plate.

- **Robot**: SO-101 follower (6 joints: 5 arm + 1 gripper)
- **Dataset**: [LightwheelAI/leisaac-pick-orange](https://huggingface.co/datasets/LightwheelAI/leisaac-pick-orange) (60 episodes, 36K frames, 30fps)
- **Cameras**: Front (480x640) + Wrist (480x640)
- **Actions**: 6D absolute joint positions (degrees in dataset, radians in env)

## Approaches Tried

| Approach | Model | Vision? | Grasp Rate | Place Rate | Issue |
|----------|-------|---------|------------|------------|-------|
| **1. BC-RNN-GMM (joint-only)** | robomimic | No | 0% | 0% | Gripper collapses to mean (~0.9 rad), never closes |
| **2. BC-RNN-GMM + ResNet18** | robomimic | Yes (84x84) | 0% | 0% | Arm goes to plate instead of oranges |
| **3. SmolVLA fine-tune** | SmolVLA (450M) | Yes (front+wrist) | **60%** | 0% | Gripper hovers at 34-44 deg, never fully closes |
| **4. GR00T N1.6 fine-tune** | GR00T N1.6 (3B) | Yes (front+wrist) | **Reaching + grasping** | **In progress** | Best result so far |

## GR00T N1.6 Results (Latest)

### Training
- **Base model**: `nvidia/GR00T-N1.6-3B`
- **Total training**: 10,000 steps across 3 phases (3K + 4K + 3K)
- **Learning rates**: 1e-4 -> 5e-5 -> 2e-5 (progressive reduction)
- **Final loss**: 0.017 (from 0.854 at step 0)
- **Frozen diffusion decoder** (`--no-tune-diffusion-model`) to fit 24GB VRAM
- **Dataset**: 60 demo episodes, dual camera (front + wrist), SO-101 5-DOF + gripper

### Loss Curve

```
Step    | Loss  | Phase
--------|-------|------
    250 | 0.854 | Phase 1 (lr=1e-4)
  1,500 | 0.058 |
  3,000 | 0.050 |
  4,000 | 0.044 | -- checkpoint saved, training crashed mid-save at 4K --
  4,500 | 0.035 | Phase 2 (lr=5e-5, from ckpt-3000)
  6,000 | 0.025 |
  7,000 | 0.023 | -- checkpoint saved --
  8,000 | 0.021 | Phase 3 (lr=2e-5, from ckpt-7000)
  9,000 | 0.018 |
 10,000 | 0.017 | -- final model --
```

Loss dropped **98%** (0.854 -> 0.017) over 10K steps.

### Eval Videos (3 Episodes)

Each episode runs 900 sim steps (15 seconds at 60Hz) with a fresh env reset.

| Episode | Front Camera | Wrist Camera |
|---------|-------------|-------------|
| 1 | [episode_0_front.mp4](videos/groot_eval_final_10k/episode_0_front.mp4) | [episode_0_wrist.mp4](videos/groot_eval_final_10k/episode_0_wrist.mp4) |
| 2 | [episode_1_front.mp4](videos/groot_eval_final_10k/episode_1_front.mp4) | [episode_1_wrist.mp4](videos/groot_eval_final_10k/episode_1_wrist.mp4) |
| 3 | [episode_2_front.mp4](videos/groot_eval_final_10k/episode_2_front.mp4) | [episode_2_wrist.mp4](videos/groot_eval_final_10k/episode_2_wrist.mp4) |

### Progress Videos (Earlier Checkpoints)

| Checkpoint | Steps | Loss | Front | Wrist |
|-----------|-------|------|-------|-------|
| ckpt-3000 | 3,000 | 0.050 | [front](videos/groot_eval/groot_eval_ckpt3000_front.mp4) | [wrist](videos/groot_eval/groot_eval_ckpt3000_wrist.mp4) |
| ckpt-7000 | 7,000 | 0.023 | [front](videos/groot_eval/groot_eval_ckpt7000_front.mp4) | [wrist](videos/groot_eval/groot_eval_ckpt7000_wrist.mp4) |

## Key Findings

### 1. Joint-only BC fails completely
Training BC-RNN-GMM on `joint_pos + joint_vel` (no images) produces a model that outputs near-constant actions. During closed-loop rollout, observations quickly diverge from training distribution, and the GMM collapses to its dominant mode (gripper open).

### 2. Vision is necessary but not sufficient
Adding a ResNet18 vision encoder helped the model move toward objects, but it attended to the wrong object (plate instead of oranges). Training from scratch with a randomly initialized CNN on only 60 demos is insufficient.

### 3. SmolVLA achieves 60% grasp rate but 0% place
SmolVLA's pretrained vision-language backbone provides better visual understanding. With 20K steps fine-tuning (loss 0.022), the model reaches toward oranges and partially closes the gripper, but never firmly grasps. Also discovered a critical postprocessor unnormalization bug.

### 4. GR00T N1.6 is the right approach
GR00T N1.6 with the LeIsaac pipeline produces the best results. The 3B parameter model with language conditioning ("Pick the orange and place it on the plate") successfully reaches, grasps, and attempts to pick oranges after 10K steps of fine-tuning.

### 5. The SmolVLA postprocessor bug
SmolVLA's `select_action()` returns **normalized** actions. The postprocessor was a no-op:
```python
action_degrees = normalized_action * ACTION_STD + ACTION_MEAN
```

### 6. GR00T client-server serialization mismatch
LeIsaac's `Gr00tServicePolicyClient` uses **torch pickle** serialization, but GR00T's `PolicyServer` uses **msgpack**. Must use GR00T's native `PolicyClient` for eval, not LeIsaac's wrapper.

### 7. GR00T observation format requirements
- Video: `(B, T, H, W, C)` — needs temporal dimension T=1
- State: `(B, T, D)` — needs temporal dimension T=1
- State dtype: `np.float32` (not float64)
- Server flag: `--use_sim_policy_wrapper` required for flat key format

### 8. VRAM management for 24GB GPU
- Training with `--no-tune-diffusion-model`: ~24GB (tight, crashes during checkpoint save)
- Inference server: ~10GB
- Isaac Lab env with cameras: ~5GB
- Checkpoint saves at 24GB cause OOM — must save fewer checkpoints with `--save-total-limit 2`

### 9. Isaac Lab + GR00T dependency conflicts
- GR00T must be installed with `pip install -e . --no-deps` to avoid breaking Isaac Lab's torch
- `nvidia-nvjitlink-cu12==12.8.61` must be installed and copied to pip_prebundle
- `deepspeed` must be uninstalled (no nvcc in container)
- `numpy` must stay at 1.26.4 (< 2.0)

### 10. Dataset format conversion
- GR00T expects LeRobot v2.1 format (`meta/episodes.jsonl`), not v3.0
- modality.json needs 4 sections: `state`, `action`, `video`, `annotation`
- Annotation column must be integer `task_index`, not text string
- Videos must be H.264 codec (not AV1) for decord compatibility

## Training Loss Curves

### GR00T N1.6 Fine-tuning (10K steps)

```
Step    | Loss
--------|-------
    250 | 0.854
    500 | 0.512
  1,000 | 0.082
  2,000 | 0.058
  3,000 | 0.050
  4,000 | 0.044
  5,000 | 0.030
  6,000 | 0.025
  7,000 | 0.023
  8,000 | 0.021
  9,000 | 0.018
 10,000 | 0.017
```

### SmolVLA Fine-tuning (20K steps)

```
Step   | Loss
-------|-------
   100 | 0.147
 1,000 | 0.079
 2,000 | 0.068
 4,000 | 0.049
 8,000 | 0.033
12,000 | 0.028
16,000 | 0.021
20,000 | 0.022
```

## Videos

### GR00T N1.6 Final Eval (10K steps, loss 0.017)
3 episodes with env reset between each:
- `videos/groot_eval_final_10k/episode_0_front.mp4` / `episode_0_wrist.mp4`
- `videos/groot_eval_final_10k/episode_1_front.mp4` / `episode_1_wrist.mp4`
- `videos/groot_eval_final_10k/episode_2_front.mp4` / `episode_2_wrist.mp4`

### GR00T Checkpoint Progress
- `videos/groot_eval/groot_eval_ckpt3000_front.mp4` — Reaching toward oranges
- `videos/groot_eval/groot_eval_ckpt7000_front.mp4` — Picking up oranges

### Ground Truth Training Demos
- `videos/ground-truth/gt_front_ep0-4.mp4`
- `videos/ground-truth/gt_wrist_ep0-4.mp4`

### SmolVLA Eval (60% grasp rate)
- `videos/smolvla-eval/smolvla_ep0-4.mp4`

### Ablation Tests
- `videos/ablation/testA-D_ep0-1.mp4`

### BC-RNN + Vision Eval
- `videos/bc-vision/epoch033_ep0-1.mp4`

## Technical Details

### Action Statistics (degrees)
```
Joint          | Mean   | Std    | Min     | Max
---------------|--------|--------|---------|--------
shoulder_pan   |  8.86  | 19.57  | -38.50  | 52.06
shoulder_lift  |  2.97  | 39.63  | -100.00 | 63.76
elbow_flex     |  2.25  | 41.97  | -99.19  | 99.46
wrist_flex     | 76.58  | 13.67  |  21.53  | 100.00
wrist_roll     | 18.28  |  9.61  | -14.45  | 51.02
gripper        | 32.57  | 12.37  |   0.60  | 91.40
```

### Environment
- **Pod**: RunPod RTX 4090 (24GB VRAM)
- **Isaac Lab**: 2.3.2 (Isaac Sim 5.1, Python 3.11)
- **LeIsaac**: v0.1.2
- **GR00T**: N1.6, `nvidia/GR00T-N1.6-3B`

## Repository Structure

```
.
├── README.md                          # This file
├── SMOLVLA_LEARNINGS.md               # Detailed SmolVLA training log
├── GROOT_LEARNINGS.md                 # GR00T N1.6 fine-tuning details
├── videos/
│   ├── groot_eval_final_10k/          # Final 10K eval (3 episodes)
│   ├── groot_eval/                    # Checkpoint progress videos
│   ├── ground-truth/                  # Training demo recordings
│   ├── smolvla-eval/                  # SmolVLA eval episodes
│   ├── ablation/                      # Ablation test videos
│   └── bc-vision/                     # BC-RNN + ResNet18 eval
└── scripts/
    ├── train_smolvla.py               # SmolVLA fine-tuning (patched)
    ├── eval_smolvla.py                # SmolVLA eval with metrics
    ├── eval_groot_3ep.py              # GR00T 3-episode eval with video
    ├── eval_ablation.py               # Ablation test script
    └── convert_dataset.py             # Dataset conversion
```
