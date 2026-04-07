# LeIsaac PickOrange: BC & VLA Training Learnings

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
| **4. GR00T N1.5** | GR00T (1.3B) | Yes | TBD | TBD | Next approach (official LeIsaac pipeline) |

## Key Findings

### 1. Joint-only BC fails completely
Training BC-RNN-GMM on `joint_pos + joint_vel` (no images) produces a model that outputs near-constant actions. During closed-loop rollout, observations quickly diverge from training distribution, and the GMM collapses to its dominant mode (gripper open). This happens regardless of dataset quality.

### 2. Vision is necessary but not sufficient
Adding a ResNet18 vision encoder helped the model move toward objects, but it attended to the wrong object (plate instead of oranges). Training from scratch with a randomly initialized CNN on only 60 demos is insufficient to learn reliable visual features.

### 3. SmolVLA achieves 60% grasp rate
SmolVLA's pretrained vision-language backbone provides much better visual understanding out of the box. With fine-tuning on 60 demos (20K steps, loss 0.147 -> 0.022), the model learned to reach toward oranges and partially close the gripper. However:
- Gripper never goes below ~34 degrees (needs ~2-5 degrees for firm grasp)
- The flow matching output clusters around the mean trajectory
- No temporal sequencing of grasp -> lift -> place

### 4. The postprocessor unnormalization bug
SmolVLA's `select_action()` returns **normalized** actions (mean-subtracted, std-divided). The postprocessor pipeline was a no-op — it did NOT unnormalize the actions. This caused near-zero joint commands until we manually applied:
```python
action_degrees = normalized_action * ACTION_STD + ACTION_MEAN
```

### 5. Autocast and denoising steps don't matter
Ablation tests showed:
- Removing `torch.autocast(fp16)` -> no improvement
- Increasing denoising steps 10 -> 50 -> no improvement  
- Scaling gripper channel 2x -> made things worse (amplified "open" not "close")

### 6. GR00T N1.5 is the intended pipeline
The LeIsaac framework was designed for GR00T fine-tuning, not SmolVLA or robomimic. The dataset format, eval scripts, and client-server architecture all target GR00T.

## Training Loss Curves

### SmolVLA Fine-tuning (20K steps)

```
Step   | Loss
-------|-------
   100 | 0.147
 1,000 | 0.079
 2,000 | 0.068
 4,000 | 0.049
 6,000 | 0.043
 8,000 | 0.033
10,000 | 0.031
12,000 | 0.028
14,000 | 0.024
16,000 | 0.021
18,000 | 0.023
20,000 | 0.022
```

Loss dropped 85% (0.147 -> 0.022) using flow matching objective. Plateaued around step 15K.

### BC-RNN-GMM on Original Dataset (200 epochs)

```
Epoch | Loss (NLL)
------|----------
    1 |  2.99
   20 | -5.59
   40 | -6.22
   80 | -7.75
  120 | -12.30
  160 | -13.25
  200 | -13.84
```

Negative log-likelihood (more negative = better for GMM). Converged well but model still failed at eval.

## Videos

### Ground Truth Training Demos
Actual recorded teleoperation demos from the LeIsaac dataset:
- `videos/ground-truth/gt_front_ep0-4.mp4` — Front camera view
- `videos/ground-truth/gt_wrist_ep0-4.mp4` — Wrist camera view

### SmolVLA Eval (with correct unnormalization)
5 episodes in PickOrange env, 60% grasp rate:
- `videos/smolvla-eval/smolvla_ep0-4.mp4`

### SmolVLA Ablation Tests
Testing autocast, denoising steps, and gripper scaling:
- `videos/ablation/testA_ep0-1.mp4` — No autocast
- `videos/ablation/testB_ep0-1.mp4` — 50 denoising steps
- `videos/ablation/testC_ep0-1.mp4` — Gripper scale 2x
- `videos/ablation/testD_ep0-1.mp4` — All combined

### BC-RNN + Vision Eval (epoch 33)
ResNet18 vision encoder, arm moves toward plate (wrong object):
- `videos/bc-vision/epoch033_ep0-1.mp4`

## Eval Metrics — SmolVLA

### Per-Episode Results

| Episode | Gripper (deg) | Orange 1 | Orange 2 | Orange 3 | Task |
|---------|--------------|----------|----------|----------|------|
| 0 | [34, 44] | no | — | no | no |
| 1 | [34, 44] | no | **GRASPED** (step 10) | no | no |
| 2 | [34, 44] | no | no | no | no |
| 3 | — | no | no | no | no |
| 4 | [34, 44] | no | no | no | no |

### Summary
- **Grasp rate**: 3/5 (60%)
- **Place rate**: 0/5 (0%)
- **Task completion**: 0/5 (0%)

### Ablation Results

| Test | Gripper (deg) | Grasps | Places |
|------|--------------|--------|--------|
| A: No autocast | [34, 44] | 1/6 | 0/6 |
| B: 50 denoise steps | [34, 44] | 1/6 | 0/6 |
| C: Grip scale 2x | [35, 60] | 0/6 | 0/6 |
| D: Combined | [35, 61] | 1/6 | 0/6 |

## Technical Details

### Dataset Units
- **Dataset (parquet)**: Degrees
- **Isaac Lab env**: Radians
- **Conversion**: `action_rad = action_deg * (pi / 180)`
- **SmolVLA normalization**: `normalized = (degrees - mean) / std`
- **At eval**: Must manually unnormalize then convert to radians

### Action Statistics (from training data, degrees)
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

### Gripper Convention
- **CLOSED**: 0.0 rad (0 deg)
- **OPEN**: 1.4 rad (80 deg)
- **Grasp threshold**: < 0.60 rad (34.4 deg) AND EE-to-orange < 0.05m

### Environment
- **Pod**: RunPod RTX 4090 (24GB VRAM)
- **Isaac Lab**: 2.3.2 (Isaac Sim 5.1, Python 3.11)
- **LeIsaac**: v0.1.2

## Repository Structure

```
.
├── README.md                    # This file
├── SMOLVLA_LEARNINGS.md         # Detailed SmolVLA training log
├── videos/
│   ├── ground-truth/            # Original training demo recordings
│   ├── smolvla-eval/            # SmolVLA evaluation episodes
│   ├── ablation/                # Ablation test videos
│   └── bc-vision/               # BC-RNN + ResNet18 eval
└── scripts/
    ├── train_smolvla.py         # SmolVLA fine-tuning script (patched)
    ├── eval_smolvla.py          # SmolVLA eval with metrics
    ├── eval_ablation.py         # Ablation test script
    └── convert_dataset.py       # Dataset conversion (LeRobot v2 -> robomimic HDF5)
```

## Next Steps

Pivoting to **GR00T N1.5** fine-tuning — the official approach for LeIsaac. See the main project repo for GR00T documentation.
