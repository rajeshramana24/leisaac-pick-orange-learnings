# SmolVLA Fine-Tuning on LeIsaac PickOrange: Detailed Learnings

## Overview

This document captures hard-won learnings from fine-tuning SmolVLA (450M param VLA) on the LeIsaac pick-orange task using an RTX 4090 (24GB) RunPod instance. The work progressed through three model architectures before concluding that GR00T N1.5 is the correct approach for this framework.

## Timeline

1. **BC-RNN-GMM (joint-only)** — trained on robomimic, 0% success
2. **BC-RNN-GMM + ResNet18 vision** — 60% grasp at epoch 33 but wrong target
3. **SmolVLA fine-tune** — 60% grasp rate, 0% placement
4. **GR00T N1.5** — next step (official LeIsaac pipeline)

---

## Phase 1: BC-RNN-GMM (No Vision)

### Setup
- Framework: robomimic 0.4.0
- Model: BC-RNN with GMM (5 modes), LSTM (2 layers, 400 hidden)
- Observations: `joint_pos (6,)` + `joint_vel (6,)` — **no camera images**
- Actions: 6D joint positions (radians)
- Dataset: 60 episodes converted from LeRobot parquet to robomimic HDF5
- Training: 200 epochs, batch_size=128, lr=1e-4

### Results
- **Loss**: 3.49 -> -13.84 NLL (excellent convergence)
- **Eval**: 0% success — gripper outputs ~0.9 rad every step (stuck open)

### Root Cause: Distribution Shift
The BC-RNN learns the training distribution well, but during closed-loop rollout:
1. First action slightly differs from training data
2. Next observation differs from any training observation
3. RNN hidden state drifts
4. GMM collapses to dominant mode (gripper open, ~0.9 rad)
5. Robot freezes with gripper open

This is a fundamental limitation of BC without vision — the model has no way to correct course from visual feedback.

### Key Learning
> **Joint-only BC with GMM is fundamentally unsuited for manipulation tasks.** Without vision, the model memorizes a blind trajectory and cannot recover from any deviation. The GMM's 5 modes collapse to one during rollout.

---

## Phase 2: BC-RNN-GMM + Vision (ResNet18)

### Setup
- Same robomimic framework
- Added ResNet18 vision encoder (randomly initialized, NOT pretrained)
- Image size: 84x84 (downsampled from 640x480)
- SpatialSoftmax pooling -> 64-dim features
- Total params: 13.4M (all trainable)
- Training: 300 epochs with eval every 33 epochs

### Dataset Conversion Issues

**Problem 1: Images stored as HWC in HDF5**
- robomimic's `process_frame` expects HWC input, converts to CHW internally
- But `obs_key_shapes` must report CHW (the processed shape, not raw)
- Fix: Register shapes as `(3, 84, 84)` while storing `(84, 84, 3)` in HDF5

**Problem 2: Dataset unit conversion**
- Original LeRobot parquet stores actions in **degrees**
- Previous robomimic HDF5 had already been converted to **radians**
- When re-converting from parquet, must apply `* (pi / 180)`

**Problem 3: joint_vel not in parquet**
- Parquet only has `observation.state` (joint positions)
- Must compute velocity via finite differences: `vel[t] = (pos[t] - pos[t-1]) * fps`

### Results at Epoch 33
- Loss: 2.99 -> -8.35 (still improving)
- **Gripper range**: [0.38, 0.95] rad — narrower than joint-only model (progress!)
- **Behavior**: Arm moved toward the **plate** instead of the oranges

### Root Cause
Randomly initialized ResNet18 with only 60 demo episodes cannot learn to distinguish oranges from plates. The larger visual feature (plate) dominates attention.

### Key Learning
> **Training a vision encoder from scratch on 60 demos is insufficient.** Use pretrained backbones (ImageNet or manipulation-specific) to get useful visual features without needing thousands of demos.

---

## Phase 3: SmolVLA Fine-Tuning

### Setup
- Model: SmolVLA base (lerobot/smolvla_base, 450M params)
  - SmolVLM2-500M-Video-Instruct backbone
  - Flow matching action decoder
  - 100M learnable params (backbone partially frozen)
- Dataset: LightwheelAI/leisaac-pick-orange (v3.0 format)
- Camera remapping: `front -> camera1`, `wrist -> camera2`
- Training: 20K steps, batch_size=8, ~2hrs on RTX 4090

### Installation Challenges

**1. LeRobot v0.4.4 requires Python 3.12, Isaac Lab has 3.11**
- Solution: Checkout `v0.4.4` tag (last to support 3.11)
- Install from source: `pip install -e .`

**2. Dataset version mismatch**
- Local dataset was v2.1, lerobot v0.4.4 code expects v3.0
- Conversion: `python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 --repo-id=...`
- But the converter queries HuggingFace Hub which has v2.1

**3. Hub version check blocks local training**
- `get_safe_version()` queries Hub even for local datasets
- Fix: Monkey-patch `get_safe_version` to return local version on exception
- Also monkey-patch `snapshot_download` to use `local_files_only=True`

**4. pip/packaging corruption**
- Installing lerobot's torch 2.10.0 overwrote Isaac Lab's torch 2.7.0+cu126
- Corrupted `pip._vendor.packaging._structures` and `torch._vendor.packaging._structures`
- Fix: `rm -rf pip*` + `get-pip.py`, manually create `_structures.py`
- Restore: `pip install numpy<2 torch==2.7.0+cu126`

### Training

```
Step   | Loss  | Notes
-------|-------|------------------
   100 | 0.147 | Rapid learning
 1,000 | 0.079 | -46% from start
 2,000 | 0.068 |
 5,000 | 0.046 |
10,000 | 0.031 | 50% done
15,000 | 0.023 | Plateau begins
20,000 | 0.022 | Final (85% reduction)
```

Training was smooth once the patches were in place. ~2.7 steps/sec, checkpoints every 2K steps.

### Eval Pipeline — The Hard Part

Getting SmolVLA inference to work inside Isaac Lab required solving 6 issues:

**Issue 1: GPU OOM**
- SmolVLA (1.8GB bfloat16) + PickOrange scene (~18GB) > 24GB
- Fix: Use `torch.autocast("cuda", dtype=torch.float16)` for inference

**Issue 2: Image dtype**
- `F.interpolate` in SmolVLA's `resize_with_pad` doesn't support uint8
- Fix: Pass images as `torch.float32`

**Issue 3: Language tokenization**
- `select_action()` expects pre-tokenized language tokens, not raw strings
- Fix: Use `make_pre_post_processors(policy.config)` to create the preprocessor pipeline

**Issue 4: Camera key naming**
- Preprocessor's rename step (`front -> camera1`) didn't fire during eval
- Fix: Pass `camera1/camera2` keys directly, skip rename

**Issue 5: Autocast dtype mismatch**
- `policy.half()` converts weights to fp16 but preprocessor outputs fp32 state
- Fix: Don't use `.half()`, use `torch.autocast` context manager instead

**Issue 6: Postprocessor is a no-op**
- `postprocessor(action)` returns the SAME tensor — no unnormalization applied
- The model outputs **normalized** values (mean=0, std=1)
- Must manually unnormalize: `degrees = normalized * std + mean`
- Action stats from training data:
  ```
  mean = [8.86, 2.97, 2.25, 76.58, 18.28, 32.57]
  std  = [19.57, 39.63, 41.97, 13.67, 9.61, 12.37]
  ```

### Results

| Metric | Value |
|--------|-------|
| Grasp rate | **3/5 (60%)** |
| Place rate | 0/5 (0%) |
| Task completion | 0/5 (0%) |
| Gripper range | 34-44 deg (0.59-0.77 rad) |

The gripper hovers near the mean (32.6 deg) and barely dips below the grasp threshold (34.4 deg). It never goes to the ~2-5 deg needed for a firm grasp.

### Ablation Tests

| Test | Change | Gripper (deg) | Grasps |
|------|--------|--------------|--------|
| Baseline | autocast, 10 steps | [34, 44] | 3/5 |
| A: No autocast | full precision | [34, 44] | 1/6 |
| B: 50 denoise | 5x more steps | [34, 44] | 1/6 |
| C: Grip 2x | scale gripper | [35, 60] | 0/6 |
| D: Combined | all above | [35, 61] | 1/6 |

**Conclusion**: The issue is NOT inference precision or denoising quality. The model genuinely outputs mean-like actions for the gripper.

### Why SmolVLA Underperforms on This Task

1. **SmolVLA was trained on SO-100, not SO-101** — different kinematic chain
2. **60 demos may not be enough** for the flow matching decoder to learn the full bimodal gripper distribution (open vs closed)
3. **Action chunking (50 steps)** means the model predicts 1.7 seconds of actions at once — if the initial prediction is "hover near mean", all 50 actions stay there
4. **The postprocessor bug** wasted significant debugging time
5. **LeIsaac was designed for GR00T, not SmolVLA** — the eval pipeline, dataset format, and action processing all assume GR00T's client-server architecture

---

## Technical Gotchas Reference

### Dataset Units
```
Parquet (training data) -> DEGREES
Isaac Lab env.step()    -> RADIANS
SmolVLA normalized      -> MEAN_STD normalized (dimensionless)

Eval pipeline:
  env_obs (rad) -> * RAD2DEG -> SmolVLA input (deg, then normalized by preprocessor)
  SmolVLA output (normalized) -> * STD + MEAN -> degrees -> * DEG2RAD -> env.step (rad)
```

### Isaac Lab + lerobot Compatibility
- **DO NOT** `pip install lerobot` into Isaac Lab's python — it breaks torch/numpy
- If you must, restore: `pip install numpy<2`, `pip install torch==2.7.0 --index-url cu126`
- Fix `_structures.py` at `/isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/torch/_vendor/packaging/`

### RunPod GPU Memory
- Isaac Lab PickOrange (kitchen scene + 2 cameras): ~18-20 GB
- SmolVLA inference (450M bfloat16): ~1.8 GB
- Combined: Barely fits in 24GB — use autocast or fp16
- Always `pkill -9 -f python` before new runs to clear zombie processes

---

## Conclusion

SmolVLA fine-tuning on 60 demos achieved 60% grasp rate — a significant improvement over blind BC (0%). However, the gripper never closes firmly enough for reliable manipulation. The LeIsaac framework is designed for GR00T N1.5, which is the recommended next step.
