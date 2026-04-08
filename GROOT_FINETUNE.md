# GR00T N1.6 Fine-tuning for PickOrange — Pod Setup & Restore Guide

**READ THIS FIRST** when starting a new session. This file contains everything needed to restore the GR00T training/eval environment on a new RunPod.

## Current Status (2026-04-08)

- **Training**: COMPLETE — 10K steps, final loss 0.017
- **Eval**: 100% pick+place on 1st orange (3/3 episodes), heading for 2nd at timeout
- **Model on HF**: https://huggingface.co/rajeshramana/groot-n1.6-pick-orange
- **Dataset on HF**: https://huggingface.co/datasets/rajeshramana/leisaac-pick-orange-prepared
- **Eval videos**: https://github.com/rajeshramana24/leisaac-pick-orange-learnings/tree/main/videos/groot_eval_final_10k
- **All learnings**: https://github.com/rajeshramana24/leisaac-pick-orange-learnings

## Pod Requirements

- **GPU**: RTX 4090 (24GB VRAM) or better
- **Container**: NVIDIA Isaac Lab (Isaac Sim 5.1, Python 3.11)
- **Template**: Use the same NGC Isaac Lab template as before
- **Volume**: Mount network storage at `/data` (need ~20GB free)

## Restore on New Pod (step by step)

### Step 1: Run Setup Script (~10 min)

The setup script is saved locally at `isaac/pod_restore/setup_new_pod.sh`. Upload and run:

```bash
# From your local machine, copy the script to the pod:
# (adapt SSH details for new pod)
ssh NEW_POD_SSH "cat > /data/setup_new_pod.sh" < ~/Desktop/Robotics/isaac/pod_restore/setup_new_pod.sh
ssh NEW_POD_SSH "bash /data/setup_new_pod.sh"
```

Or run these commands manually on the pod:

```bash
# 1. Clone and install GR00T (no-deps!)
cd /data
git clone https://github.com/NVIDIA/Isaac-GR00T.git groot
cd groot
/workspace/isaaclab/_isaac_sim/kit/python/bin/python3 -m pip install -e . --no-deps

# 2. Install dependencies manually
/workspace/isaaclab/_isaac_sim/kit/python/bin/python3 -m pip install \
  albumentations==1.4.18 av==16.1.0 diffusers==0.35.1 dm-tree lmdb \
  timm tyro wandb==0.23.0 transformers==4.51.3 einops numpy==1.26.4 \
  decord msgpack

# 3. Install flash-attn (pre-built wheel, no nvcc needed)
/workspace/isaaclab/_isaac_sim/kit/python/bin/python3 -m pip install \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# 4. Remove deepspeed (causes CUDA_HOME errors)
/workspace/isaaclab/_isaac_sim/kit/python/bin/python3 -m pip uninstall deepspeed -y

# 5. Fix nvjitlink for Isaac Sim
/workspace/isaaclab/_isaac_sim/kit/python/bin/python3 -m pip install nvidia-nvjitlink-cu12==12.8.61
cp /isaac-sim/kit/python/lib/python3.11/site-packages/nvidia/nvjitlink/lib/libnvJitLink.so.12 \
   /isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/nvidia/nvjitlink/lib/libnvJitLink.so.12

# 6. Fix libcusparseLt symlink
CUSPARSELT=$(find /isaac-sim -name "libcusparseLt.so.0" 2>/dev/null | head -1)
[ -n "$CUSPARSELT" ] && ln -sf "$CUSPARSELT" /usr/lib/x86_64-linux-gnu/libcusparseLt.so.0

# 7. System packages
apt-get update -qq && apt-get install -y -qq ffmpeg tmux

# 8. Xvfb for headless rendering
Xvfb :99 -screen 0 1024x768x24 &>/dev/null &
export DISPLAY=:99
```

### Step 2: Download Model & Dataset from HuggingFace (~5 min)

```bash
# Install huggingface CLI
/workspace/isaaclab/_isaac_sim/kit/python/bin/python3 -m pip install huggingface_hub

# Download trained model (14GB)
huggingface-cli download rajeshramana/groot-n1.6-pick-orange \
  --local-dir /data/groot_final/checkpoint

# Download prepared dataset (1.6GB)
huggingface-cli download rajeshramana/leisaac-pick-orange-prepared \
  --local-dir /data/groot/demo_data/pick_orange --repo-type dataset
```

### Step 3: Copy Config Files

The modality config is saved locally at `isaac/pod_restore/so101_pick_orange_config.py`.

```bash
# Copy from local machine to pod:
ssh NEW_POD_SSH "cat > /data/groot/so101_pick_orange_config.py" < ~/Desktop/Robotics/isaac/pod_restore/so101_pick_orange_config.py
```

Or create it on the pod — see `isaac/pod_restore/so101_pick_orange_config.py` for contents.

The `modality.json` is already included in the HF dataset download.

### Step 4: Verify Everything Works

```bash
# Test torch in Isaac Lab context
cd /workspace/isaaclab && DISPLAY=:99 ./isaaclab.sh -p -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Should print: 2.7.0+cu126 True

# Test GR00T import
/workspace/isaaclab/_isaac_sim/kit/python/bin/python3 -c "from gr00t.policy.gr00t_policy import Gr00tPolicy; print('GR00T OK')"
```

## Running Eval (with video recording)

Eval uses GR00T's client-server architecture. Two tmux sessions needed:

### Terminal 1 — Policy Server

```bash
tmux new-session -d -s server
tmux send-keys -t server 'cd /data/groot && PYTHONUNBUFFERED=1 /workspace/isaaclab/_isaac_sim/kit/python/bin/python3 gr00t/eval/run_gr00t_server.py --model_path /data/groot_final/checkpoint --embodiment_tag NEW_EMBODIMENT --port 5555 --use_sim_policy_wrapper 2>&1 | tee /data/groot_server.log' Enter

# Wait ~60s for "Server is ready and listening on tcp://0.0.0.0:5555"
sleep 60 && grep "Server is ready" /data/groot_server.log
```

### Terminal 2 — Eval Client (with video)

Upload the eval script from GitHub repo (`scripts/eval_groot_3ep.py`) or from local `pod_restore/`, then:

```bash
tmux new-session -d -s eval
tmux send-keys -t eval 'export DISPLAY=:99 && cd /workspace/isaaclab && PYTHONUNBUFFERED=1 /workspace/isaaclab/_isaac_sim/python.sh /data/groot_eval_3ep.py > /data/groot_eval_output.log 2>&1' Enter

# Monitor progress:
tail -f /data/groot_eval_output.log
```

Videos saved to `/data/groot_eval_final/episode_*_{front,wrist}.mp4`

## Running Further Training

```bash
tmux new-session -d -s train
tmux send-keys -t train 'cd /data/groot && PYTHONUNBUFFERED=1 /workspace/isaaclab/_isaac_sim/kit/python/bin/python3 gr00t/experiment/launch_finetune.py --base-model-path /data/groot_final/checkpoint --dataset-path /data/groot/demo_data/pick_orange --modality-config-path /data/groot/so101_pick_orange_config.py --embodiment-tag NEW_EMBODIMENT --num-gpus 1 --output-dir /data/groot_new_checkpoints --save-steps 2000 --save-total-limit 2 --max-steps 5000 --warmup-ratio 0.05 --weight-decay 1e-5 --learning-rate 1e-5 --global-batch-size 8 --dataloader-num-workers 2 --no-tune-diffusion-model 2>&1 | tee /data/groot_train.log' Enter
```

**IMPORTANT**: Training uses ~24GB VRAM. Checkpoint saves can cause OOM crashes. Use `--save-total-limit 2` and monitor with heartbeat script.

## CRITICAL Learnings (don't repeat these mistakes)

1. **GR00T install MUST use `--no-deps`** — otherwise it replaces Isaac Lab's torch 2.7.0+cu126
2. **Use N1.6 not N1.5** — N1.5 isn't registered in AutoModel
3. **`--use_sim_policy_wrapper`** is REQUIRED on the server for flat observation format
4. **GR00T server uses msgpack**, LeIsaac client uses torch pickle — they're INCOMPATIBLE. Use GR00T's native `PolicyClient`
5. **Observations need temporal dim T=1**: video `(B,T,H,W,C)`, state `(B,T,D)`, dtype `float32` not float64
6. **Action horizon is 16**: server returns `(B,16,6)`, iterate over dim 1 for env.step
7. **No `--resume-from-checkpoint`** flag — use `--base-model-path` pointing to previous checkpoint
8. **Checkpoint saves OOM at 24GB** — use `--save-total-limit 2`, clean disk aggressively
9. **nvjitlink must be copied** to pip_prebundle after install: `cp .../libnvJitLink.so.12 .../pip_prebundle/nvidia/nvjitlink/lib/`
10. **Videos must be H.264** (not AV1) for decord. Dataset on HF is already converted.
11. **`torch.inference_mode()` breaks `env.reset()`** — don't wrap the eval loop in it
12. **RunPod /data has per-pod quotas** — even though df shows TB free, writes fail silently

## Local Files Reference

```
Robotics/isaac/
├── GROOT_FINETUNE.md              ← THIS FILE (pod restore guide)
├── LEARNINGS.md                   ← Original Isaac Lab learnings
├── eval_videos/                   ← Downloaded eval videos
│   ├── groot_eval_ckpt3000_*.mp4
│   ├── groot_eval_ckpt7000_*.mp4
│   └── (final videos on GitHub)
└── pod_restore/                   ← Files needed to restore a pod
    ├── setup_new_pod.sh           ← Automated setup script
    ├── so101_pick_orange_config.py ← Modality config for training
    └── modality.json              ← Dataset modality mapping
```

## HuggingFace Artifacts

| Artifact | URL | Size |
|----------|-----|------|
| Trained model (10K steps) | https://huggingface.co/rajeshramana/groot-n1.6-pick-orange | 14GB |
| Prepared dataset | https://huggingface.co/datasets/rajeshramana/leisaac-pick-orange-prepared | 1.6GB |

## GitHub

| Repo | URL |
|------|-----|
| Learnings + eval videos | https://github.com/rajeshramana24/leisaac-pick-orange-learnings |
