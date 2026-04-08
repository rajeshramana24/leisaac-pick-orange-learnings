#!/bin/bash
# GR00T N1.6 Pod Setup Script
# Run this on a fresh RunPod with Isaac Lab container
# Estimated time: ~15 minutes

set -e
echo "=== GR00T N1.6 Pod Setup ==="

# 1. Install GR00T (no-deps to protect Isaac Lab torch)
echo "[1/7] Cloning and installing GR00T..."
cd /data
git clone https://github.com/NVIDIA/Isaac-GR00T.git groot
cd groot
/workspace/isaaclab/_isaac_sim/kit/python/bin/python3 -m pip install -e . --no-deps

# 2. Install GR00T dependencies manually
echo "[2/7] Installing GR00T dependencies..."
/workspace/isaaclab/_isaac_sim/kit/python/bin/python3 -m pip install \
  albumentations==1.4.18 av==16.1.0 diffusers==0.35.1 dm-tree lmdb \
  timm tyro wandb==0.23.0 transformers==4.51.3 einops numpy==1.26.4 \
  decord msgpack

# 3. Install flash-attn from pre-built wheel
echo "[3/7] Installing flash-attn..."
/workspace/isaaclab/_isaac_sim/kit/python/bin/python3 -m pip install \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# 4. Uninstall deepspeed (no nvcc in container)
echo "[4/7] Removing deepspeed..."
/workspace/isaaclab/_isaac_sim/kit/python/bin/python3 -m pip uninstall deepspeed -y 2>/dev/null || true

# 5. Fix nvjitlink for Isaac Sim
echo "[5/7] Fixing nvjitlink..."
/workspace/isaaclab/_isaac_sim/kit/python/bin/python3 -m pip install nvidia-nvjitlink-cu12==12.8.61
cp /isaac-sim/kit/python/lib/python3.11/site-packages/nvidia/nvjitlink/lib/libnvJitLink.so.12 \
   /isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/nvidia/nvjitlink/lib/libnvJitLink.so.12

# 6. Fix libcusparseLt symlink
echo "[6/7] Fixing libcusparseLt..."
CUSPARSELT=$(find /isaac-sim -name "libcusparseLt.so.0" 2>/dev/null | head -1)
if [ -n "$CUSPARSELT" ]; then
    ln -sf "$CUSPARSELT" /usr/lib/x86_64-linux-gnu/libcusparseLt.so.0
fi

# 7. Install ffmpeg and tmux
echo "[7/7] Installing system packages..."
apt-get update -qq && apt-get install -y -qq ffmpeg tmux

# Setup Xvfb
pgrep Xvfb || (Xvfb :99 -screen 0 1024x768x24 &>/dev/null &)
export DISPLAY=:99

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Download trained model:  huggingface-cli download rajeshramana/groot-n1.6-pick-orange --local-dir /data/groot_final/checkpoint"
echo "  2. Download dataset:        huggingface-cli download rajeshramana/leisaac-pick-orange-prepared --local-dir /data/groot/demo_data/pick_orange --repo-type dataset"
echo "  3. Copy config files (see below)"
echo "  4. Start training or eval"
