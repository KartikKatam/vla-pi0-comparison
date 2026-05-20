#!/usr/bin/env bash
# 01_setup_pod.sh — install prerequisites, clone openpi, set up workspace env
# Run inside a fresh RunPod 2× H100 SXM5 80GB pod (image: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04)
set -euo pipefail

echo "=== 1. apt prerequisites for openpi + LIBERO ==="
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  git git-lfs build-essential clang curl ca-certificates \
  libegl1 libgl1 libglib2.0-0 libgles2 libglfw3 libglew2.2 libosmesa6 \
  libsm6 libxext6 libxrender1 \
  rsync zstd

echo "=== 2. Install uv (modern Python project manager) ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"

echo "=== 3. Set workspace env vars ==="
cat > /workspace/env.sh <<'ENVSH'
export HF_HOME=/workspace/hf-cache
export HF_HUB_CACHE=/workspace/hf-cache/hub
export OPENPI_DATA_HOME=/workspace/openpi-assets
export UV_CACHE_DIR=/workspace/uv-cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export MUJOCO_GL=egl
export WANDB_PROJECT=libero-vla-analysis
export PATH=/root/.local/bin:$PATH
[ -f /workspace/hf-token.env ] && source /workspace/hf-token.env
ENVSH
chmod +x /workspace/env.sh
source /workspace/env.sh

# Symlink container-disk cache locations to volume disk to avoid OOM on container disk
mkdir -p /workspace/hf-cache /workspace/openpi-assets /workspace/uv-cache \
         /workspace/checkpoints /workspace/outputs /workspace/assets
ln -sfn /workspace/hf-cache /root/.cache/huggingface
ln -sfn /workspace/openpi-assets /root/.cache/openpi

echo "=== 4. Clone openpi at pinned commit e4580662 ==="
cd /workspace
GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git openpi
cd /workspace/openpi
git checkout e4580662
git submodule update --init --recursive

echo "=== 5. uv sync openpi training env (Python 3.11) ==="
GIT_LFS_SKIP_SMUDGE=1 uv sync --no-dev

echo "=== 6. Build LIBERO eval venv (Python 3.8 — mujoco-py requirement) ==="
uv venv --python 3.8 .venv-libero
uv pip sync \
  examples/libero/requirements.txt \
  third_party/libero/requirements.txt \
  packages/openpi-client/pyproject.toml \
  --python .venv-libero/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cu113 \
  --index-strategy=unsafe-best-match

echo "=== 7. Write LIBERO config ==="
mkdir -p /tmp/libero
cat > /tmp/libero/config.yaml <<YML
benchmark_root: /workspace/openpi/third_party/libero/libero/libero
bddl_files: /workspace/openpi/third_party/libero/libero/libero/bddl_files
init_states: /workspace/openpi/third_party/libero/libero/libero/init_files
datasets: /workspace/openpi/third_party/libero/libero/datasets
assets: /workspace/openpi/third_party/libero/libero/libero/assets
YML

echo ""
echo "=== Setup complete ==="
echo "Next: write your HF token to /workspace/hf-token.env (export HF_TOKEN=hf_...),"
echo "then run 02_norm_stats.sh"
