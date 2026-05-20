#!/usr/bin/env bash
# 03_train.sh — Train both Pi0 and Pi0-FAST LoRA in parallel on 2× H100.
# Each model trains for 5,000 steps with batch size 32, seed 42.
# Wall-clock: ~1h45m per model. Both run in parallel → ~1h45m total.
set -euo pipefail

source /workspace/env.sh
cd /workspace/openpi

# Important: clean up disk pressure before training (avoid mid-checkpoint OOM)
echo "=== Pre-flight: free disk pressure ==="
rm -rf /workspace/uv-cache 2>/dev/null
df -h /workspace | tail -1
# Expect ≥40 GB free here. If less, you have a problem.

# --- Pi0 LoRA on GPU 0 ---
echo "=== Launching Pi0 LoRA 5k on GPU 0 ==="
setsid bash -c '
  source /workspace/env.sh
  cd /workspace/openpi
  export CUDA_VISIBLE_DEVICES=0
  .venv/bin/python scripts/train.py pi0_libero_low_mem_finetune \
    --exp-name=pi0_pilot_5k_s42 \
    --num-train-steps=5000 \
    --batch-size=32 \
    --seed=42 \
    --checkpoint-base-dir=/workspace/checkpoints \
    --overwrite
' > /workspace/train-pi0.log 2>&1 < /dev/null &
PID_PI0=$!

sleep 5

# --- Pi0-FAST LoRA on GPU 1 ---
echo "=== Launching Pi0-FAST LoRA 5k on GPU 1 ==="
setsid bash -c '
  source /workspace/env.sh
  cd /workspace/openpi
  export CUDA_VISIBLE_DEVICES=1
  .venv/bin/python scripts/train.py pi0_fast_libero_low_mem_finetune \
    --exp-name=pi0fast_pilot_5k_s42 \
    --num-train-steps=5000 \
    --batch-size=32 \
    --seed=42 \
    --checkpoint-base-dir=/workspace/checkpoints \
    --overwrite
' > /workspace/train-pi0fast.log 2>&1 < /dev/null &
PID_FAST=$!

echo "Launched. Pi0 PID=$PID_PI0, Pi0-FAST PID=$PID_FAST"
echo "Monitor:"
echo "  watch -n 30 'tr \"\\r\" \"\\n\" < /workspace/train-pi0.log | grep \"Progress on\" | tail -3'"
echo ""
echo "Waiting for both training jobs to complete..."
wait
echo ""
echo "=== Both training jobs complete ==="
echo "Final checkpoints (use step-4000 if step-5000 save was OOM-killed):"
find /workspace/checkpoints -maxdepth 4 -mindepth 4 -type d | sort

# Free base checkpoints now that weights are loaded in trained checkpoints
rm -rf /workspace/openpi-assets/openpi-assets/checkpoints/pi0_base \
       /workspace/openpi-assets/openpi-assets/checkpoints/pi0_fast_base
echo "Cleaned base-checkpoint assets to free disk for eval."
