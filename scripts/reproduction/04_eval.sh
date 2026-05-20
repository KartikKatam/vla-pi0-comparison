#!/usr/bin/env bash
# 04_eval.sh — Evaluate both trained models in parallel on all 4 LIBERO suites.
# Pi0 on GPU 0 / port 8000, Pi0-FAST on GPU 1 / port 8002 (avoid nginx port 8001).
# Wall-clock: ~2h total with MUJOCO_GL=osmesa.
set -euo pipefail

source /workspace/env.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use the step-4000 checkpoint (final-step save was OOM-killed in our pilot;
# adjust if your run completed cleanly).
PI0_CKPT=/workspace/checkpoints/pi0_libero_low_mem_finetune/pi0_pilot_5k_s42/4000
PI0FAST_CKPT=/workspace/checkpoints/pi0_fast_libero_low_mem_finetune/pi0fast_pilot_5k_s42/4000

if [ ! -d "$PI0_CKPT" ] || [ ! -d "$PI0FAST_CKPT" ]; then
  echo "ERROR: missing checkpoints"; ls -la /workspace/checkpoints; exit 1
fi

echo "=== Launching Pi0-FAST eval on GPU 1, port 8002 (background) ==="
setsid "$SCRIPT_DIR/run-eval.sh" pi0_fast 1 8002 "$PI0FAST_CKPT" \
  > /workspace/eval-pi0fast.log 2>&1 < /dev/null &
PID_FAST=$!

sleep 3

echo "=== Launching Pi0 eval on GPU 0, port 8000 (background) ==="
setsid "$SCRIPT_DIR/run-eval.sh" pi0 0 8000 "$PI0_CKPT" \
  > /workspace/eval-pi0.log 2>&1 < /dev/null &
PID_PI0=$!

echo "Launched. Pi0 eval PID=$PID_PI0, Pi0-FAST eval PID=$PID_FAST"
echo "Monitor:"
echo "  watch -n 30 'tr \"\\r\" \"\\n\" < /workspace/eval-pi0.log | grep \"successes\" | tail -3'"

wait
echo ""
echo "=== Both eval pipelines complete ==="
echo "Outputs:"
find /workspace/outputs/eval -maxdepth 3 -name 'eval.log' | sort
