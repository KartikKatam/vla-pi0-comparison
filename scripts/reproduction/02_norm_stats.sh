#!/usr/bin/env bash
# 02_norm_stats.sh — compute action-normalization statistics for both configs.
# Downloads the LIBERO LeRobot dataset (~45 GB) as a side effect.
# Run in parallel on the 2× H100 pod after 01_setup_pod.sh.
set -euo pipefail

source /workspace/env.sh
cd /workspace/openpi

echo "=== norm stats: pi0_libero_low_mem_finetune (background) ==="
setsid bash -c '
  source /workspace/env.sh
  cd /workspace/openpi
  .venv/bin/python scripts/compute_norm_stats.py --config-name=pi0_libero_low_mem_finetune
' > /workspace/norm-stats-pi0.log 2>&1 < /dev/null &
PID_PI0=$!

echo "=== norm stats: pi0_fast_libero_low_mem_finetune (background) ==="
setsid bash -c '
  source /workspace/env.sh
  cd /workspace/openpi
  .venv/bin/python scripts/compute_norm_stats.py --config-name=pi0_fast_libero_low_mem_finetune
' > /workspace/norm-stats-pi0fast.log 2>&1 < /dev/null &
PID_FAST=$!

echo "Launched. Pi0 PID=$PID_PI0, Pi0-FAST PID=$PID_FAST"
echo "Tail logs:"
echo "  tail -f /workspace/norm-stats-pi0.log"
echo "  tail -f /workspace/norm-stats-pi0fast.log"
echo ""
echo "Both should finish in ~40-50 min (CPU-bound on 32-core machine; data download ~12 min + stats compute ~38 min)"
wait
echo "=== Both norm-stats jobs complete ==="
ls -la /workspace/openpi/assets/*/physical-intelligence/libero/norm_stats.json
