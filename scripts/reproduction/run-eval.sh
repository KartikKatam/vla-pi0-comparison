#!/usr/bin/env bash
# run-eval.sh — start a policy server + run LIBERO eval client for one model.
# Usage: run-eval.sh <model> <gpu> <port> <ckpt-dir>
#
# Example:
#   ./run-eval.sh pi0 0 8000 /workspace/checkpoints/pi0_libero_low_mem_finetune/pi0_pilot_5k_s42/4000
#
# This is the script we actually used in the pilot. It runs each suite sequentially
# (3 trials/task, 2 for libero_10), with osmesa CPU rendering to avoid the known
# EGL crash on multi-episode rollouts.
set -uo pipefail

MODEL=$1; GPU=$2; PORT=$3; CKPT_DIR=$4

source /workspace/env.sh
cd /workspace/openpi

case $MODEL in
  pi0) CONFIG=pi0_libero_low_mem_finetune ;;
  pi0_fast) CONFIG=pi0_fast_libero_low_mem_finetune ;;
  *) echo "bad model: $MODEL"; exit 1 ;;
esac

OUTPUT_BASE=/workspace/outputs/eval/$MODEL
mkdir -p $OUTPUT_BASE

EVAL_PYTHONPATH=/workspace/openpi:/workspace/openpi/packages/openpi-client/src:/workspace/openpi/third_party/libero

echo "=== $MODEL EVAL START $(date -u +%H:%M:%SZ) on GPU $GPU port $PORT ==="

# 1. Start policy server
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python scripts/serve_policy.py \
  --port=$PORT \
  policy:checkpoint \
    --policy.config=$CONFIG \
    --policy.dir=$CKPT_DIR > $OUTPUT_BASE/server.log 2>&1 &
SERVER_PID=$!

# 2. Wait for server to come up
for i in {1..120}; do
  sleep 5
  if (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null; then break; fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then echo "server died"; tail -20 $OUTPUT_BASE/server.log; exit 2; fi
done

# Per-suite trial counts (chosen to fit in time budget)
declare -A TRIALS=( [libero_spatial]=3 [libero_object]=3 [libero_goal]=3 [libero_10]=2 )

# 3. Run eval per suite — flags are --args.NAME (nested)
for suite in libero_spatial libero_object libero_goal libero_10; do
  out_dir=$OUTPUT_BASE/$suite
  mkdir -p $out_dir
  n_trials=${TRIALS[$suite]}
  echo ""
  echo "--- $MODEL on $suite ($(date -u +%H:%M:%SZ)) trials=$n_trials ---"
  CUDA_VISIBLE_DEVICES=$GPU \
  MUJOCO_GL=osmesa \
  PYTHONPATH=$EVAL_PYTHONPATH \
  LIBERO_CONFIG_PATH=/tmp/libero \
  .venv-libero/bin/python examples/libero/main.py \
    --args.host=127.0.0.1 \
    --args.port=$PORT \
    --args.task-suite-name=$suite \
    --args.num-trials-per-task=$n_trials \
    --args.replan-steps=5 \
    --args.seed=7 \
    --args.video-out-path=$out_dir/videos 2>&1 | tee $out_dir/eval.log
done

# 4. Tear down server
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null

echo "=== $MODEL EVAL END $(date -u +%H:%M:%SZ) ==="
