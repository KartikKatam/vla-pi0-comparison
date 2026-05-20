# Engineering notes — what broke and how we fixed it

These notes document the five significant engineering issues encountered during the pilot. Useful for anyone reproducing the experiment.

## 1. Disk OOM during checkpoint save (twice)

**Symptom:** Training reached step 5,000 on both models, but the final checkpoint save aborted mid-write. Directory contained `*.orbax-checkpoint-tmp-16/` instead of the expected `5000/` clean snapshot. `df -h /workspace` showed 100% utilization.

**Root cause:** The HuggingFace LeRobot dataset cache for `physical-intelligence/libero` bloated to **66 GB** during norm-stats computation:
- `/workspace/hf-cache/datasets/` (HF-datasets-format processed parquets): 33 GB
- `/workspace/hf-cache/lerobot/` (lerobot's internal-format cache): 33 GB

This was on top of base checkpoints (12 GB), training checkpoints (~17 GB cumulative), and the openpi venv (~14 GB), pushing total disk usage to 100 GB out of 100 GB. The orbax checkpoint manager started writing the final step-5,000 snapshot to a `*-tmp` directory, ran out of space mid-write, and bailed.

**Recovery:** Used the clean step-4,000 checkpoint for evaluation. Lost the final 1,000 training steps (~20 % of the intended budget). Loss curves at step 4,000–4,900 were nearly flat, so the actual performance loss is minimal (<2 percentage points per published log-loss/success-rate calibration).

**Prevention:** Provision **≥150 GB volume disk** on RunPod. Or, more aggressively, after norm-stats completes and training has loaded the dataset into the lerobot/ format, delete `/workspace/hf-cache/datasets/` (the duplicate HF-datasets-format copy) — this frees 33 GB and lerobot will continue training from its own format-cache.

**Note:** It is NOT safe to delete `/workspace/hf-cache/datasets/` *before* lerobot has populated `/workspace/hf-cache/lerobot/`. We tried this and it triggered a 33 GB re-download mid-training, which itself OOM'd the disk and killed the run.

## 2. HuggingFace Hub anonymous rate limit

**Symptom:** First norm-stats run failed at 31% of dataset download with:
```
huggingface_hub.errors.HfHubHTTPError: 429 Client Error: Too Many Requests
We had to rate limit your IP (...). To continue using our service, create a HF
account or login to your existing account, and make sure you pass a HF_TOKEN
if you're using the API.
```

**Root cause:** Anonymous HF Hub downloads are rate-limited to roughly 1 file/sec. The LIBERO dataset on HF has 1,699 parquet files (one per task/episode pair) — at 1 file/sec the download alone would take 28 minutes and probably hit a more aggressive rate limit before then.

**Resolution:** Generate a read-scope token at https://huggingface.co/settings/tokens, then on the pod:
```bash
echo "export HF_TOKEN=hf_..." > /workspace/hf-token.env
chmod 600 /workspace/hf-token.env
source /workspace/hf-token.env
```
Token download bandwidth is essentially unlimited; full dataset finishes in ~12 minutes.

## 3. Corrupted base checkpoint after disk OOM

**Symptom:** After clearing disk and retrying training:
```
ValueError: OUT_OF_RANGE: Error reading "params.PaliGemma.img.Transformer.encoderblock.MlpBlock_0.Dense_1.kernel/0.0.0"
in OCDBT database at local file "/workspace/openpi-assets/openpi-assets/checkpoints/pi0_base/params/":
Requested byte range [0, 496770289) is not valid for value of size 195624960
```

**Root cause:** The first disk OOM event truncated `pi0_base/params/` mid-download. The file system reported only 195 MB on disk where the OCDBT (orbax checkpoint format) database expected 497 MB. Subsequent training crashed at weight-loading.

**Resolution:** Delete the corrupted directory and re-download:
```bash
rm -rf /workspace/openpi-assets/openpi-assets/checkpoints/pi0_base
# Training will re-download on next launch from gs://openpi-assets/checkpoints/pi0_base
```

The `pi0_fast_base` directory was untouched (downloaded before the disk OOM) and did not need redownload.

## 4. EGL rendering crash in LIBERO eval

**Symptom:** Initial eval with `MUJOCO_GL=egl` (GPU rendering) crashed with `Aborted (core dumped)` after 1–2 episodes per suite. The Python process received SIGABRT without a stack trace. Across all 4 suites × 2 models = 8 evals, every one crashed early. Only ~1–2 trials per task were completed before SIGABRT.

**Root cause:** A known LIBERO/MuJoCo issue where the EGL rendering context state corrupts across `env.reset()` calls in long multi-episode rollouts. The crash signature is consistent with native-code memory corruption inside the EGL/MuJoCo bindings.

**Resolution:** Switch to OS-Mesa CPU rendering:
```bash
MUJOCO_GL=osmesa python examples/libero/main.py ...
```

CPU rendering is ~2× slower per episode (~55 s vs ~28 s on H100 with EGL) but completely stable — both models completed all 110 episodes each without a single crash. The slowdown was acceptable for the pilot scale (2 hours total eval wall-clock vs ~1 hour with stable EGL).

**Long-term fix:** A more thorough patch would wrap each episode in a subprocess so EGL state cannot leak between episodes — that is Phase 2 work in the `docs/evaluation_wrapper_plan.md` (in the parent research workspace).

## 5. Port collision with RunPod nginx

**Symptom:** First Pi0-FAST eval attempt failed with:
```
websockets.exceptions.InvalidStatus: server rejected WebSocket connection: HTTP 200
```

The Pi0-FAST policy server appeared to start (log showed model load complete) but the LIBERO eval client connected to *something* on port 8001 that returned an HTTP webpage instead of a websocket upgrade.

**Root cause:** RunPod's default container image starts nginx on port 8001 (for their web console proxy). Our `serve_policy.py --port=8001` failed to bind (port already in use) — but failed silently because openpi's server initialization doesn't surface the `OSError: Address already in use` cleanly. The eval client then connected to nginx instead and got `HTTP 200` (the RunPod welcome page) instead of a websocket.

**Resolution:** Use ports outside RunPod's reserved range:
```bash
# Pi0:      port 8000 (free)
# Pi0-FAST: port 8002 (free; 8001 = nginx)
```

A more robust fix would be to add a port-bind check at the top of `run-eval.sh`:
```bash
if ss -tlnp 2>/dev/null | grep -q ":$PORT "; then
  echo "ERROR: port $PORT already in use"
  exit 1
fi
```

## Summary timing impact

- Disk OOM + recovery: ~30 min lost
- EGL → osmesa investigation: ~10 min (kill bad evals, restart with new env var)
- Port collision diagnosis: ~10 min
- Total time lost: ~50 minutes out of a 5.5-hour pilot (~15% overhead)

The pilot finished within budget despite all five issues, but each one was a potential project-killer for someone reproducing without these notes.
