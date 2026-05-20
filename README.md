# Pi0 vs Pi0-FAST on LIBERO

CMPE 188 — Machine Learning, San José State University

| | |
|---|---|
| Authors | Kartik Reddy Katam, Gaurav Dharmadhikari |
| Emails | kartikreddy.katam@gmail.com, gaurav.dharrmadhikari@sjsu.edu |
| SJSU IDs | 015765542, 017038177 |

## What this is

Side-by-side LIBERO evaluation of two VLA architectures fine-tuned under identical conditions:

- **Pi0** — flow-matching, continuous actions (10 Euler steps per chunk)
- **Pi0-FAST** — autoregressive, FAST-tokenized discrete actions

Identical hyperparameters (5,000 LoRA steps, batch 32, seed 42), identical hardware (2× H100 SXM5 80GB on RunPod), identical data (`physical-intelligence/libero` HuggingFace dataset). Single experimental axis: continuous vs discrete action representation.

## Results

| Suite | Pi0 | Pi0-FAST | Δ |
|---|---|---|---|
| `libero_spatial` (n=30) | **46.7%** | 36.7% | Pi0 +10.0 |
| `libero_object` (n=30) | 50.0% | **80.0%** | Pi0-FAST +30.0 |
| `libero_goal` (n=30) | **60.0%** | 43.3% | Pi0 +16.7 |
| `libero_10` (n=20) | **5.0%** | 0.0% | Pi0 +5.0 |
| **Mean** (n=110) | **43.6%** | **43.6%** | 0.0 |

Architectures are complementary: Pi0 wins where motor control / trajectory shaping matters; Pi0-FAST wins where language → object grounding matters.

## Repo layout

```
results/
  figures/             6 PNGs (success rate, per-task heatmap, loss curves, delta)
  eval_logs/           per-suite per-model raw eval logs (8 files)
  results.json         machine-readable per-task success rates
  pi0_train.csv        per-step training loss/grad/param norms (54 rows)
  pi0fast_train.csv    same for Pi0-FAST
  wandb_runs_meta.json WandB run IDs + final state

videos/
  README.md            index of which videos to watch
  architectural_comparison/   10 stitched Pi0-vs-Pi0-FAST side-by-side MP4s
  pi0/{suite}/         59 raw Pi0 rollout videos
  pi0_fast/{suite}/    55 raw Pi0-FAST rollout videos

scripts/
  reproduction/        01_setup_pod.sh, 02_norm_stats.sh, 03_train.sh, 04_eval.sh, run-eval.sh
  analysis/            parse_results.py, extract_wandb.py, make_plots.py, stitch_videos.py

configs/
  hardware/h100.env    env vars used at training time
```

## Reproduction

```bash
# Provision 2× H100 SXM5 80GB on RunPod, ≥150 GB volume disk.
# Image: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

cd scripts/reproduction
./01_setup_pod.sh                 # apt + uv + openpi clone (pinned to e4580662)
# write HF token to /workspace/hf-token.env
./02_norm_stats.sh                # ~50 min (dataset download + compute)
./03_train.sh                     # ~1h45m, parallel on 2 GPUs
./04_eval.sh                      # ~2h, parallel on 2 GPUs (MUJOCO_GL=osmesa)

# Local analysis (after pulling /workspace/outputs back via rsync):
cd scripts/analysis
python parse_results.py           # → results.json
python extract_wandb.py           # → *_train.csv (needs wandb login)
python make_plots.py              # → figures/*.png
python stitch_videos.py --src ... --out ...
```

Pilot cost: $36 over 5.5 h.

## References

- [openpi (Physical Intelligence)](https://github.com/Physical-Intelligence/openpi) — Pi0 + Pi0-FAST source
- Black et al., *π₀: A VLA Flow Model for General Robot Control*, [arXiv:2410.24164](https://arxiv.org/abs/2410.24164)
- Pertsch et al., *FAST: Efficient Action Tokenization*, [arXiv:2501.09747](https://arxiv.org/abs/2501.09747)
- Liu et al., *LIBERO: Benchmarking Knowledge Transfer*, NeurIPS 2023, [arXiv:2306.03310](https://arxiv.org/abs/2306.03310)
