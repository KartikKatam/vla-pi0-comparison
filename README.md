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

Identical hyperparameters (5,000 LoRA steps, batch 32, seed 42), identical hardware (2× H100 SXM5 80GB on RunPod), identical data. Single experimental axis: continuous vs discrete action representation.

## Results

| Suite | Pi0 | Pi0-FAST | Δ |
|---|---|---|---|
| `libero_spatial` (n=30) | **46.7%** | 36.7% | Pi0 +10.0 |
| `libero_object` (n=30) | 50.0% | **80.0%** | Pi0-FAST +30.0 |
| `libero_goal` (n=30) | **60.0%** | 43.3% | Pi0 +16.7 |
| `libero_10` (n=20) | **5.0%** | 0.0% | Pi0 +5.0 |
| **Mean** (n=110) | **43.6%** | **43.6%** | 0.0 |

Architectures are complementary: Pi0 wins where motor control / trajectory shaping matters; Pi0-FAST wins where language → object grounding matters.

![Per-suite success rate](results/figures/fig01_success_per_suite.png)

![Per-task heatmap (40 tasks × 2 models)](results/figures/fig02_per_task_heatmap.png)

![Head-to-head per-task delta](results/figures/fig06_per_task_delta.png)

![Training loss curves](results/figures/fig03_training_loss.png)

## Videos to watch first

These three side-by-side comparisons compress the architectural story into ~90 seconds total:

1. **[01_orange_juice](videos/architectural_comparison/01_language_grounding_win_fast_orange_juice.mp4)** — Pi0 0/3 vs Pi0-FAST 3/3 on "pick up the orange juice." Strongest language-grounding signal.
2. **[04_push_plate](videos/architectural_comparison/04_motor_control_win_pi0_push_the_plate_to_the_front_of_the_stove.mp4)** — Pi0 3/3 vs Pi0-FAST 0/3 on "push the plate to the front of the stove." Continuous force modulation wins.
3. **[09_wooden_cabinet](videos/architectural_comparison/09_both_fail_black_bowl_on_the_wooden_cabinet.mp4)** — Both 0/3. Shared visual-grounding failure (not architecture-specific).

Full list of 10 stitched comparisons: [`videos/README.md`](videos/README.md).

## Finished features → evidence

| Feature | Evidence |
|---|---|
| Pi0 LoRA fine-tuned 4k steps | [`results/pi0_train.csv`](results/pi0_train.csv), loss curve `fig03` |
| Pi0-FAST LoRA fine-tuned 4k steps | [`results/pi0fast_train.csv`](results/pi0fast_train.csv), loss curve `fig03` |
| Side-by-side eval pipeline | 114 raw rollout videos in [`videos/pi0/`](videos/pi0/) and [`videos/pi0_fast/`](videos/pi0_fast/) |
| Quantitative eval, 4 LIBERO suites | [`results/results.json`](results/results.json), `fig01`, `fig02` |
| Failure mode analysis | 10 stitched comparison videos in [`videos/architectural_comparison/`](videos/architectural_comparison/) |
| Reproduction scripts | [`scripts/reproduction/01-04_*.sh`](scripts/reproduction/) |

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
