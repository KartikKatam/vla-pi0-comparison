# Video Demonstrations

This directory contains **all 114 rollout videos from the LIBERO evaluator** plus **10 curated side-by-side architectural-comparison videos** that pair Pi0 and Pi0-FAST on the same task.

## Quick start — watch these 10 videos first

The `architectural_comparison/` directory has Pi0 (left, blue label) vs Pi0-FAST (right, orange label) on the same task with the task description and a one-line headline. **These compress the architectural story into ~30 seconds each.**

| # | Category | Task | Story |
|---|---|---|---|
| 01 | language_grounding_win_fast | orange juice → basket | Pi0 0/3 vs Pi0-FAST 3/3 — Pi0-FAST's tokens share embedding space with language |
| 02 | language_grounding_win_fast | butter → basket | Pi0 1/3 vs Pi0-FAST 3/3 — same pattern |
| 03 | language_grounding_win_fast | cream cheese → basket | Pi0 1/3 vs Pi0-FAST 3/3 — same pattern |
| 04 | motor_control_win_pi0 | push plate to front of stove | Pi0 3/3 vs Pi0-FAST 0/3 — quantized actions snap sustained low-amplitude force |
| 05 | motor_control_win_pi0 | open middle drawer | Pi0 1/3 vs Pi0-FAST 0/3 — drawer-pull continuous force modulation |
| 06 | reversal | pick up milk → basket | Pi0 2/3 vs Pi0-FAST 0/3 — Pi0 wins an Object task; possible token-level confusion |
| 07 | both_succeed | bowl on ramekin → plate | Both 3/3 — both handle clear spatial predicates |
| 08 | both_succeed | put bowl on stove | Both 3/3 — simple goal-conditioned pick-and-place |
| 09 | both_fail | bowl on wooden cabinet → plate | Both 0/3 — shared visual-grounding failure (not architecture-specific) |
| 10 | both_fail | put wine bottle on rack | Both 0/3 — shared placement failure |

**Three videos to watch if you only have 90 seconds:** 01, 04, and 09. They show the architectural split (language grounding ↔ motor control) plus the shared failure mode.

## Full layout

```
videos/
├── README.md (this file)
├── architectural_comparison/    10 stitched side-by-side videos (Pi0 left, Pi0-FAST right)
│   ├── manifest.json
│   ├── 01_language_grounding_win_fast_orange_juice.mp4
│   ├── 02_language_grounding_win_fast_butter.mp4
│   ├── 03_language_grounding_win_fast_cream_cheese.mp4
│   ├── 04_motor_control_win_pi0_push_the_plate_to_the_front_of_the_stove.mp4
│   ├── 05_motor_control_win_pi0_open_the_middle_drawer_of_the_cabinet.mp4
│   ├── 06_reversal_pick_up_the_milk.mp4
│   ├── 07_both_succeed_black_bowl_on_the_ramekin.mp4
│   ├── 08_both_succeed_put_the_bowl_on_the_stove.mp4
│   ├── 09_both_fail_black_bowl_on_the_wooden_cabinet.mp4
│   └── 10_both_fail_put_the_wine_bottle_on_the_rack.mp4
├── pi0/                          all 59 original Pi0 rollouts (raw evaluator output)
│   ├── libero_spatial/  16 videos
│   ├── libero_object/   18 videos
│   ├── libero_goal/     14 videos
│   └── libero_10/       11 videos
└── pi0_fast/                     all 55 original Pi0-FAST rollouts
    ├── libero_spatial/  16 videos
    ├── libero_object/   13 videos
    ├── libero_goal/     16 videos
    └── libero_10/       10 videos
```

## What each video shows

**Raw rollouts** (`pi0/`, `pi0_fast/`):
- 224×224, 10 fps third-person camera, max 28 s (timeout) or 12 s (success).
- The openpi LIBERO evaluator saves **one representative video per (task, outcome)** — so each task has up to 2 videos (one success, one failure) per model.
- Filename embeds task description and outcome: `rollout_<task_underscored>_<success|failure>.mp4`.

**Stitched comparisons** (`architectural_comparison/`):
- 896×564, 10 fps. Pi0 (224 → upscaled 448) on left with blue header bar; Pi0-FAST (224 → upscaled 448) on right with orange header bar; black header strip with task description and one-line architectural headline.
- The shorter video (typically the success at ~12 s) is held on its last frame so both end together — making it easy to compare what each model does at every moment of the rollout.
- Source files for each stitched video are listed in `architectural_comparison/manifest.json`.

## How to use these for the report

Each video file's name is its citation. For the per-task entries in the report (Section 4 — Failure Mode Analysis), the corresponding video is at:

```
videos/{model}/{suite}/rollout_<task_description>_{success|failure}.mp4
```

For high-impact figure-equivalent shots in a presentation, use the `architectural_comparison/` files directly — they're already labeled and self-explanatory.
