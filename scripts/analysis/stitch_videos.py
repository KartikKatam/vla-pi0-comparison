#!/usr/bin/env python3
"""Stitch side-by-side Pi0 vs Pi0-FAST comparison videos for the architectural-story tasks.

For each curated task, locate the Pi0 video and the Pi0-FAST video, then stitch them
horizontally with model labels and a header carrying the task description + per-task
success rate. The shorter video is held on its last frame so both end together.

Usage:
    python stitch_videos.py \
        --src /path/to/outputs/eval \
        --out /path/to/videos/architectural_comparison
"""
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────
# Curated comparisons: (suite, task_substring, headline, category)
# task_substring matches against the rollout filename (lower-snake-case)
# ──────────────────────────────────────────────────────────────────────────
COMPARISONS = [
    # Language grounding — Pi0-FAST decisive wins (libero_object)
    ("libero_object", "orange_juice",
        "Pi0 0/3 vs Pi0-FAST 3/3 — language→object grounding gap",
        "language_grounding_win_fast"),
    ("libero_object", "butter",
        "Pi0 1/3 vs Pi0-FAST 3/3 — lexical→visual mapping",
        "language_grounding_win_fast"),
    ("libero_object", "cream_cheese",
        "Pi0 1/3 vs Pi0-FAST 3/3 — language grounding",
        "language_grounding_win_fast"),

    # Motor control — Pi0 decisive wins
    ("libero_goal", "push_the_plate_to_the_front_of_the_stove",
        "Pi0 3/3 vs Pi0-FAST 0/3 — continuous force modulation",
        "motor_control_win_pi0"),
    ("libero_goal", "open_the_middle_drawer_of_the_cabinet",
        "Pi0 1/3 vs Pi0-FAST 0/3 — drawer-pull motor control",
        "motor_control_win_pi0"),

    # Reversal — Pi0 wins in Object (where Pi0-FAST usually wins)
    ("libero_object", "pick_up_the_milk",
        "Pi0 2/3 vs Pi0-FAST 0/3 — Pi0 wins this Object task",
        "reversal"),

    # Both succeed — baseline
    ("libero_spatial", "black_bowl_on_the_ramekin",
        "Both 3/3 — both architectures handle clear spatial predicates",
        "both_succeed"),
    ("libero_goal", "put_the_bowl_on_the_stove",
        "Both 3/3 — simple goal-conditioned pick-and-place",
        "both_succeed"),

    # Both fail — shared scene/grounding failures
    ("libero_spatial", "black_bowl_on_the_wooden_cabinet",
        "Both 0/3 — shared visual grounding failure",
        "both_fail"),
    ("libero_goal", "put_the_wine_bottle_on_the_rack",
        "Both 0/3 — shared placement failure",
        "both_fail"),
]


def find_video(suite_dir: Path, task_substr: str, prefer: str | None = None) -> Path | None:
    """Locate a rollout video matching task_substr.
    prefer is "success" or "failure" — used as tie-breaker when both exist.
    """
    cands = sorted(suite_dir.glob(f"*{task_substr}*.mp4"))
    if not cands:
        return None
    if prefer:
        for c in cands:
            if prefer in c.name:
                return c
    return cands[0]


def stitch_one(pi0_video: Path, pi0fast_video: Path, out_path: Path,
               task_desc: str, headline: str) -> bool:
    """Stitch pi0_video (left) and pi0fast_video (right) into a labeled side-by-side MP4.

    Layout:
        +--- header bar (768x60) ---+
        |   <task_desc> — <headline>|
        +-------------+-------------+
        |             |             |
        |  Pi0  3x    |  Pi0-FAST   |
        | 224→448 px  |  3x scaled  |
        |  + label    |  + label    |
        |             |             |
        +-------------+-------------+
    """
    # Escape characters that ffmpeg's drawtext treats specially
    def esc(s: str) -> str:
        return (s.replace("\\", "\\\\")
                 .replace(":", r"\:")
                 .replace("'", r"\\\'")
                 .replace(",", r"\,"))

    # Two-line title: task description (truncated) + headline
    title_line = task_desc if len(task_desc) <= 75 else task_desc[:72] + "..."
    head_line = headline if len(headline) <= 90 else headline[:87] + "..."

    # Per-side resolution: 448x448 video + 36 px banner
    # Final: 896 wide, 484 high + 80 header = 896×564
    filter_complex = (
        # Left video: scale up, pad bottom for label, draw model name banner
        f"[0:v]scale=448:448,tpad=stop_mode=clone:stop_duration=15,"
        f"drawbox=x=0:y=0:w=448:h=30:color=#1f77b4@1.0:t=fill,"
        f"drawtext=text='Pi0 (Flow Matching)':x=(w-tw)/2:y=5:fontsize=18:fontcolor=white[L];"
        # Right video: same but Pi0-FAST color (orange)
        f"[1:v]scale=448:448,tpad=stop_mode=clone:stop_duration=15,"
        f"drawbox=x=0:y=0:w=448:h=30:color=#ff7f0e@1.0:t=fill,"
        f"drawtext=text='Pi0-FAST (Autoregressive)':x=(w-tw)/2:y=5:fontsize=18:fontcolor=white[R];"
        # Stack horizontally
        f"[L][R]hstack=inputs=2[stack];"
        # Add header bar with task description + headline
        f"[stack]pad=896:564:0:80:color=black,"
        f"drawtext=text='{esc(title_line)}':x=(w-tw)/2:y=15:fontsize=18:fontcolor=white,"
        f"drawtext=text='{esc(head_line)}':x=(w-tw)/2:y=45:fontsize=16:fontcolor=#cccccc[out]"
    )

    # ffmpeg cmd. -shortest ensures we cut to the longest input (tpad already handled padding).
    cmd = [
        "ffmpeg", "-y",
        "-i", str(pi0_video),
        "-i", str(pi0fast_video),
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
        "-movflags", "+faststart",
        "-r", "10",
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"FAIL {out_path.name}: {proc.stderr[-500:]}", file=sys.stderr)
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="root path containing eval/{model}/{suite}/videos/")
    ap.add_argument("--out", required=True, help="output dir for stitched comparison videos")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for i, (suite, task_substr, headline, category) in enumerate(COMPARISONS, start=1):
        pi0_dir = src / "pi0" / suite / "videos"
        pi0fast_dir = src / "pi0_fast" / suite / "videos"

        # Pick the most representative outcome:
        # - "win-for-Pi0" categories → prefer Pi0 success + Pi0-FAST failure
        # - "win-for-FAST" categories → prefer Pi0 failure + Pi0-FAST success
        # - "both_succeed" → both success
        # - "both_fail" → both failure
        # - "reversal" → Pi0 success + Pi0-FAST failure (Pi0 is the reversal winner)
        if category == "language_grounding_win_fast":
            pi0_pref, fast_pref = "failure", "success"
        elif category == "motor_control_win_pi0":
            pi0_pref, fast_pref = "success", "failure"
        elif category == "both_succeed":
            pi0_pref = fast_pref = "success"
        elif category == "both_fail":
            pi0_pref = fast_pref = "failure"
        elif category == "reversal":
            pi0_pref, fast_pref = "success", "failure"
        else:
            pi0_pref = fast_pref = None

        v0 = find_video(pi0_dir, task_substr, prefer=pi0_pref)
        v1 = find_video(pi0fast_dir, task_substr, prefer=fast_pref)
        if not (v0 and v1):
            print(f"  ✗ {i:02d} {category}/{task_substr}: missing video (pi0={v0 is not None}, fast={v1 is not None})")
            continue

        # Derive task description from filename
        task_desc = v0.stem.replace("rollout_", "").replace("_success", "").replace("_failure", "")
        task_desc = task_desc.replace("_", " ")

        out_path = out_dir / f"{i:02d}_{category}_{task_substr}.mp4"
        print(f"  ⏳ {i:02d} {category}/{task_substr} → {out_path.name}")
        if stitch_one(v0, v1, out_path, task_desc, headline):
            print(f"  ✓ {out_path.name} ({out_path.stat().st_size // 1024} KB)")
            manifest.append({
                "n": i,
                "category": category,
                "suite": suite,
                "task": task_desc,
                "headline": headline,
                "pi0_source": v0.name,
                "pi0_fast_source": v1.name,
                "stitched": out_path.name,
            })

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {len(manifest)} comparison videos → {out_dir}/")
    print(f"Manifest: {out_dir}/manifest.json")


if __name__ == "__main__":
    main()
