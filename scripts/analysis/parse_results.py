#!/usr/bin/env python3
"""Parse openpi LIBERO eval.log files into a structured results table.

Each eval.log contains per-episode 'Success: True/False' and cumulative
counters '# successes: N (PCT%)'. We extract:
- Total episodes attempted
- Total successes
- Per-task breakdown if discernible (uses task description preceding each episode)
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent / "outputs" / "eval"

MODELS = ["pi0", "pi0_fast"]
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]

# Regex patterns
RE_TASK   = re.compile(r"^Task:\s*(.+)$", re.MULTILINE)
RE_EPISODE = re.compile(r"^INFO:root:Starting episode (\d+)")
RE_SUCCESS = re.compile(r"^INFO:root:Success:\s*(True|False)")
RE_COUNTER = re.compile(r"^INFO:root:# successes:\s*(\d+)\s+\((\d+\.\d+)%\)")

def parse_eval_log(path):
    if not path.exists():
        return None
    text = path.read_text(errors="ignore")
    # Strip carriage returns to handle progress bars
    text = text.replace("\r", "\n")
    lines = text.splitlines()

    episodes = []        # list of (task, success)
    current_task = None
    pending_success = None
    last_counter = (0, 0, 0.0)  # (successes, total, pct)
    aborted = False

    for line in lines:
        m_task = RE_TASK.match(line.strip())
        if m_task:
            current_task = m_task.group(1).strip()
            continue
        m_succ = RE_SUCCESS.match(line)
        if m_succ:
            pending_success = (m_succ.group(1) == "True")
            episodes.append({"task": current_task, "success": pending_success})
            continue
        m_count = RE_COUNTER.match(line)
        if m_count:
            n_succ = int(m_count.group(1))
            pct = float(m_count.group(2))
            # derive total from succ/pct
            total = round(n_succ / (pct / 100.0)) if pct > 0 else 0
            last_counter = (n_succ, total, pct)
            continue
        if "Aborted" in line and "core dumped" in line:
            aborted = True

    return {
        "path": str(path),
        "episodes": episodes,
        "n_episodes": len(episodes),
        "n_success": sum(1 for e in episodes if e["success"]),
        "final_counter": last_counter,
        "aborted_midrun": aborted,
    }


def per_task_summary(episodes):
    """Group by task description."""
    by_task = {}
    for e in episodes:
        t = e["task"] or "<unknown>"
        by_task.setdefault(t, []).append(e["success"])
    return {
        t: {
            "n": len(succs),
            "succ": sum(succs),
            "pct": (sum(succs) / len(succs) * 100.0) if succs else 0.0,
        }
        for t, succs in by_task.items()
    }


def main():
    all_results = {}
    for model in MODELS:
        all_results[model] = {}
        for suite in SUITES:
            log_path = ROOT / model / suite / "eval.log"
            r = parse_eval_log(log_path)
            if r:
                r["per_task"] = per_task_summary(r["episodes"])
            all_results[model][suite] = r

    # Persist JSON for downstream plotting
    out_path = Path(__file__).parent / "results.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"Wrote {out_path}")

    # Human-readable summary
    print()
    print("=" * 70)
    print(f"{'SUITE':<20} {'pi0 (flow match)':<25} {'pi0_fast (autoreg)':<25}")
    print("=" * 70)
    for suite in SUITES:
        r0 = all_results["pi0"].get(suite) or {}
        r1 = all_results["pi0_fast"].get(suite) or {}
        s0 = f"{r0.get('n_success', 0)}/{r0.get('n_episodes', 0)}" if r0 else "no data"
        s1 = f"{r1.get('n_success', 0)}/{r1.get('n_episodes', 0)}" if r1 else "no data"
        p0 = ""
        p1 = ""
        if r0 and r0.get("n_episodes", 0) > 0:
            p0 = f" ({r0['n_success']/r0['n_episodes']*100:.1f}%)"
        if r1 and r1.get("n_episodes", 0) > 0:
            p1 = f" ({r1['n_success']/r1['n_episodes']*100:.1f}%)"
        print(f"{suite:<20} {s0 + p0:<25} {s1 + p1:<25}")
    print("=" * 70)

    # Aborted-midrun flags
    print()
    for model in MODELS:
        for suite in SUITES:
            r = all_results[model].get(suite)
            if r and r.get("aborted_midrun"):
                print(f"  ⚠ {model}/{suite}: eval aborted midrun")

    print()
    print("Detailed per-task breakdown:")
    for model in MODELS:
        print(f"\n--- {model} ---")
        for suite in SUITES:
            r = all_results[model].get(suite)
            if not r or not r.get("per_task"):
                continue
            print(f"  {suite}:")
            for task, stats in r["per_task"].items():
                if not task:
                    continue
                short = task if len(task) <= 60 else task[:57] + "..."
                print(f"    {short:<60} {stats['succ']}/{stats['n']} ({stats['pct']:.0f}%)")

if __name__ == "__main__":
    main()
