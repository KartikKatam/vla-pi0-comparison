#!/usr/bin/env python3
"""Extract training loss curves for both Pi0 and Pi0-FAST from WandB.

Pi0 runs survive locally (only run-20260519_220752-cvqngdxt is the full one;
others died early). Pi0-FAST run dirs were cleaned up during pod disk pressure
but synced to wandb.ai cloud. Pull from the cloud via wandb.Api().

Outputs:
- pi0_train.csv   - step, loss, grad_norm, param_norm
- pi0fast_train.csv  - same columns
- runs_meta.json - run IDs + their final state + runtime
"""
import json
import sys
from pathlib import Path
import pandas as pd

import wandb

HERE = Path(__file__).parent
ENTITY = "kartikkatam"
PROJECT = "openpi"

api = wandb.Api()

# Find all runs in the project
runs = list(api.runs(f"{ENTITY}/{PROJECT}"))
print(f"Found {len(runs)} runs in {ENTITY}/{PROJECT}")

# Filter to recent pilot runs (today's runs)
PILOT_RUN_IDS = []
for r in runs:
    cfg = r.config
    name = r.name or ""
    if "pilot_5k_s42" not in name and "pi0fast_pilot_5k_s42" not in name:
        continue
    PILOT_RUN_IDS.append({
        "id": r.id,
        "name": name,
        "state": r.state,
        "runtime_s": r.summary.get("_runtime", 0),
        "final_step": r.summary.get("_step", 0),
        "final_loss": r.summary.get("loss", None),
        "final_grad_norm": r.summary.get("grad_norm", None),
        "config_name_arg": str(cfg.get("args", [None])[0]) if "args" in cfg else "",
        "url": r.url,
    })

print(f"\nPilot runs (matching pilot_5k_s42):")
for r in PILOT_RUN_IDS:
    print(f"  {r['name']} ({r['id']}): state={r['state']}, runtime={r['runtime_s']:.0f}s, step={r['final_step']}, loss={r['final_loss']}")

# For each model, pick the run with the longest runtime (the one that actually finished)
def best_run(name_substr):
    cands = [r for r in PILOT_RUN_IDS if name_substr in r["name"]]
    if not cands:
        return None
    return max(cands, key=lambda r: r["runtime_s"])

pi0_meta = best_run("pi0_pilot_5k_s42")
pi0fast_meta = best_run("pi0fast_pilot_5k_s42")

print(f"\nSelected pi0 run: {pi0_meta['id'] if pi0_meta else 'NONE'}")
print(f"Selected pi0_fast run: {pi0fast_meta['id'] if pi0fast_meta else 'NONE'}")

(HERE / "runs_meta.json").write_text(
    json.dumps({"pi0": pi0_meta, "pi0_fast": pi0fast_meta, "all_pilot": PILOT_RUN_IDS}, indent=2, default=str)
)
print(f"Wrote {HERE / 'runs_meta.json'}")

# Pull history (per-step metrics) for the chosen runs
def pull_history(run_id, label):
    if not run_id:
        print(f"  skip {label}: no run id")
        return None
    r = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    # scan_history returns the full series without aggregation
    keys = ["_step", "loss", "grad_norm", "param_norm"]
    rows = []
    for row in r.scan_history(keys=keys):
        rows.append(row)
    if not rows:
        print(f"  {label}: history empty")
        return None
    df = pd.DataFrame(rows)
    df = df.rename(columns={"_step": "step"})
    df = df.dropna(subset=["loss"]).reset_index(drop=True)
    df = df.sort_values("step").reset_index(drop=True)
    csv_path = HERE / f"{label}_train.csv"
    df.to_csv(csv_path, index=False)
    print(f"  {label}: pulled {len(df)} rows → {csv_path}")
    print(df.tail(3).to_string(index=False))
    return df

print()
print("=== Pulling Pi0 history ===")
pi0_df = pull_history(pi0_meta["id"] if pi0_meta else None, "pi0")
print()
print("=== Pulling Pi0-FAST history ===")
pi0fast_df = pull_history(pi0fast_meta["id"] if pi0fast_meta else None, "pi0fast")
