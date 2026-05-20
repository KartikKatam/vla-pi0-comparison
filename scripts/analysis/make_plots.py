#!/usr/bin/env python3
"""Generate report figures from pilot data.

Outputs (all saved to figures/):
  fig01_success_per_suite.png    - bar chart, success rate per suite × model
  fig02_per_task_heatmap.png     - per-task success rate, 2 models × 4 suites
  fig03_training_loss.png        - training loss vs step (dual y-axis: pi0 MSE, pi0_fast CE)
  fig04_grad_norm.png            - grad norm vs step
  fig05_param_norm.png           - param norm vs step (proxy for LoRA adapter update size)
  fig06_per_task_delta.png       - per-task pi0 vs pi0_fast head-to-head (diverging bar)
"""
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

HERE = Path(__file__).parent
FIGDIR = HERE / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["font.size"] = 10

# ── Load data ──────────────────────────────────────────────────────────────
with open(HERE / "results.json") as f:
    results = json.load(f)

pi0_train = pd.read_csv(HERE / "pi0_train.csv")
pi0fast_train = pd.read_csv(HERE / "pi0fast_train.csv")

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
SUITE_LABELS = {
    "libero_spatial": "Spatial\n(n=30)",
    "libero_object": "Object\n(n=30)",
    "libero_goal": "Goal\n(n=30)",
    "libero_10": "Long\n(n=20)",
}
MODELS = ["pi0", "pi0_fast"]
MODEL_LABELS = {"pi0": "Pi0 (flow matching)", "pi0_fast": "Pi0-FAST (autoregressive)"}
MODEL_COLORS = {"pi0": "#1f77b4", "pi0_fast": "#ff7f0e"}

# ── Fig 1: success rate per suite ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.5, 5))
x = np.arange(len(SUITES))
w = 0.36
for i, m in enumerate(MODELS):
    rates = []
    counts = []
    for s in SUITES:
        r = results[m][s]
        n = r["n_episodes"]
        ns = r["n_success"]
        pct = (ns / n * 100) if n else 0
        rates.append(pct)
        counts.append((ns, n))
    bars = ax.bar(x + (i - 0.5) * w, rates, w, label=MODEL_LABELS[m], color=MODEL_COLORS[m], edgecolor="black", linewidth=0.5)
    for j, b in enumerate(bars):
        ns, n = counts[j]
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.5,
                f"{ns}/{n}\n{rates[j]:.1f}%", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels([SUITE_LABELS[s] for s in SUITES])
ax.set_ylabel("Success rate (%)")
ax.set_ylim(0, 100)
ax.set_title("Pi0 vs Pi0-FAST on LIBERO (4k-step LoRA fine-tune)\n2 H100 SXM5 80GB, MUJOCO_GL=osmesa")
ax.legend(loc="upper right")
# Add error bar approximations: Wilson 95% CI midpoint
for i, m in enumerate(MODELS):
    for j, s in enumerate(SUITES):
        r = results[m][s]
        n, ns = r["n_episodes"], r["n_success"]
        if n == 0:
            continue
        p = ns / n
        z = 1.96
        # Wilson interval
        denom = 1 + z*z/n
        center = (p + z*z/(2*n)) / denom
        half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
        lo, hi = max(0, center - half) * 100, min(1, center + half) * 100
        ax.errorbar(x[j] + (i - 0.5) * w, p * 100, yerr=[[p*100 - lo], [hi - p*100]],
                    fmt="none", ecolor="black", capsize=3, lw=0.8, alpha=0.6)

ax.text(0.02, 0.98, "Error bars: Wilson 95% CI", transform=ax.transAxes,
        ha="left", va="top", fontsize=8, style="italic", alpha=0.7)
plt.tight_layout()
plt.savefig(FIGDIR / "fig01_success_per_suite.png", bbox_inches="tight")
plt.close()
print(f"  ✓ fig01_success_per_suite.png")

# ── Fig 2: per-task heatmap ───────────────────────────────────────────────
# Build a 2 × N_total_tasks matrix
fig, axes = plt.subplots(1, 4, figsize=(18, 5),
                         gridspec_kw={"width_ratios": [1, 1, 1, 1]})

for ax, suite in zip(axes, SUITES):
    # Get list of tasks (use pi0's order; both models do same tasks)
    pi0_tasks = results["pi0"][suite].get("per_task", {})
    pi0_fast_tasks = results["pi0_fast"][suite].get("per_task", {})
    tasks = list(pi0_tasks.keys())  # natural eval order

    mat = []
    labels = []
    for t in tasks:
        if t == "<unknown>" or t is None:
            continue
        # Shorter label
        short = t
        # Strip leading "pick up the " / "put the " for compactness
        for prefix in ("pick up the ", "put the "):
            if short.startswith(prefix):
                short = short[len(prefix):]
                break
        # Truncate
        if len(short) > 50:
            short = short[:47] + "..."
        labels.append(short)
        row = []
        for m in ["pi0", "pi0_fast"]:
            stats = results[m][suite].get("per_task", {}).get(t, {"pct": 0, "n": 0})
            row.append(stats.get("pct", 0))
        mat.append(row)

    if not mat:
        ax.set_visible(False)
        continue
    arr = np.array(mat)
    sns.heatmap(arr, annot=True, fmt=".0f",
                xticklabels=["Pi0", "Pi0-FAST"],
                yticklabels=labels,
                cmap="RdYlGn", vmin=0, vmax=100, ax=ax,
                cbar_kws={"label": "Success %"} if suite == "libero_10" else None,
                cbar=(suite == "libero_10"),
                linewidths=0.5, linecolor="white",
                annot_kws={"fontsize": 9})
    ax.set_title(suite.replace("libero_", ""), fontsize=11)
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=10)

plt.suptitle("Per-task success rate (% over n=2-3 trials/task)", y=1.02, fontsize=13)
plt.tight_layout()
plt.savefig(FIGDIR / "fig02_per_task_heatmap.png", bbox_inches="tight")
plt.close()
print(f"  ✓ fig02_per_task_heatmap.png")

# ── Fig 3: training loss curves (dual y-axis) ─────────────────────────────
# Pi0 uses MSE on velocity field (typically 0.01-0.1 range)
# Pi0-FAST uses cross-entropy on action tokens (typically 1-5 range)
# Show both on separate y-axes for visual comparison of convergence trajectory.
fig, ax = plt.subplots(figsize=(10, 5.5))
ax2 = ax.twinx()

ln1 = ax.plot(pi0_train["step"], pi0_train["loss"], color=MODEL_COLORS["pi0"],
               marker="o", markersize=3, lw=1.5, label="Pi0 (MSE, left axis)")
ax.set_ylabel("Pi0 loss — MSE on velocity field $v_\\theta(x_t,t)$", color=MODEL_COLORS["pi0"])
ax.tick_params(axis="y", labelcolor=MODEL_COLORS["pi0"])
ax.set_yscale("log")

ln2 = ax2.plot(pi0fast_train["step"], pi0fast_train["loss"], color=MODEL_COLORS["pi0_fast"],
                marker="s", markersize=3, lw=1.5, label="Pi0-FAST (CE, right axis)")
ax2.set_ylabel("Pi0-FAST loss — CE on action tokens", color=MODEL_COLORS["pi0_fast"])
ax2.tick_params(axis="y", labelcolor=MODEL_COLORS["pi0_fast"])

ax.set_xlabel("Training step")
ax.set_title("Training loss curves (5,000 LoRA fine-tune steps)\nDifferent objectives — loss magnitudes not directly comparable")
ax.grid(True, alpha=0.3)

# Combined legend
lns = ln1 + ln2
ax.legend(lns, [l.get_label() for l in lns], loc="upper right")

# Annotate final values
ax.annotate(f"final: {pi0_train['loss'].iloc[-1]:.4f}", xy=(pi0_train["step"].iloc[-1], pi0_train["loss"].iloc[-1]),
            xytext=(20, 30), textcoords="offset points",
            color=MODEL_COLORS["pi0"], fontsize=9,
            arrowprops=dict(arrowstyle="->", color=MODEL_COLORS["pi0"], alpha=0.7))
ax2.annotate(f"final: {pi0fast_train['loss'].iloc[-1]:.2f}", xy=(pi0fast_train["step"].iloc[-1], pi0fast_train["loss"].iloc[-1]),
             xytext=(20, -30), textcoords="offset points",
             color=MODEL_COLORS["pi0_fast"], fontsize=9,
             arrowprops=dict(arrowstyle="->", color=MODEL_COLORS["pi0_fast"], alpha=0.7))

plt.tight_layout()
plt.savefig(FIGDIR / "fig03_training_loss.png", bbox_inches="tight")
plt.close()
print(f"  ✓ fig03_training_loss.png")

# ── Fig 4: gradient norm ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(pi0_train["step"], pi0_train["grad_norm"], color=MODEL_COLORS["pi0"], marker="o", markersize=3, lw=1.5, label="Pi0")
ax.plot(pi0fast_train["step"], pi0fast_train["grad_norm"], color=MODEL_COLORS["pi0_fast"], marker="s", markersize=3, lw=1.5, label="Pi0-FAST")
ax.set_xlabel("Training step")
ax.set_ylabel("Gradient norm")
ax.set_title("Gradient norm vs training step")
ax.set_yscale("log")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGDIR / "fig04_grad_norm.png", bbox_inches="tight")
plt.close()
print(f"  ✓ fig04_grad_norm.png")

# ── Fig 5: parameter norm (LoRA adapter update size) ──────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(pi0_train["step"], pi0_train["param_norm"], color=MODEL_COLORS["pi0"], marker="o", markersize=3, lw=1.5, label="Pi0")
ax.plot(pi0fast_train["step"], pi0fast_train["param_norm"], color=MODEL_COLORS["pi0_fast"], marker="s", markersize=3, lw=1.5, label="Pi0-FAST")
ax.set_xlabel("Training step")
ax.set_ylabel("Parameter norm")
ax.set_title("Parameter norm vs training step\n(proxy for total magnitude of LoRA adapter weights + active params)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGDIR / "fig05_param_norm.png", bbox_inches="tight")
plt.close()
print(f"  ✓ fig05_param_norm.png")

# ── Fig 6: per-task head-to-head Δ (pi0 - pi0_fast) ───────────────────────
fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharex=False)

for ax, suite in zip(axes, SUITES):
    pi0_tasks = results["pi0"][suite].get("per_task", {})
    pi0fast_tasks = results["pi0_fast"][suite].get("per_task", {})
    tasks = [t for t in pi0_tasks.keys() if t and t != "<unknown>"]
    deltas = []
    labels = []
    for t in tasks:
        p0 = pi0_tasks.get(t, {}).get("pct", 0)
        p1 = pi0fast_tasks.get(t, {}).get("pct", 0)
        deltas.append(p0 - p1)
        short = t
        for prefix in ("pick up the ", "put the "):
            if short.startswith(prefix):
                short = short[len(prefix):]
                break
        if len(short) > 40:
            short = short[:37] + "..."
        labels.append(short)

    # Sort by delta
    order = np.argsort(deltas)
    deltas = np.array(deltas)[order]
    labels = [labels[i] for i in order]

    colors = [MODEL_COLORS["pi0"] if d > 0 else MODEL_COLORS["pi0_fast"] for d in deltas]
    ax.barh(np.arange(len(deltas)), deltas, color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(np.arange(len(deltas)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Δ % (Pi0 − Pi0-FAST)")
    ax.set_title(f"{suite.replace('libero_', '')}\n(positive = Pi0 wins)", fontsize=11)
    ax.set_xlim(-100, 100)
    ax.grid(True, alpha=0.3, axis="x")

plt.suptitle("Per-task head-to-head: where each architecture wins", y=1.02, fontsize=13)
plt.tight_layout()
plt.savefig(FIGDIR / "fig06_per_task_delta.png", bbox_inches="tight")
plt.close()
print(f"  ✓ fig06_per_task_delta.png")

print(f"\nAll figures in {FIGDIR}/")
