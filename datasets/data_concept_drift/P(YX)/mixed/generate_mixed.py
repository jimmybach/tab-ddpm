"""
Real Concept Drift — Mixed Generator (river).

Design (4,000 instances, 40 windows × 100):
  fn0 (t=0..1,999):     AND rule — low positive rate ~13%
  fn1 (t=2,000..3,999): OR  rule — high positive rate ~67%

P(Y|X) changes: output mapping flips at t=2,000 (abrupt concept drift).
P(X): stable — features sampled from the same distribution throughout.
"""

import csv, os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from river.datasets import synth
from matplotlib.lines import Line2D

OUT_DIR  = os.path.dirname(os.path.abspath(__file__))
N_TOTAL  = 4_000
W        = 100
SEED     = 42
BOUNDARY = N_TOTAL // 2   # t=2,000

# ── 1. generate ───────────────────────────────────────────────────────────────
print("Generating Mixed stream...")
rows = []
t = 0
for fn, count in [(0, BOUNDARY), (1, N_TOTAL - BOUNDARY)]:
    for x, y in synth.Mixed(classification_function=fn, seed=SEED).take(count):
        rows.append({"t": t, **{str(k): v for k, v in x.items()}, "y": int(y)})
        t += 1

feat_cols = [k for k in rows[0] if k not in ("t", "y")]

# ── 2. write CSV ──────────────────────────────────────────────────────────────
out_csv = os.path.join(OUT_DIR, "mixed_concept_drift.csv")
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["t"] + feat_cols + ["y"])
    writer.writeheader()
    writer.writerows(rows)
print(f"  saved -> {out_csv}")

# ── 3. visualize ──────────────────────────────────────────────────────────────
labels = np.array([r["y"] for r in rows])
t_arr  = np.arange(N_TOTAL)
x_roll = np.arange(W, N_TOTAL)

fig, axes = plt.subplots(1, 3, figsize=(18, 4))
fig.suptitle(
    "Real Concept Drift — Mixed (river)\n"
    f"fn0 → fn1 at t={BOUNDARY}: boolean+sinusoidal rule output flips (abrupt)",
    fontsize=11, fontweight="bold"
)

ax = axes[0]
for cl, color, lbl in [(0, "#2563EB", "class 0"), (1, "#EF4444", "class 1")]:
    prop = np.array([np.mean(labels[i - W: i] == cl) for i in x_roll])
    ax.plot(t_arr[x_roll], prop, color=color, lw=1.5, label=lbl)
ax.axvline(BOUNDARY, color="black", lw=2, ls="--", alpha=0.6, label="drift boundary")
ax.set_ylim(-0.02, 1.05)
ax.set_xlabel("t", fontsize=9)
ax.set_ylabel(f"class proportion (w={W})", fontsize=9)
ax.set_title("Rolling Class Proportion", fontsize=10, fontweight="bold")
ax.legend(fontsize=8)
ax.tick_params(labelsize=8)

ax2 = axes[1]
colors = ["#2563EB", "#EF4444", "#16A34A", "#D97706"]
for fname, color in zip(feat_cols, colors):
    vals = np.array([float(r[fname]) for r in rows])
    rm   = np.array([vals[i - W: i].mean() for i in x_roll])
    ax2.plot(t_arr[x_roll], rm, color=color, lw=1.4, label=fname)
ax2.axvline(BOUNDARY, color="black", lw=2, ls="--", alpha=0.5)
ax2.set_xlabel("t", fontsize=9)
ax2.set_ylabel(f"rolling mean (w={W})", fontsize=9)
ax2.set_title("Feature Rolling Mean — Stable P(X)", fontsize=10, fontweight="bold")
ax2.legend(fontsize=8)
ax2.tick_params(labelsize=8)

ax3 = axes[2]
rng    = np.random.default_rng(SEED)
x_feat = np.array([float(r["2"]) for r in rows])
y_feat = np.array([float(r["3"]) for r in rows])
for seg_mask, marker in [(t_arr < BOUNDARY, "o"), (t_arr >= BOUNDARY, "x")]:
    idx = np.where(seg_mask)[0]
    idx = rng.choice(idx, min(300, len(idx)), replace=False)
    for cl, color in [(0, "#2563EB"), (1, "#EF4444")]:
        m = labels[idx] == cl
        ax3.scatter(x_feat[idx[m]], y_feat[idx[m]], c=color, marker=marker, s=10, alpha=0.4)
xs = np.linspace(0, 1, 200)
ys = 0.5 + 0.3 * np.sin(3 * math.pi * xs)
ax3.plot(xs, ys, "k--", lw=1.5, label="sin boundary")
legend_elems = [
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#EF4444", markersize=7, label="class 1"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#2563EB", markersize=7, label="class 0"),
    Line2D([0],[0], marker="o", color="gray", markersize=6, ls="", label="pre-drift"),
    Line2D([0],[0], marker="x", color="gray", markersize=6, ls="", label="post-drift"),
    Line2D([0],[0], color="black", lw=1.5, ls="--", label="sin boundary"),
]
ax3.legend(handles=legend_elems, fontsize=7, framealpha=0.8)
ax3.set_xlabel("x_feat", fontsize=9)
ax3.set_ylabel("y_feat", fontsize=9)
ax3.set_title("Scatter: Pre vs Post  (label flips)", fontsize=10, fontweight="bold")
ax3.tick_params(labelsize=8)

fig.tight_layout()
out_fig = os.path.join(OUT_DIR, "mixed_concept_drift.png")
fig.savefig(out_fig, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved -> {out_fig}")
print("Done.")
