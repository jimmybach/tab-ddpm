"""
Real Concept Drift — Hyperplane Generator (river).

Design (4,000 instances, 40 windows × 100):
  4 of 10 feature weights rotate continuously at mag_change=0.001 per step.

P(Y|X) changes: decision boundary rotates continuously (gradual concept drift).
P(X): stable — all features are uniform [0,1].
"""

import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from river.datasets import synth

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
N_TOTAL = 4_000
W       = 100
SEED    = 42

FEAT_COLORS = ["#2563EB", "#EF4444", "#16A34A", "#D97706",
               "#7C3AED", "#0891B2", "#DB2777", "#65A30D",
               "#0F766E", "#B45309"]

# ── 1. generate ───────────────────────────────────────────────────────────────
print("Generating Hyperplane stream...")
gen = synth.Hyperplane(
    seed=SEED,
    n_features=10,
    n_drift_features=4,
    mag_change=0.001,
)
rows = []
for t, (x, y) in enumerate(gen.take(N_TOTAL)):
    rows.append({"t": t, **x, "y": int(y)})

feat_cols = [k for k in rows[0] if k not in ("t", "y")]

# ── 2. write CSV ──────────────────────────────────────────────────────────────
out_csv = os.path.join(OUT_DIR, "hyperplane_concept_drift.csv")
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
    "Real Concept Drift — Hyperplane (river)\n"
    "4 of 10 feature weights rotate continuously  |  gradual drift, no fixed boundary",
    fontsize=11, fontweight="bold"
)

ax = axes[0]
for cl, color, lbl in [(0, "#2563EB", "class 0"), (1, "#EF4444", "class 1")]:
    prop = np.array([np.mean(labels[i - W: i] == cl) for i in x_roll])
    ax.plot(t_arr[x_roll], prop, color=color, lw=1.5, label=lbl)
ax.set_ylim(-0.02, 1.05)
ax.set_xlabel("t", fontsize=9)
ax.set_ylabel(f"class proportion (w={W})", fontsize=9)
ax.set_title("Rolling Class Proportion", fontsize=10, fontweight="bold")
ax.legend(fontsize=8)
ax.tick_params(labelsize=8)

ax2 = axes[1]
for fname, color in zip(feat_cols[:5], FEAT_COLORS[:5]):
    vals = np.array([float(r[fname]) for r in rows])
    rm   = np.array([vals[i - W: i].mean() for i in x_roll])
    ax2.plot(t_arr[x_roll], rm, color=color, lw=1.2, label=fname)
ax2.set_xlabel("t", fontsize=9)
ax2.set_ylabel(f"rolling mean (w={W})", fontsize=9)
ax2.set_title("Feature Rolling Mean — Stable P(X)", fontsize=10, fontweight="bold")
ax2.legend(fontsize=7, ncol=2)
ax2.tick_params(labelsize=8)

ax3 = axes[2]
Q = N_TOTAL // 4
quarter_labels = [f"Q{i+1}\nt={i*Q}–{(i+1)*Q-1}" for i in range(4)]
pos_rates = [labels[i * Q:(i + 1) * Q].mean() for i in range(4)]
x_pos = np.arange(4)
ax3.bar(x_pos, pos_rates, color="#EF4444", alpha=0.8, label="class 1")
ax3.bar(x_pos, [1 - r for r in pos_rates],
        bottom=pos_rates, color="#2563EB", alpha=0.8, label="class 0")
ax3.set_xticks(x_pos)
ax3.set_xticklabels(quarter_labels, fontsize=7.5)
ax3.set_ylabel("class proportion", fontsize=9)
ax3.set_title("Class Distribution per Quarter", fontsize=10, fontweight="bold")
ax3.legend(fontsize=8)
for i, r in enumerate(pos_rates):
    ax3.text(i, r / 2, f"{r:.1%}", ha="center", va="center",
             color="white", fontsize=9, fontweight="bold")
ax3.tick_params(labelsize=8)

fig.tight_layout()
out_fig = os.path.join(OUT_DIR, "hyperplane_concept_drift.png")
fig.savefig(out_fig, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved -> {out_fig}")
print("Done.")
