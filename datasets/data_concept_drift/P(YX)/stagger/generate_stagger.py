"""
Real Concept Drift — STAGGER Generator (river).

Design (4,000 instances, 40 windows × 100):
  fn0 (t=0..1,332):     y=1 iff size=small AND color=red
  fn1 (t=1,333..2,665): y=1 iff color=green OR shape=circular
  fn2 (t=2,666..3,999): y=1 iff size=medium OR size=large

P(Y|X) changes: completely different boolean rules across three segments.
P(X): stable — features sampled uniformly throughout.
"""

import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from river.datasets import synth

OUT_DIR  = os.path.dirname(os.path.abspath(__file__))
N_TOTAL  = 4_000
W        = 100
SEED     = 42
SEG      = N_TOTAL // 3       # 1,333 per segment
B1, B2   = SEG, 2 * SEG       # 1333, 2666

FEAT_COLORS = ["#2563EB", "#EF4444", "#16A34A"]

# ── 1. generate ───────────────────────────────────────────────────────────────
print("Generating STAGGER stream...")
rows = []
t = 0
for fn in [0, 1, 2]:
    count = SEG if fn < 2 else (N_TOTAL - 2 * SEG)
    for x, y in synth.STAGGER(classification_function=fn, seed=SEED).take(count):
        rows.append({"t": t, **x, "y": int(y)})
        t += 1

feat_cols = [k for k in rows[0] if k not in ("t", "y")]

# ── 2. write CSV ──────────────────────────────────────────────────────────────
out_csv = os.path.join(OUT_DIR, "stagger_concept_drift.csv")
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
    "Real Concept Drift — STAGGER (river)\n"
    "fn0: size=small∧color=red  →  fn1: color=green∨shape=circular  →  fn2: size=medium∨large",
    fontsize=11, fontweight="bold"
)

ax = axes[0]
for cl, color, lbl in [(0, "#2563EB", "class 0"), (1, "#EF4444", "class 1")]:
    prop = np.array([np.mean(labels[i - W: i] == cl) for i in x_roll])
    ax.plot(t_arr[x_roll], prop, color=color, lw=1.5, label=lbl)
for b, lbl in [(B1, "fn0→fn1"), (B2, "fn1→fn2")]:
    ax.axvline(b, color="black", lw=2, ls="--", alpha=0.6)
    ax.text(b + 20, 0.95, lbl, fontsize=7)
ax.set_ylim(-0.02, 1.05)
ax.set_xlabel("t", fontsize=9)
ax.set_ylabel(f"class proportion (w={W})", fontsize=9)
ax.set_title("Rolling Class Proportion", fontsize=10, fontweight="bold")
ax.legend(fontsize=8)
ax.tick_params(labelsize=8)

ax2 = axes[1]
for fname, color in zip(feat_cols, FEAT_COLORS):
    vals = np.array([float(r[fname]) for r in rows])
    rm   = np.array([vals[i - W: i].mean() for i in x_roll])
    ax2.plot(t_arr[x_roll], rm, color=color, lw=1.4, label=fname)
for b in [B1, B2]:
    ax2.axvline(b, color="black", lw=2, ls="--", alpha=0.5)
ax2.set_xlabel("t", fontsize=9)
ax2.set_ylabel(f"rolling mean (w={W})", fontsize=9)
ax2.set_title("Feature Rolling Mean — Stable P(X)", fontsize=10, fontweight="bold")
ax2.legend(fontsize=8)
ax2.tick_params(labelsize=8)

ax3 = axes[2]
seg_labels = [f"fn0\n(t=0–{B1-1})", f"fn1\n(t={B1}–{B2-1})", f"fn2\n(t={B2}–{N_TOTAL-1})"]
seg_slices = [labels[:B1], labels[B1:B2], labels[B2:]]
pos_rates  = [seg.mean() for seg in seg_slices]
x_pos = np.arange(3)
ax3.bar(x_pos, pos_rates, color="#EF4444", alpha=0.8, label="class 1")
ax3.bar(x_pos, [1 - r for r in pos_rates], bottom=pos_rates,
        color="#2563EB", alpha=0.8, label="class 0")
ax3.set_xticks(x_pos)
ax3.set_xticklabels(seg_labels, fontsize=8)
ax3.set_ylabel("class proportion", fontsize=9)
ax3.set_title("Class Distribution per Segment", fontsize=10, fontweight="bold")
ax3.legend(fontsize=8)
for i, r in enumerate(pos_rates):
    ax3.text(i, r / 2, f"{r:.1%}", ha="center", va="center",
             color="white", fontsize=9, fontweight="bold")
ax3.tick_params(labelsize=8)

fig.tight_layout()
out_fig = os.path.join(OUT_DIR, "stagger_concept_drift.png")
fig.savefig(out_fig, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved -> {out_fig}")
print("Done.")
