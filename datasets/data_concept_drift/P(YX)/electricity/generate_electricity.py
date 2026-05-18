"""
Real Concept Drift — Electricity Dataset (OpenML data_id=151).

NSW (Australia) electricity market data, 1996–1998.
Task: predict whether the price will go UP or DOWN relative to a 24h moving average.

The relationship between market features (demand, supply, time) and price direction
changes gradually as market conditions evolve — a classic real-world gradual
concept drift benchmark.

First 20,000 instances are retained (original ordering = chronological).

Features: date, day, period, nswprice, nswdemand, vicprice, vicdemand, transfer
P(Y|X) changes: gradual, market dynamics shift over time.
"""

import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
N_TOTAL = 20_000
W       = 500

# ── 1. load ───────────────────────────────────────────────────────────────────
print("Loading Electricity dataset from OpenML...")
ds = fetch_openml(data_id=151, as_frame=True, parser="auto")
df = ds.frame.head(N_TOTAL)
target = ds.target_names[0]
feat_cols = [c for c in df.columns if c != target]
print(f"  shape: {df.shape},  classes: {df[target].value_counts().to_dict()}")

# ── 2. write CSV ──────────────────────────────────────────────────────────────
out_csv = os.path.join(OUT_DIR, "electricity_concept_drift.csv")
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["t"] + feat_cols + ["y"])
    for t, (_, row) in enumerate(df.iterrows()):
        y = 1 if row[target] == "UP" else 0
        writer.writerow([t] + [row[c] for c in feat_cols] + [y])
print(f"  saved -> {out_csv}")

# ── 3. visualize ──────────────────────────────────────────────────────────────
labels  = np.array([1 if v == "UP" else 0 for v in df[target]])
t_arr   = np.arange(N_TOTAL)
x_roll  = np.arange(W, N_TOTAL)

SHOW_FEATS = ["nswdemand", "nswprice", "vicdemand", "vicprice"]
COLORS     = ["#2563EB", "#EF4444", "#16A34A", "#D97706"]

fig, axes = plt.subplots(1, 3, figsize=(18, 4))
fig.suptitle(
    "Real Concept Drift — Electricity (NSW, 1996–1998)\n"
    "Gradual drift: market rules evolve as supply/demand dynamics shift over time",
    fontsize=11, fontweight="bold"
)

# ── col1: rolling class proportion ───────────────────────────────────────────
ax = axes[0]
for cl, color, lbl in [(0, "#2563EB", "DOWN"), (1, "#EF4444", "UP")]:
    prop = np.array([np.mean(labels[i - W: i] == cl) for i in x_roll])
    ax.plot(t_arr[x_roll], prop, color=color, lw=1.5, label=lbl)
ax.set_ylim(-0.02, 1.05)
ax.set_xlabel("t", fontsize=9)
ax.set_ylabel(f"class proportion (w={W})", fontsize=9)
ax.set_title("Rolling Class Proportion", fontsize=10, fontweight="bold")
ax.legend(fontsize=8)
ax.tick_params(labelsize=8)

# ── col2: rolling mean of key features ───────────────────────────────────────
ax2 = axes[1]
for fname, color in zip(SHOW_FEATS, COLORS):
    vals = df[fname].values.astype(float)
    rm   = np.array([vals[i - W: i].mean() for i in x_roll])
    ax2.plot(t_arr[x_roll], rm, color=color, lw=1.3, label=fname)
ax2.set_xlabel("t", fontsize=9)
ax2.set_ylabel(f"rolling mean (w={W})", fontsize=9)
ax2.set_title("Feature Rolling Mean", fontsize=10, fontweight="bold")
ax2.legend(fontsize=8)
ax2.tick_params(labelsize=8)

# ── col3: class distribution per 4 quarters ──────────────────────────────────
ax3 = axes[2]
Q = N_TOTAL // 4
quarter_labels = [f"Q{i+1}\nt={i*Q}–{(i+1)*Q}" for i in range(4)]
pos_rates = [labels[i * Q:(i + 1) * Q].mean() for i in range(4)]
x_pos = np.arange(4)
ax3.bar(x_pos, pos_rates, color="#EF4444", alpha=0.8, label="UP")
ax3.bar(x_pos, [1 - r for r in pos_rates], bottom=pos_rates,
        color="#2563EB", alpha=0.8, label="DOWN")
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
out_fig = os.path.join(OUT_DIR, "electricity_concept_drift.png")
fig.savefig(out_fig, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved -> {out_fig}")
print("Done.")
