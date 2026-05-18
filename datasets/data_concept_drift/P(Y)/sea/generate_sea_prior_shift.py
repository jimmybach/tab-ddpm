"""
Prior Probability Shift dataset — SEA Generator (river).

Design (4,000 instances, 40 windows × 100):
  Sigmoid gradual drift from 20% to 80%, centered at t=2,000.
  Transition steepness = 300 (scaled from 1500 at N=20000, same 45% coverage).

  rate(t) = 0.20 + 0.60 * sigmoid((t - 2000) / 300)

Output:
  sea_prior_shift.csv
  sea_prior_shift_class_proportion.png
"""

import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from river.datasets import synth

OUT_DIR   = os.path.dirname(os.path.abspath(__file__))
SEED      = 42
W         = 100
N_WIN     = 40
N_TOTAL   = W * N_WIN        # 4,000
BOUNDARY  = 2_000
STEEPNESS = 300
POOL_SIZE = 24_000

def sigmoid_rate(t, center=BOUNDARY, steepness=STEEPNESS, lo=0.20, hi=0.80):
    return lo + (hi - lo) / (1 + np.exp(-(t - center) / steepness))

# ── 1. generate pool ──────────────────────────────────────────────────────────
print(f"Generating SEA pool ({POOL_SIZE} samples, variant=0)...")
gen = synth.SEA(variant=0, seed=SEED)
pos_pool, neg_pool = [], []
for x, y in gen.take(POOL_SIZE):
    (pos_pool if y else neg_pool).append(dict(x))

print(f"  positive: {len(pos_pool)},  negative: {len(neg_pool)}")

# ── 2. compute per-window counts ──────────────────────────────────────────────
window_centers = np.array([w * W + W // 2 for w in range(N_WIN)])
target_rates   = sigmoid_rate(window_centers)
pos_counts     = [round(W * r) for r in target_rates]
neg_counts     = [W - p for p in pos_counts]

total_pos_needed = sum(pos_counts)
total_neg_needed = sum(neg_counts)
assert total_pos_needed <= len(pos_pool), "Not enough positive samples!"
assert total_neg_needed <= len(neg_pool), "Not enough negative samples!"

feat_cols = list(pos_pool[0].keys())

# ── 3. sample — no replacement ────────────────────────────────────────────────
rng = np.random.default_rng(SEED)
pos_idx = rng.choice(len(pos_pool), total_pos_needed, replace=False)
neg_idx = rng.choice(len(neg_pool), total_neg_needed, replace=False)

# ── 4. build stream window by window ─────────────────────────────────────────
print("Building stream...")
stream = []
pos_ptr = neg_ptr = 0

for n_pos, n_neg in zip(pos_counts, neg_counts):
    win_pos = [pos_pool[i] for i in pos_idx[pos_ptr: pos_ptr + n_pos]]
    win_neg = [neg_pool[i] for i in neg_idx[neg_ptr: neg_ptr + n_neg]]
    pos_ptr += n_pos
    neg_ptr += n_neg

    n_total = n_pos + n_neg
    pos_positions = set(rng.choice(n_total, n_pos, replace=False))
    pi = ni = 0
    for i in range(n_total):
        if i in pos_positions:
            stream.append((win_pos[pi], 1)); pi += 1
        else:
            stream.append((win_neg[ni], 0)); ni += 1

assert len(stream) == N_TOTAL

# ── 5. write CSV ──────────────────────────────────────────────────────────────
out_csv = os.path.join(OUT_DIR, "sea_prior_shift.csv")
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["t"] + feat_cols + ["y"])
    for t, (row, y) in enumerate(stream):
        writer.writerow([t] + [row[c] for c in feat_cols] + [y])
print(f"  saved -> {out_csv}")

# ── 6. visualize ──────────────────────────────────────────────────────────────
labels = np.array([y for _, y in stream])
t_arr  = np.arange(N_TOTAL)
x_roll = np.arange(W, N_TOTAL)

t_smooth   = np.linspace(0, N_TOTAL, 500)
rate_curve = sigmoid_rate(t_smooth)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
fig.suptitle(
    "Prior Probability Shift — SEA (river, variant=0, P(Y|X) fixed)\n"
    f"Sigmoid gradual drift  20% → 80%,  centered at t={BOUNDARY}  (steepness={STEEPNESS})",
    fontsize=11, fontweight="bold"
)

ax = axes[0]
for cl, color, lbl in [(0, "#2563EB", "negative (class 0)"),
                        (1, "#EF4444", "positive (class 1)")]:
    prop = np.array([np.mean(labels[i - W: i] == cl) for i in x_roll])
    ax.plot(t_arr[x_roll], prop, color=color, lw=1.5, label=lbl)
ax.plot(t_smooth, rate_curve, color="#16A34A", lw=1.5, ls="--",
        alpha=0.8, label="sigmoid target")
ax.axvline(BOUNDARY, color="black", lw=1.5, ls=":", alpha=0.5, label="sigmoid midpoint")
ax.set_ylim(-0.02, 1.05)
ax.set_xlabel("t", fontsize=9)
ax.set_ylabel(f"class proportion  (window={W})", fontsize=9)
ax.set_title("Rolling Class Proportion", fontsize=10, fontweight="bold")
ax.legend(fontsize=8, framealpha=0.8)
ax.tick_params(labelsize=8)

ax2 = axes[1]
pos_prop = np.array([np.mean(labels[i - W: i] == 1) for i in x_roll])
ax2.plot(t_arr[x_roll], pos_prop, color="#EF4444", lw=1.5, label="positive rolling rate")
ax2.plot(t_smooth, rate_curve, color="#16A34A", lw=1.8, ls="--",
         alpha=0.85, label="sigmoid target curve")
ax2.axvline(BOUNDARY, color="black", lw=1.5, ls=":", alpha=0.5,
            label=f"sigmoid midpoint t={BOUNDARY}")
ax2.axhline(0.20, color="#2563EB", lw=1.2, ls=":", alpha=0.7, label="asymptote 20%")
ax2.axhline(0.80, color="#EF4444", lw=1.2, ls=":", alpha=0.7, label="asymptote 80%")
ax2.set_ylim(-0.02, 1.02)
ax2.set_xlabel("t", fontsize=9)
ax2.set_ylabel(f"positive proportion  (window={W})", fontsize=9)
ax2.set_title("Positive Rate vs Sigmoid Target", fontsize=10, fontweight="bold")
ax2.legend(fontsize=8, framealpha=0.8)
ax2.tick_params(labelsize=8)

fig.tight_layout()
out_fig = os.path.join(OUT_DIR, "sea_prior_shift_class_proportion.png")
fig.savefig(out_fig, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved -> {out_fig}")
print("Done.")
