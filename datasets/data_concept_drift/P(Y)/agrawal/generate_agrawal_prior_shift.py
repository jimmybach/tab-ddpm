"""
Prior Probability Shift dataset — Agrawal Generator (river).

Design (4,000 instances, 40 windows × 100):
  Windows  1-20  (t=0    ..1,999): positive rate 20%  — stable period
  Windows 21-40  (t=2,000..3,999): positive rate 80%  — abrupt drift

P(Y|X) is kept constant: only classification_function=0 is used throughout.
P(X)   is kept constant: Agrawal samples features independently of the label.
Only P(Y) shifts — achieved by resampling from a pre-generated pool.

Output:
  agrawal_prior_shift.csv
  agrawal_prior_shift_class_proportion.png
"""

import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from river.datasets import synth

OUT_DIR  = os.path.dirname(os.path.abspath(__file__))
SEED     = 42
W        = 100
N_WIN    = 40
N_TOTAL  = W * N_WIN        # 4,000
BOUNDARY = 2_000

PRE_RATE  = 0.20
POST_RATE = 0.80
PRE_POS   = round(N_TOTAL // 2 * PRE_RATE)
PRE_NEG   = N_TOTAL // 2 - PRE_POS
POST_POS  = round(N_TOTAL // 2 * POST_RATE)
POST_NEG  = N_TOTAL // 2 - POST_POS

POOL_SIZE = 24_000

# ── 1. generate pool ──────────────────────────────────────────────────────────
print(f"Generating Agrawal pool ({POOL_SIZE} samples, fn=0)...")
gen = synth.Agrawal(classification_function=0, seed=SEED)
pos_pool, neg_pool = [], []
for x, y in gen.take(POOL_SIZE):
    (pos_pool if y else neg_pool).append(dict(x))

print(f"  positive: {len(pos_pool)},  negative: {len(neg_pool)}")
assert PRE_POS + POST_POS <= len(pos_pool), "Not enough positive samples!"
assert PRE_NEG + POST_NEG <= len(neg_pool), "Not enough negative samples!"

feat_cols = list(pos_pool[0].keys())

# ── 2. sample — no replacement ────────────────────────────────────────────────
rng = np.random.default_rng(SEED)
pos_idx = rng.choice(len(pos_pool), PRE_POS + POST_POS, replace=False)
neg_idx = rng.choice(len(neg_pool), PRE_NEG + POST_NEG, replace=False)

pre_pos  = [pos_pool[i] for i in pos_idx[:PRE_POS]]
post_pos = [pos_pool[i] for i in pos_idx[PRE_POS:]]
pre_neg  = [neg_pool[i] for i in neg_idx[:PRE_NEG]]
post_neg = [neg_pool[i] for i in neg_idx[PRE_NEG:]]

# ── 3. random interleave within each segment ─────────────────────────────────
def interleave(pos_rows, neg_rows, rng):
    n = len(pos_rows) + len(neg_rows)
    pos_pos_set = set(rng.choice(n, len(pos_rows), replace=False))
    pi = ni = 0
    out = []
    for i in range(n):
        if i in pos_pos_set:
            out.append((pos_rows[pi], 1)); pi += 1
        else:
            out.append((neg_rows[ni], 0)); ni += 1
    return out

stream = interleave(pre_pos, pre_neg, rng) + interleave(post_pos, post_neg, rng)
assert len(stream) == N_TOTAL

# ── 4. write CSV ──────────────────────────────────────────────────────────────
out_csv = os.path.join(OUT_DIR, "agrawal_prior_shift.csv")
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["t"] + feat_cols + ["y"])
    for t, (row, y) in enumerate(stream):
        writer.writerow([t] + [row[c] for c in feat_cols] + [y])
print(f"  saved -> {out_csv}")

# ── 5. visualize ──────────────────────────────────────────────────────────────
labels = np.array([y for _, y in stream])
t_arr  = np.arange(N_TOTAL)
x_roll = np.arange(W, N_TOTAL)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
fig.suptitle(
    "Prior Probability Shift — Agrawal (river, fn=0, P(Y|X) fixed)\n"
    f"Stable {int(PRE_RATE*100)}%  →  abrupt jump to {int(POST_RATE*100)}%  at t={BOUNDARY}",
    fontsize=11, fontweight="bold"
)

ax = axes[0]
for cl, color, lbl in [(0, "#2563EB", "negative (class 0)"),
                        (1, "#EF4444", "positive (class 1)")]:
    prop = np.array([np.mean(labels[i - W: i] == cl) for i in x_roll])
    ax.plot(t_arr[x_roll], prop, color=color, lw=1.5, label=lbl)
ax.axvline(BOUNDARY, color="black", lw=2, ls="--", alpha=0.6, label="drift boundary")
ax.set_ylim(-0.02, 1.05)
ax.set_xlabel("t", fontsize=9)
ax.set_ylabel(f"class proportion  (window={W})", fontsize=9)
ax.set_title("Rolling Class Proportion", fontsize=10, fontweight="bold")
ax.legend(fontsize=8, framealpha=0.8)
ax.tick_params(labelsize=8)

ax2 = axes[1]
pos_prop = np.array([np.mean(labels[i - W: i] == 1) for i in x_roll])
ax2.plot(t_arr[x_roll], pos_prop, color="#EF4444", lw=1.5, label="positive rolling rate")
ax2.axvline(BOUNDARY, color="black", lw=2, ls="--", alpha=0.6, label="drift boundary")
ax2.axhline(PRE_RATE,  color="#2563EB", lw=1.3, ls=":", alpha=0.8,
            label=f"pre  target  {PRE_RATE:.0%}")
ax2.axhline(POST_RATE, color="#EF4444", lw=1.3, ls=":", alpha=0.8,
            label=f"post target  {POST_RATE:.0%}")
ax2.set_ylim(-0.02, 1.02)
ax2.set_xlabel("t", fontsize=9)
ax2.set_ylabel(f"positive proportion  (window={W})", fontsize=9)
ax2.set_title("Positive Rate — Zoomed", fontsize=10, fontweight="bold")
ax2.legend(fontsize=8, framealpha=0.8)
ax2.tick_params(labelsize=8)

fig.tight_layout()
out_fig = os.path.join(OUT_DIR, "agrawal_prior_shift_class_proportion.png")
fig.savefig(out_fig, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved -> {out_fig}")
print("Done.")
