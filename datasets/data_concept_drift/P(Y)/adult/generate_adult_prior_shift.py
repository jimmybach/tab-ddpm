"""
Prior Probability Shift dataset from UCI Adult.

Design (4,000 instances, 40 windows × 100):
  Windows  1-20  (t=0    ..1,999): positive rate 10%          — stable
  Windows 21-25  (t=2,000..2,499): positive rate 70%          — abrupt jump
  Windows 26-40  (t=2,500..3,999): linear decline 70% → 30%  — gradual drift

Positive label = '>50K'
P(X|Y) and P(Y|X) unchanged throughout — only P(Y) shifts.

Output:
  adult_prior_shift.csv
  adult_prior_shift_class_proportion.png
"""

import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
# adult.data lives in data_cd (the full-size archive folder)
SRC  = os.path.join(BASE, "..", "..", "..", "data_cd", "P(Y)", "adult", "adult.data")

SEED    = 42
W       = 100
N_WIN   = 40
N_TOTAL = W * N_WIN   # 4,000

COLS = [
    "age","workclass","fnlwgt","education","education-num",
    "marital-status","occupation","relationship","race","sex",
    "capital-gain","capital-loss","hours-per-week","native-country","income"
]

# ── 1. load & parse ───────────────────────────────────────────────────────────
print("Loading adult.data...")
pos_rows, neg_rows = [], []
with open(SRC, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 15:
            continue
        row = dict(zip(COLS, parts))
        if row["income"] == ">50K":
            pos_rows.append(row)
        elif row["income"] == "<=50K":
            neg_rows.append(row)

print(f"  positive (>50K): {len(pos_rows)},  negative (<=50K): {len(neg_rows)}")

rng = np.random.default_rng(SEED)
rng.shuffle(pos_rows)
rng.shuffle(neg_rows)

# ── 2. per-window positive rates ─────────────────────────────────────────────
# windows 1-20  → 10%
# windows 21-25 → 70%
# windows 26-40 → linear 70% → 30%
rates = []
for w in range(1, N_WIN + 1):
    if w <= 20:
        rates.append(0.10)
    elif w <= 25:
        rates.append(0.70)
    else:
        i = w - 26          # 0..14
        rate = 0.70 - i * (0.70 - 0.30) / 14
        rates.append(rate)

pos_counts = [round(W * r) for r in rates]
neg_counts = [W - p for p in pos_counts]

total_pos_needed = sum(pos_counts)
total_neg_needed = sum(neg_counts)
print(f"  positive needed: {total_pos_needed}  (available: {len(pos_rows)})")
print(f"  negative needed: {total_neg_needed}  (available: {len(neg_rows)})")
assert total_pos_needed <= len(pos_rows), "Not enough positive samples!"
assert total_neg_needed <= len(neg_rows), "Not enough negative samples!"

# ── 3. build stream ───────────────────────────────────────────────────────────
print("Building stream...")
stream = []
pos_ptr, neg_ptr = 0, 0

for n_pos, n_neg in zip(pos_counts, neg_counts):
    win_pos = pos_rows[pos_ptr: pos_ptr + n_pos]
    win_neg = neg_rows[neg_ptr: neg_ptr + n_neg]
    pos_ptr += n_pos
    neg_ptr += n_neg
    window  = win_pos + win_neg
    indices = rng.permutation(len(window))
    for idx in indices:
        stream.append(window[idx])

assert len(stream) == N_TOTAL

# ── 4. write CSV ──────────────────────────────────────────────────────────────
out_csv   = os.path.join(BASE, "adult_prior_shift.csv")
feat_cols = [c for c in COLS if c != "income"]

with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["t"] + feat_cols + ["y"])
    for t, row in enumerate(stream):
        y = 1 if row["income"] == ">50K" else 0
        writer.writerow([t] + [row[c] for c in feat_cols] + [y])
print(f"  saved -> {out_csv}")

# ── 5. visualize ──────────────────────────────────────────────────────────────
labels = np.array([1 if r["income"] == ">50K" else 0 for r in stream])
t_arr  = np.arange(N_TOTAL)
x_roll = np.arange(W, N_TOTAL)

BOUNDARY1 = 2_000   # abrupt jump
BOUNDARY2 = 2_500   # gradual decline start

t_targets    = np.array([w * W + W // 2 for w in range(N_WIN)])
target_rates = np.array(rates)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
fig.suptitle(
    "Prior Probability Shift — Adult Dataset\n"
    f"Stable 10%  →  abrupt jump 70% (t={BOUNDARY1})  →  linear decline to 30% (t={BOUNDARY2}–{N_TOTAL})",
    fontsize=11, fontweight="bold"
)

ax = axes[0]
prop_pos = np.array([np.mean(labels[i - W: i] == 1) for i in x_roll])
prop_neg = np.array([np.mean(labels[i - W: i] == 0) for i in x_roll])
ax.plot(t_arr[x_roll], prop_neg, color="#2563EB", lw=1.5, label="<=50K (class 0)")
ax.plot(t_arr[x_roll], prop_pos, color="#EF4444", lw=1.5, label=">50K  (class 1)")
ax.step(t_targets, target_rates, where="mid",
        color="#16A34A", lw=1.2, ls="--", alpha=0.7, label="target rate")
ax.axvline(BOUNDARY1, color="black", lw=1.8, ls="--", alpha=0.5)
ax.axvline(BOUNDARY2, color="gray",  lw=1.5, ls=":",  alpha=0.5)
ax.text(BOUNDARY1, 1.02, "abrupt",    fontsize=7, ha="center")
ax.text((BOUNDARY2 + N_TOTAL) // 2, 1.02, "gradual ↓", fontsize=7, ha="center")
ax.set_ylim(-0.02, 1.08)
ax.set_xlabel("t", fontsize=9)
ax.set_ylabel(f"class proportion (window={W})", fontsize=9)
ax.set_title("Rolling Class Proportion", fontsize=10, fontweight="bold")
ax.legend(fontsize=8, framealpha=0.8)
ax.tick_params(labelsize=8)

ax2 = axes[1]
ax2.plot(t_arr[x_roll], prop_pos, color="#EF4444", lw=1.5, label=">50K rolling rate")
ax2.step(t_targets, target_rates, where="mid",
         color="#16A34A", lw=1.5, ls="--", alpha=0.8, label="target rate")
ax2.axvline(BOUNDARY1, color="black", lw=1.8, ls="--", alpha=0.5,
            label=f"abrupt drift t={BOUNDARY1}")
ax2.axvline(BOUNDARY2, color="gray",  lw=1.5, ls=":",  alpha=0.5,
            label=f"gradual drift t={BOUNDARY2}")
ax2.set_ylim(-0.02, 0.85)
ax2.set_xlabel("t", fontsize=9)
ax2.set_ylabel(f">50K proportion (window={W})", fontsize=9)
ax2.set_title("Positive Rate — Zoomed", fontsize=10, fontweight="bold")
ax2.legend(fontsize=8, framealpha=0.8)
ax2.tick_params(labelsize=8)

fig.tight_layout()
out_fig = os.path.join(BASE, "adult_prior_shift_class_proportion.png")
fig.savefig(out_fig, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved -> {out_fig}")
print("Done.")
