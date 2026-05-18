"""
Prior Probability Shift dataset from UCI Bank Marketing (OpenML data_id=1461).

Design (4,000 instances, 40 windows × 100):
  Windows  1-20  (t=0    ..1,999): positive rate 10%  — stable period
  Windows 21-40  (t=2,000..3,999): positive rate 40%  — abrupt drift

P(X|Y) and P(Y|X) unchanged throughout — only P(Y) shifts at t=2,000.

Output:
  bank_prior_shift.csv
  bank_prior_shift_class_proportion.png
"""

import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

OUT_DIR  = os.path.dirname(os.path.abspath(__file__))
SEED     = 42
W        = 100
N_WIN    = 40
N_TOTAL  = W * N_WIN     # 4,000
BOUNDARY = 2_000

PRE_RATE  = 0.10
POST_RATE = 0.40
PRE_POS   = round(N_TOTAL // 2 * PRE_RATE)    # 200
PRE_NEG   = N_TOTAL // 2 - PRE_POS            # 1,800
POST_POS  = round(N_TOTAL // 2 * POST_RATE)   # 800
POST_NEG  = N_TOTAL // 2 - POST_POS           # 1,200

COL_RENAME = {
    "V1":"age","V2":"job","V3":"marital","V4":"education","V5":"default",
    "V6":"balance","V7":"housing","V8":"loan","V9":"contact","V10":"day",
    "V11":"month","V12":"duration","V13":"campaign","V14":"pdays",
    "V15":"previous","V16":"poutcome",
}
DURATION_COL = "duration"

# ── 1. fetch data ─────────────────────────────────────────────────────────────
print("Fetching bank-marketing from OpenML (data_id=1461)...")
bm = fetch_openml(data_id=1461, as_frame=True, parser="auto")
df = bm.frame.rename(columns=COL_RENAME).drop(columns=[DURATION_COL])
print(f"  shape after drop: {df.shape}")

feat_cols = [c for c in df.columns if c != "Class"]
pos_df = df[df["Class"] == "2"].reset_index(drop=True)
neg_df = df[df["Class"] == "1"].reset_index(drop=True)
print(f"  positive: {len(pos_df)},  negative: {len(neg_df)}")
assert PRE_POS + POST_POS <= len(pos_df)
assert PRE_NEG + POST_NEG <= len(neg_df)

# ── 2. sample ─────────────────────────────────────────────────────────────────
rng     = np.random.default_rng(SEED)
pos_idx = rng.choice(len(pos_df), PRE_POS + POST_POS, replace=False)
neg_idx = rng.choice(len(neg_df), PRE_NEG + POST_NEG, replace=False)

pre_pos_rows  = pos_df.iloc[pos_idx[:PRE_POS]].to_dict("records")
post_pos_rows = pos_df.iloc[pos_idx[PRE_POS:]].to_dict("records")
pre_neg_rows  = neg_df.iloc[neg_idx[:PRE_NEG]].to_dict("records")
post_neg_rows = neg_df.iloc[neg_idx[PRE_NEG:]].to_dict("records")

# ── 3. build stream ───────────────────────────────────────────────────────────
def build_segment(pos_rows, neg_rows, rng):
    n_total = len(pos_rows) + len(neg_rows)
    pos_positions = set(rng.choice(n_total, len(pos_rows), replace=False))
    pi, ni = 0, 0
    segment = []
    for idx in range(n_total):
        if idx in pos_positions:
            segment.append(pos_rows[pi]); pi += 1
        else:
            segment.append(neg_rows[ni]); ni += 1
    return segment

print("Building stream...")
stream = build_segment(pre_pos_rows, pre_neg_rows, rng) + \
         build_segment(post_pos_rows, post_neg_rows, rng)
assert len(stream) == N_TOTAL

# ── 4. write CSV ──────────────────────────────────────────────────────────────
out_csv = os.path.join(OUT_DIR, "bank_prior_shift.csv")
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["t"] + feat_cols + ["y"])
    for t, row in enumerate(stream):
        y = 1 if row["Class"] == "2" else 0
        writer.writerow([t] + [row[c] for c in feat_cols] + [y])
print(f"  saved -> {out_csv}  ({N_TOTAL} rows)")

# ── 5. visualize ──────────────────────────────────────────────────────────────
labels = np.array([1 if r["Class"] == "2" else 0 for r in stream])
t_arr  = np.arange(N_TOTAL)
x_roll = np.arange(W, N_TOTAL)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
fig.suptitle(
    "Prior Probability Shift — Bank Marketing Dataset  (duration removed)\n"
    f"Stable {int(PRE_RATE*100)}%  →  abrupt jump to {int(POST_RATE*100)}%  at t={BOUNDARY}",
    fontsize=11, fontweight="bold"
)

ax = axes[0]
for cl, color, label in [(0, "#2563EB", "no  (class 0)"),
                          (1, "#EF4444", "yes (class 1)")]:
    prop = np.array([np.mean(labels[i - W: i] == cl) for i in x_roll])
    ax.plot(t_arr[x_roll], prop, color=color, lw=1.5, label=label)
ax.axvline(BOUNDARY, color="black", lw=2, ls="--", alpha=0.6,
           label=f"drift boundary t={BOUNDARY}")
ax.set_ylim(-0.02, 1.05)
ax.set_xlabel("t", fontsize=9)
ax.set_ylabel(f"class proportion  (window={W})", fontsize=9)
ax.set_title("Rolling Class Proportion", fontsize=10, fontweight="bold")
ax.legend(fontsize=8, framealpha=0.8)
ax.tick_params(labelsize=8)

ax2 = axes[1]
pos_prop = np.array([np.mean(labels[i - W: i] == 1) for i in x_roll])
ax2.plot(t_arr[x_roll], pos_prop, color="#EF4444", lw=1.5, label="yes rolling rate")
ax2.axvline(BOUNDARY, color="black", lw=2, ls="--", alpha=0.6, label="drift boundary")
ax2.axhline(PRE_RATE,  color="#2563EB", lw=1.3, ls=":", alpha=0.8,
            label=f"pre  target  {PRE_RATE:.0%}")
ax2.axhline(POST_RATE, color="#EF4444", lw=1.3, ls=":", alpha=0.8,
            label=f"post target  {POST_RATE:.0%}")
ax2.set_ylim(-0.02, 0.60)
ax2.set_xlabel("t", fontsize=9)
ax2.set_ylabel(f"yes proportion  (window={W})", fontsize=9)
ax2.set_title("Positive Rate — Zoomed", fontsize=10, fontweight="bold")
ax2.legend(fontsize=8, framealpha=0.8)
ax2.tick_params(labelsize=8)

fig.tight_layout()
out_fig = os.path.join(OUT_DIR, "bank_prior_shift_class_proportion.png")
fig.savefig(out_fig, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved -> {out_fig}")
print("Done.")
