"""
Prior Probability Shift dataset from creditcard (OpenML).

Design (4,000 instances, 40 windows × 100):
  pre-drift  (t=0..1,999):   3 fraud / 1997 normal  → fraud ~0.15%
  post-drift (t=2,000..3,999): 95 fraud / 1905 normal → fraud ~4.75%

P(X|Y) and P(Y|X) unchanged — only P(Y) shifts at t=2,000.

Output:
  creditcard_prior_shift.csv
  prior_shift_class_proportion.png
"""

import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

OUT_DIR     = os.path.dirname(os.path.abspath(__file__))
SEED        = 42
W           = 100
N_TOTAL     = 4_000
BOUNDARY    = 2_000
PRE_FRAUD   = 3
POST_FRAUD  = 95
PRE_NORMAL  = BOUNDARY - PRE_FRAUD
POST_NORMAL = BOUNDARY - POST_FRAUD

# ── 1. load data ──────────────────────────────────────────────────────────────
print("Loading creditcard dataset from OpenML...")
credit = fetch_openml("creditcard", version=1, as_frame=True, parser="auto")
df = credit.frame
df["Class"] = df["Class"].astype(int)

fraud  = df[df["Class"] == 1].reset_index(drop=True)
normal = df[df["Class"] == 0].reset_index(drop=True)
print(f"  fraud: {len(fraud)}, normal: {len(normal)}")

rng = np.random.default_rng(SEED)

# ── 2. sample ─────────────────────────────────────────────────────────────────
pre_fraud_idx  = rng.choice(len(fraud),  PRE_FRAUD,   replace=False)
pre_normal_idx = rng.choice(len(normal), PRE_NORMAL,  replace=False)

remaining_fraud = np.setdiff1d(np.arange(len(fraud)), pre_fraud_idx)
post_fraud_idx  = rng.choice(remaining_fraud, POST_FRAUD, replace=False)
post_normal_idx = rng.choice(
    np.setdiff1d(np.arange(len(normal)), pre_normal_idx),
    POST_NORMAL, replace=False
)

pre_fraud_rows   = fraud.iloc[pre_fraud_idx]
pre_normal_rows  = normal.iloc[pre_normal_idx]
post_fraud_rows  = fraud.iloc[post_fraud_idx]
post_normal_rows = normal.iloc[post_normal_idx]

# ── 3. build stream ───────────────────────────────────────────────────────────
def build_segment(fraud_rows, normal_rows, rng):
    n_total = len(fraud_rows) + len(normal_rows)
    fraud_positions = set(rng.choice(n_total, len(fraud_rows), replace=False))
    fi, ni = 0, 0
    rows = []
    for pos in range(n_total):
        if pos in fraud_positions:
            rows.append(fraud_rows.iloc[fi]); fi += 1
        else:
            rows.append(normal_rows.iloc[ni]); ni += 1
    return rows

print("Building stream...")
stream = build_segment(pre_fraud_rows, pre_normal_rows, rng) + \
         build_segment(post_fraud_rows, post_normal_rows, rng)

# ── 4. write CSV ──────────────────────────────────────────────────────────────
feat_cols = [c for c in df.columns if c != "Class"]
out_csv = os.path.join(OUT_DIR, "creditcard_prior_shift.csv")
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["t"] + feat_cols + ["y"])
    for t, row in enumerate(stream):
        writer.writerow([t] + [row[c] for c in feat_cols] + [int(row["Class"])])
print(f"  saved -> {out_csv}  ({len(stream)} rows)")

# ── 5. visualize ──────────────────────────────────────────────────────────────
labels = np.array([int(r["Class"]) for r in stream])
t_arr  = np.arange(len(labels))
x_roll = np.arange(W, len(t_arr))

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
fig.suptitle(
    "Prior Probability Shift — Creditcard Dataset\n"
    f"P(X|Y) and P(Y|X) unchanged; fraud prior P(Y=1) shifts at t={BOUNDARY}",
    fontsize=11, fontweight="bold"
)

ax = axes[0]
for cl, color, label in [(0, "#2563EB", "normal (class 0)"),
                          (1, "#EF4444", "fraud  (class 1)")]:
    prop = np.array([np.mean(labels[i - W: i] == cl) for i in x_roll])
    ax.plot(t_arr[x_roll], prop, color=color, lw=1.5, label=label)
ax.axvline(BOUNDARY, color="black", lw=2, ls="--", alpha=0.6, label="drift boundary")
ax.set_ylim(-0.02, 1.05)
ax.set_xlabel("t", fontsize=9)
ax.set_ylabel(f"class proportion (window={W})", fontsize=9)
ax.set_title("Rolling Class Proportion", fontsize=10, fontweight="bold")
ax.legend(fontsize=8, framealpha=0.8)
ax.tick_params(labelsize=8)

ax2 = axes[1]
fraud_prop = np.array([np.mean(labels[i - W: i] == 1) for i in x_roll])
ax2.plot(t_arr[x_roll], fraud_prop, color="#EF4444", lw=1.5)
ax2.axvline(BOUNDARY, color="black", lw=2, ls="--", alpha=0.6, label="drift boundary")
ax2.axhline(PRE_FRAUD  / BOUNDARY, color="#2563EB", lw=1.2, ls=":", alpha=0.7,
            label=f"pre target  {PRE_FRAUD/BOUNDARY:.3%}")
ax2.axhline(POST_FRAUD / BOUNDARY, color="#EF4444", lw=1.2, ls=":", alpha=0.7,
            label=f"post target {POST_FRAUD/BOUNDARY:.2%}")
ax2.set_xlabel("t", fontsize=9)
ax2.set_ylabel(f"fraud proportion (window={W})", fontsize=9)
ax2.set_title("Fraud Rate — Zoomed", fontsize=10, fontweight="bold")
ax2.legend(fontsize=8, framealpha=0.8)
ax2.tick_params(labelsize=8)

fig.tight_layout()
out_fig = os.path.join(OUT_DIR, "prior_shift_class_proportion.png")
fig.savefig(out_fig, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved -> {out_fig}")
print("Done.")
