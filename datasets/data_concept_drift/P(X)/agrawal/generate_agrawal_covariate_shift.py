"""
Covariate Shift dataset — Agrawal Generator with feature shift (river).

Design (4,000 instances, 40 windows × 100):
  Pre-drift  (t=0..1,999):  salary/loan original range
  Post-drift (t=2,000..3,999): salary mean += 40,000, loan mean += 100,000
  Labels recomputed from fn=0 (depends only on age — P(Y|X) unchanged).

Output:
  agrawal_covariate_shift.csv
  agrawal_covariate_shift_feature_drift.png
"""

import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from river.datasets import synth
from scipy.stats import gaussian_kde

OUT_DIR  = os.path.dirname(os.path.abspath(__file__))
SEED     = 42
W        = 100
N_TOTAL  = 4_000
BOUNDARY = 2_000

SALARY_SHIFT = 40_000
LOAN_SHIFT   = 100_000

# ── 1. generate pool ──────────────────────────────────────────────────────────
print(f"Generating Agrawal pool ({N_TOTAL * 3} samples, fn=0)...")
gen  = synth.Agrawal(classification_function=0, seed=SEED)
pool = []
for x, _ in gen.take(N_TOTAL * 3):
    pool.append(dict(x))

feat_cols = list(pool[0].keys())
rng = np.random.default_rng(SEED)

# ── 2. build stream ───────────────────────────────────────────────────────────
pre_idx  = rng.choice(len(pool), N_TOTAL // 2, replace=False)
post_idx = rng.choice(
    np.setdiff1d(np.arange(len(pool)), pre_idx),
    N_TOTAL // 2, replace=False
)

def apply_label(row):
    age = float(row["age"])
    return 1 if (age < 40 or age >= 60) else 0

stream = []

for i in pre_idx:
    row = dict(pool[i])
    stream.append((row, apply_label(row)))

for i in post_idx:
    row = dict(pool[i])
    row["salary"] = row["salary"] + SALARY_SHIFT
    row["loan"]   = row["loan"]   + LOAN_SHIFT
    stream.append((row, apply_label(row)))

pre_part  = stream[:N_TOTAL // 2]
post_part = stream[N_TOTAL // 2:]
rng.shuffle(pre_part)
rng.shuffle(post_part)
stream = pre_part + post_part

print(f"  pre  label=1 rate: {np.mean([y for _,y in stream[:N_TOTAL//2]]):.3f}")
print(f"  post label=1 rate: {np.mean([y for _,y in stream[N_TOTAL//2:]]):.3f}")

# ── 3. write CSV ──────────────────────────────────────────────────────────────
out_csv = os.path.join(OUT_DIR, "agrawal_covariate_shift.csv")
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["t"] + feat_cols + ["y"])
    for t, (row, y) in enumerate(stream):
        writer.writerow([t] + [row[c] for c in feat_cols] + [y])
print(f"  saved -> {out_csv}")

# ── 4. visualize ──────────────────────────────────────────────────────────────
labels      = np.array([y for _, y in stream])
t_arr       = np.arange(N_TOTAL)
x_roll      = np.arange(W, N_TOTAL)
salary_vals = np.array([row["salary"] for row, _ in stream])
loan_vals   = np.array([row["loan"]   for row, _ in stream])
age_vals    = np.array([row["age"]    for row, _ in stream])

fig, axes = plt.subplots(1, 3, figsize=(18, 4))
fig.suptitle(
    "Covariate Shift — Agrawal (river, fn=0)\n"
    f"Abrupt shift at t={BOUNDARY}: salary +{SALARY_SHIFT:,}, loan +{LOAN_SHIFT:,}  |  "
    "P(Y|X) fixed (label depends only on age)",
    fontsize=11, fontweight="bold"
)

ax = axes[0]
for vals, color, label in [
    (salary_vals / 1e5, "#2563EB", "salary (×10⁵)"),
    (loan_vals   / 1e5, "#EF4444", "loan   (×10⁵)"),
    (age_vals    / 100, "#16A34A", "age    (×10²)"),
]:
    rm = np.array([vals[i - W: i].mean() for i in x_roll])
    ax.plot(t_arr[x_roll], rm, lw=1.4, alpha=0.9, label=label, color=color)
ax.axvline(BOUNDARY, color="black", lw=2, ls="--", alpha=0.6, label="drift boundary")
ax.set_xlabel("t", fontsize=9)
ax.set_ylabel(f"rolling mean  (window={W})", fontsize=9)
ax.set_title("Feature Rolling Mean", fontsize=10, fontweight="bold")
ax.legend(fontsize=8, framealpha=0.8)
ax.tick_params(labelsize=8)

ax2 = axes[1]
for cl, color, lbl in [(0, "#2563EB", "class 0"), (1, "#EF4444", "class 1")]:
    prop = np.array([np.mean(labels[i - W: i] == cl) for i in x_roll])
    ax2.plot(t_arr[x_roll], prop, color=color, lw=1.4, label=lbl)
ax2.axvline(BOUNDARY, color="black", lw=2, ls="--", alpha=0.6, label="drift boundary")
ax2.set_ylim(-0.02, 1.05)
ax2.set_xlabel("t", fontsize=9)
ax2.set_ylabel(f"class proportion  (window={W})", fontsize=9)
ax2.set_title("Rolling Class Proportion — P(Y) Stable", fontsize=10, fontweight="bold")
ax2.legend(fontsize=8, framealpha=0.8)
ax2.tick_params(labelsize=8)

ax3 = axes[2]
for seg, color, lbl in [
    (salary_vals[:BOUNDARY],  "#2563EB", "salary pre-drift"),
    (salary_vals[BOUNDARY:],  "#EF4444", "salary post-drift"),
]:
    lo, hi = np.percentile(seg, 0.5), np.percentile(seg, 99.5)
    xs = np.linspace(lo, hi, 300)
    kde = gaussian_kde(np.clip(seg, lo, hi), bw_method=0.15)
    ax3.plot(xs / 1e5, kde(xs), color=color, lw=1.8, label=lbl)
    ax3.fill_between(xs / 1e5, kde(xs), alpha=0.10, color=color)
ax3.set_xlabel("salary (×10⁵)", fontsize=9)
ax3.set_ylabel("density", fontsize=9)
ax3.set_title("Salary Distribution Shift (KDE)", fontsize=10, fontweight="bold")
ax3.legend(fontsize=8, framealpha=0.8)
ax3.tick_params(labelsize=8)

fig.tight_layout()
out_fig = os.path.join(OUT_DIR, "agrawal_covariate_shift_feature_drift.png")
fig.savefig(out_fig, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved -> {out_fig}")
print("Done.")
