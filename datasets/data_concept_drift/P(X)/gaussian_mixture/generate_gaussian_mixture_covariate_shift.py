"""
Covariate Shift dataset — Custom Gaussian Mixture with parallel feature drift.

Design (4,000 instances, 40 windows × 100):
  mu2(t) drifts from 0 to 5 via sigmoid centered at t=2,000 (steepness=300).
  Both class means drift together along x2 axis — boundary x1=0 unchanged.

Output:
  gaussian_mixture_covariate_shift.csv
  gaussian_mixture_feature_drift.png
"""

import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D

OUT_DIR      = os.path.dirname(os.path.abspath(__file__))
SEED         = 42
W            = 100
N_TOTAL      = 4_000
BOUNDARY_T   = 2_000
STEEPNESS    = 300
DRIFT_MAGNITUDE = 5.0

rng = np.random.default_rng(SEED)

def mu2(t):
    return DRIFT_MAGNITUDE / (1 + np.exp(-(t - BOUNDARY_T) / STEEPNESS))

# ── 1. generate stream ────────────────────────────────────────────────────────
print("Generating Gaussian Mixture stream...")
stream = []
class_seq = np.tile([0, 1], N_TOTAL // 2)

for i, cls in enumerate(class_seq):
    t = i
    shift = mu2(t)
    x1 = rng.normal(-2.0, 1.0) if cls == 0 else rng.normal(+2.0, 1.0)
    x2 = rng.normal(shift, 1.0)
    y  = 1 if x1 >= 0 else 0
    stream.append((x1, x2, y))

stream  = np.array(stream)
x1_arr  = stream[:, 0]
x2_arr  = stream[:, 1]
labels  = stream[:, 2].astype(int)
t_arr   = np.arange(N_TOTAL)

print(f"  pre  label=1 rate: {np.mean(labels[:N_TOTAL//2]):.3f}")
print(f"  post label=1 rate: {np.mean(labels[N_TOTAL//2:]):.3f}")

# ── 2. write CSV ──────────────────────────────────────────────────────────────
out_csv = os.path.join(OUT_DIR, "gaussian_mixture_covariate_shift.csv")
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["t", "x1", "x2", "y"])
    for t, (x1, x2, y) in enumerate(zip(x1_arr, x2_arr, labels)):
        writer.writerow([t, round(float(x1), 6), round(float(x2), 6), int(y)])
print(f"  saved -> {out_csv}")

# ── 3. visualize ──────────────────────────────────────────────────────────────
x_roll   = np.arange(W, N_TOTAL)
t_smooth = np.linspace(0, N_TOTAL, 500)
mu2_curve = np.array([mu2(t) for t in t_smooth])

fig, axes = plt.subplots(1, 3, figsize=(18, 4))
fig.suptitle(
    "Covariate Shift — Gaussian Mixture (custom)\n"
    "Both class means drift along x2 (parallel to boundary x1=0)  |  "
    "P(Y|X) fixed: y = 1 iff x1 ≥ 0",
    fontsize=11, fontweight="bold"
)

ax = axes[0]
rm_x1 = np.array([x1_arr[i - W: i].mean() for i in x_roll])
rm_x2 = np.array([x2_arr[i - W: i].mean() for i in x_roll])
ax.plot(t_arr[x_roll], rm_x1, color="#2563EB", lw=1.4, label="x1 (stable)")
ax.plot(t_arr[x_roll], rm_x2, color="#EF4444", lw=1.4, label="x2 (drifting)")
ax.plot(t_smooth, mu2_curve, color="#16A34A", lw=1.5, ls="--",
        alpha=0.8, label="x2 target μ(t)")
ax.axvline(BOUNDARY_T, color="black", lw=1.5, ls=":", alpha=0.5, label="sigmoid midpoint")
ax.set_xlabel("t", fontsize=9)
ax.set_ylabel(f"rolling mean  (window={W})", fontsize=9)
ax.set_title("Feature Rolling Mean", fontsize=10, fontweight="bold")
ax.legend(fontsize=8, framealpha=0.8)
ax.tick_params(labelsize=8)

ax2 = axes[1]
for cl, color, lbl in [(0, "#2563EB", "class 0"), (1, "#EF4444", "class 1")]:
    prop = np.array([np.mean(labels[i - W: i] == cl) for i in x_roll])
    ax2.plot(t_arr[x_roll], prop, color=color, lw=1.4, label=lbl)
ax2.axvline(BOUNDARY_T, color="black", lw=1.5, ls=":", alpha=0.5)
ax2.set_ylim(-0.02, 1.05)
ax2.set_xlabel("t", fontsize=9)
ax2.set_ylabel(f"class proportion  (window={W})", fontsize=9)
ax2.set_title("Rolling Class Proportion — P(Y) Stable", fontsize=10, fontweight="bold")
ax2.legend(fontsize=8, framealpha=0.8)
ax2.tick_params(labelsize=8)

ax3 = axes[2]
n_samp   = 400
pre_idx  = rng.choice(N_TOTAL // 2, n_samp, replace=False)
post_idx = rng.choice(np.arange(N_TOTAL // 2, N_TOTAL), n_samp, replace=False)
for idx, marker in [(pre_idx, "o"), (post_idx, "x")]:
    for cl, color in [(0, "#2563EB"), (1, "#EF4444")]:
        mask = labels[idx] == cl
        ax3.scatter(x1_arr[idx[mask]], x2_arr[idx[mask]],
                    c=color, marker=marker, s=8, alpha=0.35)
ax3.axvline(0, color="black", lw=2, ls="--", alpha=0.7, label="decision boundary x1=0")
legend_elems = [
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#EF4444", markersize=7, label="class 1"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#2563EB", markersize=7, label="class 0"),
    Line2D([0],[0], marker="o", color="gray", markersize=6, ls="", label="pre-drift"),
    Line2D([0],[0], marker="x", color="gray", markersize=6, ls="", label="post-drift"),
    Line2D([0],[0], color="black", lw=2, ls="--", label="boundary x1=0"),
]
ax3.legend(handles=legend_elems, fontsize=7, framealpha=0.8)
ax3.set_xlabel("x1", fontsize=9)
ax3.set_ylabel("x2", fontsize=9)
ax3.set_title("Scatter: Pre vs Post  (x2 drifts upward)", fontsize=10, fontweight="bold")
ax3.tick_params(labelsize=8)

fig.tight_layout()
out_fig = os.path.join(OUT_DIR, "gaussian_mixture_feature_drift.png")
fig.savefig(out_fig, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved -> {out_fig}")
print("Done.")
