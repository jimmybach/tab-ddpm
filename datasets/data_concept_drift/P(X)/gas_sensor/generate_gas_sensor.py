"""
Covariate Shift dataset — Gas Sensor Array Drift (OpenML: gas-drift).

Drift type: P(X) only
  - P(X)   changes: sensor readings degrade gradually over 36 months
  - P(Y|X) constant: gas identity labeling rule is physically unchanged
  - P(Y)   stable:  all 6 gas classes are sampled across all batches

The dataset contains 10 temporal batches of metal-oxide gas sensor measurements.
Sensor drift (oxidation, contamination) causes feature values to shift over time
while the classification task (identifying which gas is present) remains the same.

Features: 128 sensor response features (V1-V128), all continuous.
Classes:  6 gas types (Ethanol=1, Ethylene=2, Ammonia=3, Acetaldehyde=4,
                       Acetone=5, Toluene=6)

Total instances: 13,910 (original dataset size, not forced to 20,000)
Window size:     500
Total windows:   27 (approx)

Output:
  gas_sensor_covariate_shift.csv
  gas_sensor_feature_drift.png
"""

import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
W = 500

GAS_NAMES = {
    "1": "Ethanol", "2": "Ethylene", "3": "Ammonia",
    "4": "Acetaldehyde", "5": "Acetone", "6": "Toluene"
}
SEG_COLORS = ["#2563EB", "#EF4444", "#16A34A",
              "#D97706", "#7C3AED", "#0891B2"]

# ── 1. load ───────────────────────────────────────────────────────────────────
print("Loading gas-drift from OpenML...")
ds = fetch_openml(name="gas-drift", as_frame=True, parser="auto")
df = ds.frame
print(f"  shape: {df.shape}")

feat_cols = [c for c in df.columns if c != "Class"]
n = len(df)
t_arr = np.arange(n)
labels = df["Class"].values   # strings "1".."6"

# ── 2. write CSV ──────────────────────────────────────────────────────────────
out_csv = os.path.join(OUT_DIR, "gas_sensor_covariate_shift.csv")
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["t"] + feat_cols + ["y"])
    for i, row in enumerate(df.itertuples(index=False)):
        feats = [getattr(row, c) for c in feat_cols]
        y = int(getattr(row, "Class"))
        writer.writerow([i] + feats + [y])
print(f"  saved -> {out_csv}  ({n} rows)")

# ── 3. visualize ──────────────────────────────────────────────────────────────
# Select 4 representative features spread across sensors
SHOW_FEATS = ["V1", "V17", "V33", "V65"]
feat_arrays = {f: df[f].values.astype(float) for f in SHOW_FEATS}

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
fig.suptitle(
    "Covariate Shift — Gas Sensor Array Drift\n"
    "Sensor degradation over 36 months: P(X) drifts, P(Y|X) constant (gas identity unchanged)",
    fontsize=11, fontweight="bold"
)

# ── left: rolling mean of selected features ───────────────────────────────────
ax = axes[0]
colors = ["#2563EB", "#EF4444", "#16A34A", "#D97706"]
x_roll = np.arange(W, n)
for idx, (fname, vals) in enumerate(feat_arrays.items()):
    rm = np.array([vals[i - W: i].mean() for i in x_roll])
    ax.plot(t_arr[x_roll], rm, color=colors[idx], lw=1.4,
            alpha=0.9, label=fname)
ax.set_xlabel("t", fontsize=9)
ax.set_ylabel(f"rolling mean  (window={W})", fontsize=9)
ax.set_title("Feature Rolling Mean — Sensor Degradation", fontsize=10, fontweight="bold")
ax.legend(fontsize=8, framealpha=0.8)
ax.tick_params(labelsize=8)

# ── right: rolling class proportion ──────────────────────────────────────────
ax2 = axes[1]
x_roll2 = np.arange(W, n)
for ci, (cls, cname) in enumerate(GAS_NAMES.items()):
    prop = np.array([np.mean(labels[i - W: i] == cls) for i in x_roll2])
    ax2.plot(t_arr[x_roll2], prop,
             color=SEG_COLORS[ci], lw=1.3, alpha=0.85,
             label=f"{cls}: {cname}")
ax2.set_ylim(-0.02, 0.65)
ax2.set_xlabel("t", fontsize=9)
ax2.set_ylabel(f"class proportion  (window={W})", fontsize=9)
ax2.set_title("Rolling Class Proportion — Stable P(Y)", fontsize=10, fontweight="bold")
ax2.legend(fontsize=7, framealpha=0.8, ncol=2)
ax2.tick_params(labelsize=8)

fig.tight_layout()
out_fig = os.path.join(OUT_DIR, "gas_sensor_feature_drift.png")
fig.savefig(out_fig, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved -> {out_fig}")
print("Done.")
