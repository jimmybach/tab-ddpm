"""
Real Concept Drift — MAGIC Telescope Dataset (OpenML data_id=1120).

Design (4,000 instances, 40 windows × 100):
  Stratified sampling across 20 fAlpha bins × 200 per bin.
  Covers the full incident angle range 0°–90°, preserving the complete
  incremental drift trajectory (vs. naive truncation to first 4,000).

Output:
  magic_telescope_concept_drift.csv
  magic_telescope_concept_drift.png
"""

import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

OUT_DIR  = os.path.dirname(os.path.abspath(__file__))
N_TOTAL  = 4_000
N_BINS   = 20
W        = 100
SEED     = 42

FEAT_COLORS = ["#2563EB", "#EF4444", "#16A34A", "#D97706"]

# ── 1. load & sort by fAlpha ──────────────────────────────────────────────────
print("Loading MAGIC Telescope from OpenML (data_id=1120)...")
ds = fetch_openml(data_id=1120, as_frame=True, parser="auto")
df = ds.frame.copy()
target    = "class:"
feat_cols = [c for c in df.columns if c != target]

df_sorted = df.sort_values("fAlpha:").reset_index(drop=True)
n_full    = len(df_sorted)
print(f"  full shape: {df_sorted.shape}")
print(f"  fAlpha range: {df_sorted['fAlpha:'].min():.2f}° – {df_sorted['fAlpha:'].max():.2f}°")

# ── 2. stratified sampling: 20 bins × 200 = 4,000 ────────────────────────────
rng     = np.random.default_rng(SEED)
per_bin = N_TOTAL // N_BINS   # 200 per bin
sampled_idx = []
for b in range(N_BINS):
    lo  = int(b * n_full / N_BINS)
    hi  = int((b + 1) * n_full / N_BINS)
    idx = rng.choice(np.arange(lo, hi), per_bin, replace=False)
    sampled_idx.extend(sorted(idx))   # keep ascending fAlpha within bin

df_stream = df_sorted.iloc[sampled_idx].reset_index(drop=True)
print(f"  sampled: {len(df_stream)} rows")
print(f"  fAlpha range after sampling: "
      f"{df_stream['fAlpha:'].min():.2f}° – {df_stream['fAlpha:'].max():.2f}°")

# ── 3. write CSV ──────────────────────────────────────────────────────────────
clean_cols = [c.rstrip(":") for c in feat_cols]
out_csv = os.path.join(OUT_DIR, "magic_telescope_concept_drift.csv")
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["t"] + clean_cols + ["y"])
    for t, (_, row) in enumerate(df_stream.iterrows()):
        y = 1 if row[target] == "g" else 0
        writer.writerow([t] + [row[c] for c in feat_cols] + [y])
print(f"  saved -> {out_csv}  ({N_TOTAL} rows)")

# ── 4. visualize ──────────────────────────────────────────────────────────────
labels     = np.array([1 if v == "g" else 0 for v in df_stream[target]])
alpha_vals = df_stream["fAlpha:"].values.astype(float)
t_arr      = np.arange(N_TOTAL)
x_roll     = np.arange(W, N_TOTAL)

SHOW_FEATS = ["fLength:","fWidth:","fSize:","fAlpha:"]

fig, axes = plt.subplots(1, 3, figsize=(18, 4))
fig.suptitle(
    "Real Concept Drift — MAGIC Telescope (stratified by fAlpha, 20 bins × 200)\n"
    "Incremental drift: gamma/hadron boundary shifts as incident angle increases",
    fontsize=11, fontweight="bold"
)

ax = axes[0]
for cl, color, lbl in [(0, "#2563EB", "hadron (class 0)"),
                        (1, "#EF4444", "gamma  (class 1)")]:
    prop = np.array([np.mean(labels[i - W: i] == cl) for i in x_roll])
    ax.plot(t_arr[x_roll], prop, color=color, lw=1.5, label=lbl)
ax.set_ylim(-0.02, 1.05)
ax.set_xlabel("t  (increasing fAlpha →)", fontsize=9)
ax.set_ylabel(f"class proportion (w={W})", fontsize=9)
ax.set_title("Rolling Class Proportion", fontsize=10, fontweight="bold")
ax.legend(fontsize=8, framealpha=0.8)
ax.tick_params(labelsize=8)

ax2 = axes[1]
for fname, color in zip(SHOW_FEATS, FEAT_COLORS):
    vals  = df_stream[fname].values.astype(float)
    v_min = np.percentile(vals, 1)
    v_max = np.percentile(vals, 99)
    vals_n = np.clip((vals - v_min) / (v_max - v_min + 1e-9), 0, 1)
    rm = np.array([vals_n[i - W: i].mean() for i in x_roll])
    ax2.plot(t_arr[x_roll], rm, color=color, lw=1.3, label=fname.rstrip(":"))
ax2.set_xlabel("t", fontsize=9)
ax2.set_ylabel(f"normalised rolling mean (w={W})", fontsize=9)
ax2.set_title("Feature Rolling Mean (normalised)", fontsize=10, fontweight="bold")
ax2.legend(fontsize=8, framealpha=0.8)
ax2.tick_params(labelsize=8)

ax3 = axes[2]
bin_size    = N_TOTAL // 5
bin_labels  = []
gamma_rates = []
for b in range(5):
    lo, hi = b * bin_size, (b + 1) * bin_size
    seg    = labels[lo:hi]
    bin_labels.append(f"B{b+1}\n{alpha_vals[lo]:.0f}°–{alpha_vals[hi-1]:.0f}°")
    gamma_rates.append(seg.mean())

x_pos = np.arange(5)
ax3.bar(x_pos, gamma_rates, color="#EF4444", alpha=0.8, label="gamma rate")
ax3.bar(x_pos, [1-r for r in gamma_rates], bottom=gamma_rates,
        color="#2563EB", alpha=0.8, label="hadron rate")
ax3.set_xticks(x_pos)
ax3.set_xticklabels(bin_labels, fontsize=8)
ax3.set_ylabel("class proportion", fontsize=9)
ax3.set_title("Gamma Rate per fAlpha Bin", fontsize=10, fontweight="bold")
ax3.legend(fontsize=8, framealpha=0.8)
for i, r in enumerate(gamma_rates):
    ax3.text(i, r/2, f"{r:.1%}", ha="center", va="center",
             color="white", fontsize=9, fontweight="bold")
ax3.tick_params(labelsize=8)

fig.tight_layout()
out_fig = os.path.join(OUT_DIR, "magic_telescope_concept_drift.png")
fig.savefig(out_fig, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved -> {out_fig}")
print("Done.")
