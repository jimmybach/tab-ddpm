"""
Real Concept Drift — Covertype Dataset (OpenML data_id=1596).

Design (4,000 instances, 40 windows × 100):
  Uniform stratified sampling across 20 elevation bands × 200 per band.
  Preserves the full elevation range and the complete incremental drift trajectory.

Output:
  covertype_concept_drift.csv
  covertype_concept_drift.png
"""

import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

OUT_DIR  = os.path.dirname(os.path.abspath(__file__))
N_TOTAL  = 4_000
N_BANDS  = 20
W        = 100
SEED     = 42

CLASS_NAMES = {
    "1":"Spruce/Fir","2":"Lodgepole Pine","3":"Ponderosa Pine",
    "4":"Cottonwood/Willow","5":"Aspen","6":"Douglas-fir","7":"Krummholz"
}
CLASS_COLORS = {
    "1":"#2563EB","2":"#EF4444","3":"#16A34A",
    "4":"#D97706","5":"#7C3AED","6":"#0891B2","7":"#DB2777"
}

# ── 1. load & sort by elevation ───────────────────────────────────────────────
print("Loading Covertype from OpenML...")
ds  = fetch_openml(data_id=1596, as_frame=True, parser="auto")
df  = ds.frame
target    = "class"
feat_cols = [c for c in df.columns if c != target]
print(f"  full shape: {df.shape}")

df_sorted = df.sort_values("Elevation").reset_index(drop=True)
n_full    = len(df_sorted)

# ── 2. stratified sampling: 20 bands × 200 = 4,000 ──────────────────────────
rng      = np.random.default_rng(SEED)
per_band = N_TOTAL // N_BANDS   # 200 per band
sampled_idx = []
for b in range(N_BANDS):
    lo  = int(b * n_full / N_BANDS)
    hi  = int((b + 1) * n_full / N_BANDS)
    idx = rng.choice(np.arange(lo, hi), per_band, replace=False)
    sampled_idx.extend(idx)

df_stream = df_sorted.iloc[sampled_idx].reset_index(drop=True)
print(f"  sampled: {len(df_stream)} rows, elevation "
      f"{df_stream['Elevation'].min():.0f}–{df_stream['Elevation'].max():.0f}m")

# ── 3. write CSV ──────────────────────────────────────────────────────────────
out_csv = os.path.join(OUT_DIR, "covertype_concept_drift.csv")
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["t"] + feat_cols + ["y"])
    for t, (_, row) in enumerate(df_stream.iterrows()):
        writer.writerow([t] + [row[c] for c in feat_cols] + [int(row[target])])
print(f"  saved -> {out_csv}")

# ── 4. visualize ──────────────────────────────────────────────────────────────
labels = np.array([int(r) for r in df_stream[target]])
elev   = df_stream["Elevation"].values.astype(float)
t_arr  = np.arange(N_TOTAL)
x_roll = np.arange(W, N_TOTAL)

fig, axes = plt.subplots(1, 3, figsize=(18, 4))
fig.suptitle(
    "Real Concept Drift — Covertype (sorted by Elevation, 20 bands × 200)\n"
    "Incremental drift: terrain→vegetation mapping changes as altitude rises",
    fontsize=11, fontweight="bold"
)

ax = axes[0]
for cls in ["1","2","3","7"]:
    prop = np.array([np.mean(labels[i - W: i] == int(cls)) for i in x_roll])
    ax.plot(t_arr[x_roll], prop, color=CLASS_COLORS[cls], lw=1.4,
            label=f"C{cls}: {CLASS_NAMES[cls]}")
ax.set_ylim(-0.02, 1.05)
ax.set_xlabel("t  (increasing elevation →)", fontsize=9)
ax.set_ylabel(f"class proportion (w={W})", fontsize=9)
ax.set_title("Rolling Class Proportion", fontsize=10, fontweight="bold")
ax.legend(fontsize=7.5, framealpha=0.8)
ax.tick_params(labelsize=8)

ax2 = axes[1]
ax2.plot(t_arr, elev, color="#2563EB", lw=0.8, alpha=0.7)
rm_elev = np.array([elev[i - W: i].mean() for i in x_roll])
ax2.plot(t_arr[x_roll], rm_elev, color="#EF4444", lw=1.8,
         label=f"rolling mean (w={W})")
ax2.set_xlabel("t", fontsize=9)
ax2.set_ylabel("Elevation (m)", fontsize=9)
ax2.set_title("Elevation Profile (increasing)", fontsize=10, fontweight="bold")
ax2.legend(fontsize=8)
ax2.tick_params(labelsize=8)

ax3 = axes[2]
zone_size   = N_TOTAL // 5
zone_labels = []
all_classes = ["1","2","3","7","other"]
zone_data   = {c: [] for c in all_classes}
for z in range(5):
    lo, hi = z * zone_size, (z + 1) * zone_size
    seg    = labels[lo:hi]
    zone_labels.append(f"Z{z+1}\n{elev[lo]:.0f}–{elev[hi-1]:.0f}m")
    for cls in ["1","2","3","7"]:
        zone_data[cls].append(np.mean(seg == int(cls)))
    zone_data["other"].append(np.mean(~np.isin(seg, [1,2,3,7])))

x_pos  = np.arange(5)
bottom = np.zeros(5)
colors = [CLASS_COLORS["1"],CLASS_COLORS["2"],CLASS_COLORS["3"],
          CLASS_COLORS["7"],"#9CA3AF"]
for cls, color in zip(all_classes, colors):
    vals = np.array(zone_data[cls])
    ax3.bar(x_pos, vals, bottom=bottom, color=color, alpha=0.85,
            label=f"C{cls}" if cls != "other" else "other")
    bottom += vals
ax3.set_xticks(x_pos)
ax3.set_xticklabels(zone_labels, fontsize=7)
ax3.set_ylabel("class proportion", fontsize=9)
ax3.set_title("Class Distribution per Elevation Zone", fontsize=10, fontweight="bold")
ax3.legend(fontsize=7, loc="upper right", framealpha=0.8)
ax3.tick_params(labelsize=8)

fig.tight_layout()
out_fig = os.path.join(OUT_DIR, "covertype_concept_drift.png")
fig.savefig(out_fig, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved -> {out_fig}")
print("Done.")
