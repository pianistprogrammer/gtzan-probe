"""Regenerate Figure 1 as horizontal bar chart (genres on Y-axis)."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("/Volumes/AI/Projects/gtzan-probe/results/all_shap_lime/genre_aggregates.csv")
out = Path("/Volumes/AI/Projects/gtzan-probe/Paper")

GENRES = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]
BANDS = ["B1","B2","B3","B4"]
BAND_LABELS = ["B1 (0–284 Hz)","B2 (284–800 Hz)","B3 (800–3 kHz)","B4 (3–11 kHz)"]
BAND_COLORS = ["#2196F3","#FF9800","#4CAF50","#E91E63"]

plt.rcParams.update({
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

for ax, prefix, xlabel, title in [
    (axes[0], "shap_abs", "Mean |SHAP| (×10⁻³)", "(A) SHAP"),
    (axes[1], "lime_abs", "Mean |LIME weight|",   "(B) LIME"),
]:
    y = np.arange(len(GENRES))
    h = 0.18
    scale = 1000 if prefix == "shap_abs" else 1

    for i, (b, bl, bc) in enumerate(zip(BANDS, BAND_LABELS, BAND_COLORS)):
        means = [float(df.loc[df.genre==g, f"{prefix}_{b}_mean"].iloc[0]) * scale for g in GENRES]
        stds  = [float(df.loc[df.genre==g, f"{prefix}_{b}_std"].iloc[0])  * scale for g in GENRES]
        offset = (i - 1.5) * h
        bars = ax.barh(y + offset, means, h,
                       label=bl, color=bc, alpha=0.88,
                       xerr=stds, error_kw={"linewidth": 0.7, "capsize": 2})

    ax.set_yticks(y)
    ax.set_yticklabels(GENRES, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_title(title, fontsize=10, pad=4)
    ax.legend(title="Band", fontsize=7, title_fontsize=7,
              loc="lower right", framealpha=0.85)
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.invert_yaxis()  # blues at top, rock at bottom

ns = {g: int(df.loc[df.genre==g,"n"].iloc[0]) for g in GENRES}
n_str = ", ".join(f"{g}={ns[g]}" for g in GENRES)
fig.suptitle(
    f"Per-genre band attribution: mean ± std over all correctly classified clips  (N: {n_str})",
    fontsize=8, y=1.01
)
fig.tight_layout()
p = out / "all_recordings_shap_bands.pdf"
fig.savefig(p, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved {p}")
