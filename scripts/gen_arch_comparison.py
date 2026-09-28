"""Generate architecture comparison figure: MusicCNN vs MusicResNet SHAP band profiles."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Load both band profile CSVs
cnn_df = pd.read_csv("/Volumes/AI/Projects/gtzan-probe/results/revision_audit/band_profiles.csv")
res_df = pd.read_csv("/Volumes/AI/Projects/gtzan-probe/results/resnet_explanations/resnet_band_profiles.csv")
out = Path("/Volumes/AI/Projects/gtzan-probe/Paper")

GENRES = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]
BANDS  = ["B1","B2","B3","B4"]
BAND_LABELS = ["B1\n(0–284 Hz)","B2\n(284–800 Hz)","B3\n(800–3 kHz)","B4\n(3–11 kHz)"]
COLORS = ["#2196F3","#F44336","#4CAF50","#FF9800"]

plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

for ax, df, title, scale in [
    (axes[0], cnn_df,  "(A) MusicCNN (plain conv)", 1000),
    (axes[1], res_df,  "(B) MusicResNet (residual)", 1000),
]:
    # cnn_df uses column name "shap_mean_abs", resnet uses "shap_abs_B1" etc.
    # Detect column naming
    if "shap_mean_abs" in df.columns:
        col_fn = lambda b: f"shap_mean_abs"   # not per-band in cnn_df
    else:
        col_fn = lambda b: f"shap_abs_{b}"

    x = np.arange(len(GENRES)); w = 0.19
    for i, (b, bl, bc) in enumerate(zip(BANDS, BAND_LABELS, COLORS)):
        col = f"shap_abs_{b}" if f"shap_abs_{b}" in df.columns else "shap_mean_abs"
        # For cnn_df (band_profiles.csv), it has shap_mean_abs per band per genre
        if "band" in df.columns:
            vals = [float(df[(df.genre==g)&(df.band==b)]["shap_mean_abs"].iloc[0])*scale
                    if len(df[(df.genre==g)&(df.band==b)])>0 else 0.0 for g in GENRES]
        else:
            vals = [float(df[df.genre==g][col].iloc[0])*scale
                    if len(df[df.genre==g])>0 else 0.0 for g in GENRES]
        ax.bar(x + i*w, vals, w, label=bl, color=bc, alpha=0.88)

    ax.set_xticks(x + 1.5*w)
    ax.set_xticklabels(GENRES, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Mean |SHAP| (×10⁻³)", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(title="Band", fontsize=7, title_fontsize=7, loc="upper right")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)

fig.suptitle("Architecture comparison: SHAP band attribution on the same 10 clips\n"
             "(MusicCNN val acc 83.4%; MusicResNet val acc 83.9%; identical splits and preprocessing)",
             fontsize=9, y=1.02)
fig.tight_layout()
p = out / "architecture_comparison.pdf"
fig.savefig(p, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved {p}")
