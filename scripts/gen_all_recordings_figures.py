"""Generate revision figures from all-recordings SHAP/LIME results."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("results/all_shap_lime/genre_aggregates.csv")
out = Path("figures/revision")
out.mkdir(parents=True, exist_ok=True)

GENRES = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]
BANDS = ["B1","B2","B3","B4"]
BAND_LABELS = ["B1 (0–284 Hz)","B2 (284–800 Hz)","B3 (800–3 kHz)","B4 (3–11 kHz)"]
COLORS = ["#2196F3","#F44336","#4CAF50","#FF9800","#9C27B0","#00BCD4","#E91E63","#8BC34A","#FF5722","#607D8B"]
GCOLOR = dict(zip(GENRES, COLORS))

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42})

# --- Figure 1: per-genre SHAP & LIME band means with std bars ----------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ns = {g: int(df.loc[df.genre == g, "n"].iloc[0]) for g in GENRES}
n_label = ", ".join(f"{g[:3]}={ns[g]}" for g in GENRES)

for ax, prefix, ylabel, title_suffix in [
    (axes[0], "shap_abs", "Mean |SHAP| (×10⁻³)", "(A) SHAP"),
    (axes[1], "lime_abs", "Mean |LIME weight|", "(B) LIME"),
]:
    x = np.arange(len(GENRES)); w = 0.19
    for i, (b, bl) in enumerate(zip(BANDS, BAND_LABELS)):
        col = f"{prefix}_{b}_mean"
        err = f"{prefix}_{b}_std"
        scale = 1000 if prefix == "shap_abs" else 1
        means = [float(df.loc[df.genre == g, col].iloc[0]) * scale for g in GENRES]
        stds  = [float(df.loc[df.genre == g, err].iloc[0]) * scale for g in GENRES]
        ax.bar(x + i * w, means, w, label=bl, yerr=stds, capsize=2,
               error_kw={"linewidth": 0.8})
    ax.set_xticks(x + 1.5 * w)
    ax.set_xticklabels(GENRES, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title_suffix + " band attribution", fontsize=10)
    ax.legend(title="Band", fontsize=8, title_fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

fig.suptitle(
    "Per-genre band attribution: mean ± std over all correctly classified clips "
    "(N: " + n_label + ")",
    fontsize=9, y=1.01
)
fig.tight_layout()
p1 = out / "all_recordings_shap_bands.pdf"
fig.savefig(p1, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved {p1}")

# --- Figure 2: B1 vs B4 scatter (SHAP) ---------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
for _, row in df.iterrows():
    g = row.genre
    b1m = float(row.shap_abs_B1_mean) * 1000
    b4m = float(row.shap_abs_B4_mean) * 1000
    b1s = float(row.shap_abs_B1_std) * 1000
    b4s = float(row.shap_abs_B4_std) * 1000
    c = GCOLOR.get(g, "gray")
    ax.errorbar(b1m, b4m, xerr=b1s, yerr=b4s, fmt="o", color=c, capsize=4,
                markersize=8, label=g, linewidth=1.2)
    ax.annotate(g, (b1m, b4m), textcoords="offset points", xytext=(5, 3), fontsize=8)
ax.set_xlabel("Mean |SHAP| in B1 (×10⁻³, low freq 0–284 Hz)")
ax.set_ylabel("Mean |SHAP| in B4 (×10⁻³, high freq 3–11 kHz)")
ax.set_title("SHAP low-vs-high frequency attribution per genre\n(mean ± std, all correct clips)",
             fontsize=10)
ax.grid(alpha=0.3)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
fig.tight_layout()
p2 = out / "shap_b1_vs_b4_scatter.pdf"
fig.savefig(p2, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved {p2}")
