"""Generate GTZAN vs FMA-small comparison figure for shared genres (hip-hop, pop, rock)."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

gtzan = pd.read_csv("/Volumes/AI/Projects/gtzan-probe/results/all_shap_lime/genre_aggregates.csv")
fma   = pd.read_csv("/Volumes/AI/Projects/gtzan-probe/Paper/results/fma_analysis/fma_genre_aggregates.csv")
out   = Path("/Volumes/AI/Projects/gtzan-probe/Paper")

SHARED = ["hiphop", "pop", "rock"]
BANDS  = ["B1", "B2", "B3", "B4"]
BAND_LABELS = ["B1\n(0–284 Hz)", "B2\n(284–800 Hz)", "B3\n(800–3 kHz)", "B4\n(3–11 kHz)"]
COLORS = {"hiphop": "#9C27B0", "pop": "#8BC34A", "rock": "#607D8B"}
HATCH  = {"GTZAN": "", "FMA": "//"}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

for ax, prefix, ylabel, scale, title in [
    (axes[0], "shap_abs", "Mean |SHAP| (×10⁻³)", 1000, "(A) SHAP band attribution"),
    (axes[1], "input",    "Mean normalised log-mel", 1,   "(B) Input spectrum (baseline)"),
]:
    x = np.arange(len(BANDS)); w = 0.13
    for gi, genre in enumerate(SHARED):
        c = COLORS[genre]
        # GTZAN
        g_row = gtzan[gtzan.genre == genre]
        vals_g = [float(g_row[f"{prefix}_{b}_mean"].iloc[0]) * scale for b in BANDS]
        errs_g = [float(g_row[f"{prefix}_{b}_std"].iloc[0])  * scale for b in BANDS]
        offset_g = (gi * 2) * w - 2.5 * w
        ax.bar(x + offset_g, vals_g, w, label=f"{genre} GTZAN",
               color=c, alpha=0.9, yerr=errs_g, capsize=2,
               error_kw={"linewidth": 0.7})
        # FMA
        f_row = fma[fma.genre_gtzan == genre]
        if len(f_row) > 0:
            vals_f = [float(f_row[f"{prefix}_{b}_mean"].iloc[0]) * scale for b in BANDS]
            errs_f = [float(f_row[f"{prefix}_{b}_std"].iloc[0])  * scale for b in BANDS]
            offset_f = (gi * 2 + 1) * w - 2.5 * w
            ax.bar(x + offset_f, vals_f, w, label=f"{genre} FMA",
                   color=c, alpha=0.5, hatch="//", yerr=errs_f, capsize=2,
                   error_kw={"linewidth": 0.7})

    ax.set_xticks(x); ax.set_xticklabels(BAND_LABELS, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9); ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)

fig.suptitle(
    "GTZAN vs FMA-small SHAP band attribution for shared genre labels\n"
    "(solid = GTZAN, hatched = FMA-small; genre-label equivalence is not assumed)",
    fontsize=9, y=1.02
)
fig.tight_layout()
p = out / "fma_comparison.pdf"
fig.savefig(p, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved {p}")
