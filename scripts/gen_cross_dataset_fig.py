"""Generate the 2×2 cross-dataset SHAP comparison figure."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Load all four conditions
gtzan_on_gtzan = pd.read_csv("/Volumes/AI/Projects/gtzan-probe/results/all_shap_lime/genre_aggregates.csv")
gtzan_on_fma   = pd.read_csv("/Volumes/AI/Projects/gtzan-probe/results/cross_dataset/gtzan_model_on_fma_aggregates.csv")
fma_on_fma     = pd.read_csv("/Volumes/AI/Projects/gtzan-probe/results/fma_analysis/fma_genre_aggregates.csv")
fma_on_gtzan   = pd.read_csv("/Volumes/AI/Projects/gtzan-probe/results/cross_dataset/fma_model_on_gtzan_aggregates.csv")

out = Path("/Volumes/AI/Projects/gtzan-probe/Paper")

SHARED = ["hiphop", "pop", "rock"]
BANDS  = ["B1", "B2", "B3", "B4"]
BAND_LABELS = ["B1\n(0–284 Hz)", "B2\n(284–800 Hz)", "B3\n(800–3 kHz)", "B4\n(3–11 kHz)"]
COLORS = {"hiphop": "#9C27B0", "pop": "#8BC34A", "rock": "#607D8B"}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})

# Helper to get mean value safely
def get_mean(df, genre, band, genre_col="genre"):
    col = f"shap_abs_{band}_mean"
    rows = df[df[genre_col] == genre]
    return float(rows[col].iloc[0]) * 1000 if len(rows) > 0 and col in rows.columns else 0.0

def get_std(df, genre, band, genre_col="genre"):
    col = f"shap_abs_{band}_std"
    rows = df[df[genre_col] == genre]
    return float(rows[col].iloc[0]) * 1000 if len(rows) > 0 and col in rows.columns else 0.0

# 2×2 figure
fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
fig.suptitle("Cross-dataset SHAP band attribution: 2×2 transfer design\n"
             "(rows = model source, columns = data source; shared genres hip-hop/pop/rock only)",
             fontsize=10, y=1.01)

configs = [
    (axes[0,0], gtzan_on_gtzan, "genre",      "(A) GTZAN model on GTZAN data\n(within-dataset, 10-class, all genres)",   "blue"),
    (axes[0,1], gtzan_on_fma,   "genre",      "(B) GTZAN model on FMA-small data\n(transfer; acc: hiphop=57%, pop=1%, rock=22%)", "red"),
    (axes[1,0], fma_on_gtzan,   "genre",      "(C) FMA model on GTZAN data\n(transfer; acc: hiphop=95%, pop=25%, rock=40%)",    "red"),
    (axes[1,1], fma_on_fma,     "genre_gtzan","(D) FMA model on FMA-small data\n(within-dataset, 3-class)", "blue"),
]

for ax, df, gcol, title, border_color in configs:
    x = np.arange(len(BANDS)); w = 0.22
    for gi, genre in enumerate(SHARED):
        c = COLORS[genre]
        vals = [get_mean(df, genre, b, gcol) for b in BANDS]
        errs = [get_std(df, genre, b, gcol) for b in BANDS]
        offset = (gi - 1) * w
        ax.bar(x + offset, vals, w, label=genre, color=c, alpha=0.85,
               yerr=errs, capsize=2, error_kw={"linewidth": 0.7})
    ax.set_xticks(x)
    ax.set_xticklabels(BAND_LABELS, fontsize=8)
    ax.set_ylabel("Mean |SHAP| (×10⁻³)", fontsize=8)
    ax.set_title(title, fontsize=8.5, pad=4)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    # Red border = transfer condition, blue = within-dataset
    for spine in ax.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(1.5 if border_color == "red" else 0.8)

fig.tight_layout()
p = out / "cross_dataset_shap.pdf"
fig.savefig(p, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved {p}")
