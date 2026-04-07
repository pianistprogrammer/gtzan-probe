"""Quick script to regenerate SHAP gallery figure from saved pickle."""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from src.gtzan_probe.config import (
    DATA_DIR, GENRES, GENRE_COLORS, setup_plotting, savefig,
)

setup_plotting()

with open(DATA_DIR / "shap_values.pkl", "rb") as f:
    shap_results = pickle.load(f)

shap_cmap = LinearSegmentedColormap.from_list(
    "shap_div", ["#2196F3", "#FFFFFF", "#F44336"], N=256
)
available = [g for g in GENRES if g in shap_results]
n = len(available)
cols = 5
rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows))
if rows == 1:
    axes = [axes]
axes_flat = [
    ax for row_axes in axes
    for ax in (row_axes if hasattr(row_axes, "__len__") else [row_axes])
]

for i, genre in enumerate(available):
    ax = axes_flat[i]
    res = shap_results[genre]
    sv = res["shap_values"]
    vmax = np.abs(sv).max()
    ax.imshow(res["spectrogram"], aspect="auto", origin="lower", cmap="gray_r", alpha=0.4)
    im = ax.imshow(sv, aspect="auto", origin="lower", cmap=shap_cmap,
                   alpha=0.7, vmin=-vmax, vmax=vmax)
    ax.set_title(f"{genre.capitalize()}", fontsize=14, fontweight="bold",
                 color=GENRE_COLORS[genre])
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Mel bin", fontsize=12)
    ax.tick_params(labelsize=10)

for j in range(i + 1, len(axes_flat)):
    axes_flat[j].set_visible(False)

fig.subplots_adjust(wspace=0.35, hspace=0.4)
fig.suptitle("SHAP Explanation Overlays per Genre", fontsize=16, fontweight="bold")
savefig("03_shap_gallery", fig)
print("Done — regenerated 03_shap_gallery")
