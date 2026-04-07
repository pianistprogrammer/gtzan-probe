"""Regenerate ALL figures from saved data — no training, no inference, no feature extraction.

Strips the 'Figure X —' prefix that notebooks baked in; uses clean titles from the modules.

Usage:
    .venv/bin/python regen_all_figures.py
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from src.gtzan_probe.config import (
    DATA_DIR, FIGURES_DIR, GENRES, GENRE_COLORS, GENRE_PALETTE,
    BAND_RANGES, setup_plotting, savefig,
)

ISMIR_FULL = 16.0

setup_plotting()

# ── Load all saved data ──────────────────────────────────────────────────────
feat_df = pd.read_csv(DATA_DIR / "gtzan_features.csv")
meta = pd.read_csv(DATA_DIR / "gtzan_metadata.csv")
preds_df = pd.read_csv(DATA_DIR / "test_predictions.csv")
alignment_df = pd.read_csv(DATA_DIR / "musicological_alignment.csv")
faith_df = pd.read_csv(DATA_DIR / "faithfulness_results.csv")
agree_df = pd.read_csv(DATA_DIR / "method_agreement.csv")
spurious_df = pd.read_csv(DATA_DIR / "spurious_correlation.csv")

with open(DATA_DIR / "shap_values.pkl", "rb") as f:
    shap_results = pickle.load(f)
with open(DATA_DIR / "lime_results.pkl", "rb") as f:
    lime_results = pickle.load(f)
with open(DATA_DIR / "attention_results.pkl", "rb") as f:
    attention_results = pickle.load(f)

print("All data loaded.\n")


# ══════════════════════════════════════════════════════════════════════════════
# 01 — Dataset EDA figures
# ══════════════════════════════════════════════════════════════════════════════

# 01_dataset_composition
print("01_dataset_composition ...")
fig = plt.figure(figsize=(16, 6))
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

ax1 = fig.add_subplot(gs[0])
counts = meta.groupby("genre").size().reindex(GENRES)
bars = ax1.bar(GENRES, counts, color=[GENRE_COLORS[g] for g in GENRES],
               edgecolor="white", linewidth=0.8)
ax1.set_xticklabels(GENRES, rotation=45, ha="right")
ax1.set_ylabel("Number of clips")
ax1.set_title("A  Clips per Genre")
ax1.set_ylim(0, 115)
for bar, v in zip(bars, counts):
    ax1.text(bar.get_x() + bar.get_width() / 2, v + 1, str(v),
             ha="center", va="bottom", fontsize=9)

ax2 = fig.add_subplot(gs[1])
valid = meta.dropna(subset=["duration_s"])
ax2.hist(valid["duration_s"], bins=30, color="#2196F3", edgecolor="white", linewidth=0.5)
ax2.axvline(30.0, color="#F44336", linestyle="--", linewidth=1.5, label="Expected 30s")
ax2.set_xlabel("Duration (seconds)")
ax2.set_ylabel("Count")
ax2.set_title("B  Clip Duration Distribution")
ax2.legend()

ax3 = fig.add_subplot(gs[2])
fault_pivot = meta.pivot_table(index="genre", columns="fault", aggfunc="size", fill_value=0)
fault_pivot = fault_pivot.reindex(GENRES)
fault_cols = [c for c in ["clean", "corrupt", "duplicate", "mislabeled"] if c in fault_pivot.columns]
fault_pivot[fault_cols].plot(
    kind="bar", ax=ax3, stacked=True,
    color=["#4CAF50", "#F44336", "#FF9800", "#9C27B0"],
    edgecolor="white", linewidth=0.5,
)
ax3.set_xticklabels(GENRES, rotation=45, ha="right")
ax3.set_ylabel("Number of clips")
ax3.set_title("C  Fault Audit per Genre")
ax3.get_legend().remove()

fig.suptitle("GTZAN Dataset Composition & Fault Audit", fontweight="bold")
savefig("01_dataset_composition", fig)

# 01_feature_distributions
print("01_feature_distributions ...")
feature_showcase = ["tempo", "zcr_mean", "centroid_mean", "mfcc_0_mean", "rms_mean", "chroma_mean"]
feature_labels = ["Tempo (BPM)", "ZCR Mean", "Spectral Centroid", "MFCC-0 Mean", "RMS Energy", "Chroma Mean"]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes_flat = axes.flatten()
for ax, feat, label in zip(axes_flat, feature_showcase, feature_labels):
    for genre in GENRES:
        vals = feat_df[feat_df.genre == genre][feat].dropna()
        ax.hist(vals, bins=20, alpha=0.55, label=genre,
                color=GENRE_COLORS[genre], edgecolor="none")
    ax.set_title(label, fontweight="bold")
    ax.set_xlabel(label)
    ax.set_ylabel("Count")

handles = [plt.Rectangle((0, 0), 1, 1, color=GENRE_COLORS[g]) for g in GENRES]
fig.legend(handles, GENRES, loc="lower center", ncol=5, fontsize=10,
           bbox_to_anchor=(0.5, -0.02), framealpha=0.9)
fig.suptitle("Acoustic Feature Distributions by Genre", fontweight="bold")
savefig("01_feature_distributions", fig)

# 01_feature_correlation
print("01_feature_correlation ...")
num_cols = [c for c in feat_df.columns
            if feat_df[c].dtype in [np.float64, np.int64] and c != "fault"]
corr = feat_df[num_cols].corr()
fig, ax = plt.subplots(figsize=(18, 15))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            annot=False, linewidths=0.3, cbar_kws={"label": "Pearson r", "shrink": 0.7})
ax.set_title("Acoustic Feature Correlation Matrix", fontweight="bold")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
savefig("01_feature_correlation", fig)

# 01_tempo_boxplot
print("01_tempo_boxplot ...")
fig, ax = plt.subplots(figsize=(14, 6))
data_by_genre = [feat_df[feat_df.genre == g]["tempo"].dropna().values for g in GENRES]
bp = ax.boxplot(data_by_genre, patch_artist=True, notch=True,
                medianprops=dict(color="black", linewidth=2))
for patch, genre in zip(bp["boxes"], GENRES):
    patch.set_facecolor(GENRE_COLORS[genre])
    patch.set_alpha(0.8)
ax.set_xticks(range(1, 11))
ax.set_xticklabels([g.capitalize() for g in GENRES], rotation=30, ha="right")
ax.set_ylabel("Estimated Tempo (BPM)")
ax.set_title("Tempo Distribution by Genre", fontweight="bold")
savefig("01_tempo_boxplot", fig)

# 01_mfcc_heatmap
print("01_mfcc_heatmap ...")
mfcc_cols_mean = [f"mfcc_{i}_mean" for i in range(20)]
genre_mfcc = feat_df.groupby("genre")[mfcc_cols_mean].mean().reindex(GENRES)
fig, ax = plt.subplots(figsize=(14, 7))
sns.heatmap(
    genre_mfcc.T, ax=ax, cmap="coolwarm", center=0,
    xticklabels=[g.capitalize() for g in GENRES],
    yticklabels=[f"MFCC-{i}" for i in range(20)],
    cbar_kws={"label": "Mean Coefficient Value"},
    annot=True, fmt=".1f", annot_kws={"fontsize": 7},
)
ax.set_title("Mean MFCC Coefficients per Genre", fontweight="bold")
ax.set_xlabel("")
savefig("01_mfcc_heatmap", fig)

# Note: 01_spectrogram_gallery requires loading audio files, not just CSVs.
# If spectrograms were cached in a pickle we could replot them, but they
# aren't. We'll skip this one — it already has a clean title from the module.
# If you need it, run: .venv/bin/python -c "from src.gtzan_probe.prepare_dataset import fig_spectrogram_gallery; ..."
print("01_spectrogram_gallery — requires audio files; loading from saved SHAP spectrograms as proxy...")
# We can build a gallery from the SHAP pickle spectrograms (one per genre)
fig, axes = plt.subplots(2, 5, figsize=(20, 7))
axes_flat = axes.flatten()
for i, genre in enumerate(GENRES):
    if genre in shap_results:
        axes_flat[i].imshow(shap_results[genre]["spectrogram"],
                            aspect="auto", origin="lower", cmap="magma")
    axes_flat[i].set_title(genre.capitalize(), fontweight="bold", color=GENRE_COLORS[genre])
    axes_flat[i].set_xlabel("")
    axes_flat[i].set_ylabel("")
fig.suptitle("Mel Spectrogram Gallery (one clip per genre)", fontweight="bold")
savefig("01_spectrogram_gallery", fig)


# ══════════════════════════════════════════════════════════════════════════════
# 02 — Training & evaluation figures
# ══════════════════════════════════════════════════════════════════════════════

# 02_confusion_matrix & 02_per_class_metrics from test_predictions.csv
from sklearn.metrics import confusion_matrix, classification_report

true_genres = preds_df["true_genre"].values
pred_genres = preds_df["pred_genre"].values
report = classification_report(true_genres, pred_genres, target_names=GENRES, output_dict=True)

print("02_confusion_matrix ...")
cm = confusion_matrix(true_genres, pred_genres, labels=GENRES)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(20, 8))
for ax, data, fmt, title in zip(
    axes, [cm, cm_norm], ["d", ".2f"],
    ["A  Raw Counts", "B  Row-Normalised (Recall)"],
):
    sns.heatmap(
        data, ax=ax, annot=True, fmt=fmt,
        xticklabels=[g.capitalize() for g in GENRES],
        yticklabels=[g.capitalize() for g in GENRES],
        cmap="Blues", linewidths=0.5, cbar_kws={"shrink": 0.75},
    )
    ax.set_xlabel("Predicted", fontweight="bold")
    ax.set_ylabel("True", fontweight="bold")
    ax.set_title(title, fontweight="bold")

fig.suptitle("Confusion Matrix: MusicCNN on GTZAN Test Set", fontweight="bold")
savefig("02_confusion_matrix", fig)

# 02_per_class_metrics
print("02_per_class_metrics ...")
metrics_df = pd.DataFrame({
    "Genre": GENRES,
    "Precision": [report[g]["precision"] for g in GENRES],
    "Recall": [report[g]["recall"] for g in GENRES],
    "F1-Score": [report[g]["f1-score"] for g in GENRES],
})

fig, ax = plt.subplots(figsize=(16, 7))
x = np.arange(len(GENRES))
w = 0.25
colors = ["#2196F3", "#4CAF50", "#F44336"]
metrics = ["Precision", "Recall", "F1-Score"]
for i, (metric, color) in enumerate(zip(metrics, colors)):
    ax.bar(x + i * w, metrics_df[metric], w, label=metric, color=color, alpha=0.85, edgecolor="white")
ax.set_xticks(x + w)
ax.set_xticklabels([g.capitalize() for g in GENRES], rotation=30, ha="right")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.1)
ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
ax.set_title("Per-Class Classification Metrics", fontweight="bold")
ax.legend()
for i, (genre, f1) in enumerate(zip(GENRES, metrics_df["F1-Score"])):
    ax.text(i + 2 * w, f1 + 0.02, f"{f1:.2f}", ha="center", fontsize=8, color="#F44336", fontweight="bold")
savefig("02_per_class_metrics", fig)

# 02_training_curves — we don't have saved history; skip
# (The module title is already clean.)
print("02_training_curves — skipped (requires training history, not saved as CSV)")


# ══════════════════════════════════════════════════════════════════════════════
# 03 — SHAP figures
# ══════════════════════════════════════════════════════════════════════════════
shap_cmap = LinearSegmentedColormap.from_list("shap_div", ["#2196F3", "#FFFFFF", "#F44336"], N=256)
available_shap = [g for g in GENRES if g in shap_results]

def robust_signed_overlay(sv):
    """Clip extremes and compute per-pixel alpha for clearer signed attributions."""
    abs_sv = np.abs(sv)
    vmax = float(np.percentile(abs_sv, 99))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(abs_sv.max() if abs_sv.size else 1.0)
    vmax = max(vmax, 1e-8)
    sv_clip = np.clip(sv, -vmax, vmax)
    norm = np.abs(sv_clip) / vmax
    alpha = np.clip((norm - 0.15) / 0.85, 0.0, 1.0) ** 0.7
    alpha *= 0.98
    return sv_clip, alpha, vmax

print("03_shap_gallery ...")
n = len(available_shap)
cols = 5
rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows))
if rows == 1:
    axes = [axes]
axes_flat = [ax for row_axes in axes for ax in (row_axes if hasattr(row_axes, "__len__") else [row_axes])]

for i, genre in enumerate(available_shap):
    ax = axes_flat[i]
    res = shap_results[genre]
    sv = res["shap_values"]
    vmax = np.abs(sv).max()
    ax.imshow(res["spectrogram"], aspect="auto", origin="lower", cmap="gray_r", alpha=0.4)
    im = ax.imshow(sv, aspect="auto", origin="lower", cmap=shap_cmap, alpha=0.7, vmin=-vmax, vmax=vmax)
    ax.set_title(genre.capitalize(), fontsize=14, fontweight="bold", color=GENRE_COLORS[genre])
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Mel bin", fontsize=12)
    ax.tick_params(labelsize=10)

for j in range(i + 1, len(axes_flat)):
    axes_flat[j].set_visible(False)

fig.subplots_adjust(wspace=0.35, hspace=0.4)
fig.suptitle("SHAP Explanation Overlays per Genre", fontsize=16, fontweight="bold")
savefig("03_shap_gallery", fig)

# 03_shap_frequency_bands
print("03_shap_frequency_bands ...")
band_names = list(BAND_RANGES.keys())
band_data_shap = {genre: [] for genre in available_shap}
for genre in available_shap:
    sv = np.abs(shap_results[genre]["shap_values"])
    for band_name in band_names:
        lo, hi = BAND_RANGES[band_name]
        band_data_shap[genre].append(sv[lo:hi, :].mean())

band_df_shap = pd.DataFrame(band_data_shap, index=band_names).T
fig, ax = plt.subplots(figsize=(14, 7))
band_df_shap.plot(kind="bar", ax=ax, colormap="viridis", edgecolor="white", linewidth=0.5)
ax.set_xticklabels([g.capitalize() for g in available_shap], rotation=30, ha="right")
ax.set_ylabel("Mean |SHAP| Attribution")
ax.set_title("Mean Absolute SHAP per Frequency Band per Genre", fontweight="bold")
ax.legend(title="Frequency Band")
savefig("03_shap_frequency_bands", fig)


# ══════════════════════════════════════════════════════════════════════════════
# 04 — LIME figures
# ══════════════════════════════════════════════════════════════════════════════
available_lime = [g for g in GENRES if g in lime_results]
lime_cmap = LinearSegmentedColormap.from_list("lime_div", ["#2196F3", "#FFFFFF", "#4CAF50"], N=256)

print("04_lime_gallery ...")
n = len(available_lime)
cols = 5
rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
if rows == 1:
    axes = [axes]
axes_flat = [ax for row_axes in axes for ax in (row_axes if hasattr(row_axes, "__len__") else [row_axes])]

for i, genre in enumerate(available_lime):
    ax = axes_flat[i]
    res = lime_results[genre]
    ax.imshow(res["spectrogram"], aspect="auto", origin="lower", cmap="gray_r", alpha=0.4)
    mask = res["mask"].astype(float)
    vmax = max(np.abs(mask).max(), 1e-8)
    ax.imshow(mask, aspect="auto", origin="lower", cmap=lime_cmap, alpha=0.6, vmin=-vmax, vmax=vmax)
    ax.set_title(genre.capitalize(), fontweight="bold", color=GENRE_COLORS[genre])
    ax.set_xlabel("Time")
    ax.set_ylabel("Mel bin")

for j in range(i + 1, len(axes_flat)):
    axes_flat[j].set_visible(False)

fig.suptitle("LIME Superpixel Explanations per Genre", fontweight="bold")
savefig("04_lime_gallery", fig)

# 04_lime_frequency_bands
print("04_lime_frequency_bands ...")
band_data_lime = {genre: [] for genre in available_lime}
for genre in available_lime:
    res = lime_results[genre]
    segments = res["segments"]
    local_exp = res["local_exp"]
    importance_map = np.zeros_like(res["spectrogram"])
    for seg_id, weight in local_exp.items():
        importance_map[segments == seg_id] = weight
    abs_imp = np.abs(importance_map)
    for band_name in band_names:
        lo, hi = BAND_RANGES[band_name]
        band_data_lime[genre].append(abs_imp[lo:hi, :].mean())

band_df_lime = pd.DataFrame(band_data_lime, index=band_names).T
fig, ax = plt.subplots(figsize=(14, 7))
band_df_lime.plot(kind="bar", ax=ax, colormap="viridis", edgecolor="white", linewidth=0.5)
ax.set_xticklabels([g.capitalize() for g in available_lime], rotation=30, ha="right")
ax.set_ylabel("Mean |LIME| Attribution")
ax.set_title("LIME Frequency Band Importance per Genre", fontweight="bold")
ax.legend(title="Frequency Band")
savefig("04_lime_frequency_bands", fig)


# ══════════════════════════════════════════════════════════════════════════════
# 05 — Validation / quantitative figures
# ══════════════════════════════════════════════════════════════════════════════

# 05_musicological_alignment
print("05_musicological_alignment ...")
fig, ax = plt.subplots(figsize=(10, 6))
avail_align = alignment_df["genre"].tolist()
heat_data = []
for genre in avail_align:
    sv = np.abs(shap_results[genre]["shap_values"])
    row_vals = []
    for band in band_names:
        lo, hi = BAND_RANGES[band]
        row_vals.append(sv[lo:hi, :].mean())
    heat_data.append(row_vals)

heat_df = pd.DataFrame(heat_data, index=avail_align, columns=band_names)
sns.heatmap(heat_df, ax=ax, cmap="YlOrRd", annot=True, fmt=".3f",
            cbar_kws={"label": "Mean |SHAP|"})
ax.set_title("Musicological Alignment: SHAP Attribution per Band", fontweight="bold")
savefig("05_musicological_alignment", fig)

# 05_faithfulness_curves
print("05_faithfulness_curves ...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for genre in [g for g in GENRES if g in shap_results]:
    gdf = faith_df[faith_df.genre == genre]
    axes[0].plot(gdf["k_fraction"] * 100, gdf["conf_drop"], "-o",
                 label=genre, color=GENRE_COLORS[genre], linewidth=2, markersize=5)
    axes[1].plot(gdf["k_fraction"] * 100, gdf["masked_conf"], "-o",
                 label=genre, color=GENRE_COLORS[genre], linewidth=2, markersize=5)

axes[0].set_xlabel("% Pixels Removed")
axes[0].set_ylabel("Confidence Drop")
axes[0].set_title("A  Confidence Drop (higher = more faithful)")
axes[0].legend(fontsize=8, ncol=2)
axes[1].set_xlabel("% Pixels Removed")
axes[1].set_ylabel("Remaining Confidence")
axes[1].set_title("B  Remaining Confidence")
axes[1].legend(fontsize=8, ncol=2)
fig.suptitle("Faithfulness: Top-K SHAP Removal", fontweight="bold")
savefig("05_faithfulness_curves", fig)

# 05_method_agreement
print("05_method_agreement ...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
x = np.arange(len(agree_df))
w = 0.28
metrics_plot = [
    ("pearson_r", "Pearson r", "#2196F3"),
    ("spearman_r", "Spearman ρ", "#4CAF50"),
    ("top10_iou", "Top-10% IoU", "#F44336"),
]
for i, (col, label, color) in enumerate(metrics_plot):
    axes[0].bar(x + i * w, agree_df[col], w, label=label, color=color, alpha=0.85)
axes[0].set_xticks(x + w)
axes[0].set_xticklabels([g.capitalize() for g in agree_df["genre"]], rotation=30, ha="right")
axes[0].set_ylabel("Score")
axes[0].set_title("A  SHAP vs LIME Agreement per Genre")
axes[0].legend()

axes[1].scatter(agree_df["pearson_r"], agree_df["spearman_r"], s=100,
                c=[GENRE_COLORS[g] for g in agree_df["genre"]], edgecolors="black", linewidth=1)
for _, row in agree_df.iterrows():
    axes[1].annotate(row["genre"], (row["pearson_r"], row["spearman_r"]),
                     fontsize=9, ha="left", va="bottom")
axes[1].set_xlabel("Pearson r")
axes[1].set_ylabel("Spearman ρ")
axes[1].set_title("B  Correlation Agreement Scatter")
fig.suptitle("Inter-Method Agreement: SHAP vs LIME", fontweight="bold")
savefig("05_method_agreement", fig)

# 05_spurious_correlation
print("05_spurious_correlation ...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
x = np.arange(len(spurious_df))
w = 0.35
axes[0].bar(x - w / 2, spurious_df["silence_shap"], w, label="Silence regions", color="#F44336")
axes[0].bar(x + w / 2, spurious_df["high_energy_shap"], w, label="High-energy regions", color="#4CAF50")
axes[0].set_xticks(x)
axes[0].set_xticklabels([g.capitalize() for g in spurious_df["genre"]], rotation=30, ha="right")
axes[0].set_ylabel("Mean |SHAP|")
axes[0].set_title("A  Attribution: Silence vs High-Energy")
axes[0].legend()

axes[1].bar(x - w / 2, spurious_df["edge_shap"], w, label="Edge regions", color="#FF9800")
axes[1].bar(x + w / 2, spurious_df["center_shap"], w, label="Center regions", color="#2196F3")
axes[1].set_xticks(x)
axes[1].set_xticklabels([g.capitalize() for g in spurious_df["genre"]], rotation=30, ha="right")
axes[1].set_ylabel("Mean |SHAP|")
axes[1].set_title("B  Attribution: Edge vs Center")
axes[1].legend()
fig.suptitle("Spurious Correlation Detection", fontweight="bold")
savefig("05_spurious_correlation", fig)


# ══════════════════════════════════════════════════════════════════════════════
# 06 — Attention figures
# ══════════════════════════════════════════════════════════════════════════════
available_attn = [g for g in GENRES if g in attention_results]

# 06_attention_gallery
print("06_attention_gallery ...")
n = len(available_attn)
cols = 5
rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
axes_flat = np.array(axes).flatten() if n > 1 else [axes]

for i, genre in enumerate(available_attn):
    ax = axes_flat[i]
    attn = attention_results[genre]["cls_attention"]
    side = int(np.sqrt(len(attn)))
    if side * side <= len(attn):
        attn_2d = attn[: side * side].reshape(side, side)
    else:
        attn_2d = attn.reshape(1, -1)
    ax.imshow(attn_2d, aspect="auto", cmap="hot", origin="lower")
    ax.set_title(genre.capitalize(), fontweight="bold", color=GENRE_COLORS[genre])
    ax.set_xlabel("Patch")
    ax.set_ylabel("Patch")

for j in range(i + 1, len(axes_flat)):
    axes_flat[j].set_visible(False)

fig.suptitle("CLS Attention Maps (AST last layer)", fontweight="bold")
savefig("06_attention_gallery", fig)

# 06_ast_training_curves — skipped (needs training history)
print("06_ast_training_curves — skipped (requires training history, not saved as CSV)")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL — Publication hero figures (from generate_figures.py)
# ══════════════════════════════════════════════════════════════════════════════
print("FINAL_01_shap_hero ...")
fig, axes = plt.subplots(2, 5, figsize=(ISMIR_FULL, 6))
axes_flat = axes.flatten()
for i, genre in enumerate(available_shap):
    ax = axes_flat[i]
    res = shap_results[genre]
    sv = res["shap_values"]
    sv_clip, alpha_mask, vmax = robust_signed_overlay(sv)
    ax.imshow(res["spectrogram"], aspect="auto", origin="lower", cmap="gray", alpha=0.52)
    im = ax.imshow(sv_clip, aspect="auto", origin="lower", cmap=shap_cmap, alpha=alpha_mask, vmin=-vmax, vmax=vmax)
    ax.set_title(genre.capitalize(), fontsize=10, fontweight="bold", color=GENRE_COLORS[genre])
    if i >= 5:
        ax.set_xlabel("Time", fontsize=8)
    if i % 5 == 0:
        ax.set_ylabel("Mel bin", fontsize=8)
for j in range(len(available_shap), len(axes_flat)):
    axes_flat[j].set_visible(False)
fig.subplots_adjust(right=0.90, wspace=0.20, hspace=0.25)
cax = fig.add_axes([0.92, 0.14, 0.012, 0.72])
cbar = fig.colorbar(im, cax=cax)
cbar.set_label("SHAP value\n(blue = negative, red = positive)", fontsize=8)
cbar.ax.tick_params(labelsize=7)
fig.suptitle("SHAP Attribution Overlays Across All GTZAN Genres", fontweight="bold")
savefig("FINAL_01_shap_hero", fig, tight=False)

print("FINAL_02_quantitative_panel ...")
fig = plt.figure(figsize=(ISMIR_FULL, 9))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

ax_a = fig.add_subplot(gs[0, 0])
x_a = np.arange(len(alignment_df))
colors_a = [GENRE_COLORS.get(g, "#999") for g in alignment_df["genre"]]
ax_a.bar(x_a, alignment_df["spearman_r"], color=colors_a, edgecolor="white")
ax_a.set_xticks(x_a)
ax_a.set_xticklabels([g[:4].capitalize() for g in alignment_df["genre"]], rotation=45, ha="right", fontsize=8)
ax_a.set_ylabel("Spearman ρ")
ax_a.set_title("A  Musicological Alignment", fontweight="bold")
ax_a.axhline(0, color="black", linewidth=0.5)

ax_b = fig.add_subplot(gs[0, 1])
for genre in [g for g in GENRES if g in faith_df["genre"].values]:
    gdf = faith_df[faith_df.genre == genre]
    ax_b.plot(gdf["k_fraction"] * 100, gdf["conf_drop"], "-o",
              color=GENRE_COLORS[genre], label=genre, markersize=3, linewidth=1.5)
ax_b.set_xlabel("% Pixels Removed")
ax_b.set_ylabel("Conf. Drop")
ax_b.set_title("B  Faithfulness (SHAP removal)", fontweight="bold")
ax_b.legend(fontsize=6, ncol=2)

ax_c = fig.add_subplot(gs[1, 0])
w_c = 0.35
x_c = np.arange(len(agree_df))
ax_c.bar(x_c - w_c / 2, agree_df["pearson_r"], w_c, label="Pearson r", color="#2196F3")
ax_c.bar(x_c + w_c / 2, agree_df["top10_iou"], w_c, label="Top-10% IoU", color="#F44336")
ax_c.set_xticks(x_c)
ax_c.set_xticklabels([g[:4].capitalize() for g in agree_df["genre"]], rotation=45, ha="right", fontsize=8)
ax_c.set_ylabel("Score")
ax_c.set_title("C  SHAP vs LIME Agreement", fontweight="bold")
ax_c.legend(fontsize=8)

ax_d = fig.add_subplot(gs[1, 1])
x_d = np.arange(len(spurious_df))
ax_d.bar(x_d, spurious_df["silence_ratio"], color="#F44336", alpha=0.8, label="Silence/Signal ratio")
ax_d.set_xticks(x_d)
ax_d.set_xticklabels([g[:4].capitalize() for g in spurious_df["genre"]], rotation=45, ha="right", fontsize=8)
ax_d.set_ylabel("Ratio")
ax_d.set_title("D  Spurious: Silence Attribution Ratio", fontweight="bold")

fig.suptitle("Quantitative XAI Evaluation Results", fontweight="bold", fontsize=14)
savefig("FINAL_02_quantitative_panel", fig)


# ══════════════════════════════════════════════════════════════════════════════
# Done — copy to Paper/
# ══════════════════════════════════════════════════════════════════════════════
import shutil
paper_dir = FIGURES_DIR.parent / "Paper"
if paper_dir.exists():
    for pdf in FIGURES_DIR.glob("*.pdf"):
        shutil.copy2(pdf, paper_dir / pdf.name)
    print(f"\n✓ All PDFs copied to {paper_dir}")

print("\n✓ All figures regenerated with clean titles (no 'Figure X —' prefix).")
