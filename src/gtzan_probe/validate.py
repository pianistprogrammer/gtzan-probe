"""Step 5 — Quantitative XAI validation: musicological alignment, faithfulness,
inter-method agreement, spurious correlation detection.

Usage:
    python -m gtzan_probe.validate
"""

import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from pathlib import Path
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F

from .config import (
    DATA_DIR, FIGURES_DIR, MODELS_DIR, GENRES, GENRE_COLORS,
    SR, N_MELS, N_FFT, HOP, BAND_RANGES, GT_BANDS, SEG_S,
    setup_plotting, savefig, get_device,
)

GENRE_DISPLAY = {
    "blues": "Blues", "classical": "Classical", "country": "Country",
    "disco": "Disco", "hiphop": "Hip-hop", "jazz": "Jazz",
    "metal": "Metal", "pop": "Pop", "reggae": "Reggae", "rock": "Rock",
}
from .model import MusicCNN
from .dataset import load_spectrogram


def musicological_alignment(abs_shap, genre_gt):
    """Spearman correlation between model attribution and musicological ground truth per frequency band."""
    band_names = list(BAND_RANGES.keys())
    model_weights = []
    gt_weights = []
    for band_name in band_names:
        lo, hi = BAND_RANGES[band_name]
        model_weights.append(abs_shap[lo:hi, :].mean())
        gt_weights.append(genre_gt.get(band_name, 0.0))

    if np.std(model_weights) == 0 or np.std(gt_weights) == 0:
        return 0.0, 1.0
    r, p = stats.spearmanr(model_weights, gt_weights)
    return r, p


def main():
    setup_plotting()
    device = get_device()
    print(f"Device: {device}")

    # ── Load artifacts ────────────────────────────────────────────────────────
    model = MusicCNN().to(device)
    model.load_state_dict(torch.load(MODELS_DIR / "cnn_gtzan_best.pt", map_location=device))
    model.eval()

    with open(MODELS_DIR / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    with open(DATA_DIR / "shap_values.pkl", "rb") as f:
        shap_results = pickle.load(f)

    with open(DATA_DIR / "lime_results.pkl", "rb") as f:
        lime_results = pickle.load(f)

    preds_df = pd.read_csv(DATA_DIR / "test_predictions.csv")
    print("All artifacts loaded.")

    # ══════════════════════════════════════════════════════════════════════════
    # METRIC 1: Musicological Alignment Score
    # ══════════════════════════════════════════════════════════════════════════
    alignment_rows = []
    for genre in GENRES:
        if genre not in shap_results:
            continue
        abs_sv = np.abs(shap_results[genre]["shap_values"])
        r, p = musicological_alignment(abs_sv, GT_BANDS[genre])
        alignment_rows.append({"genre": genre, "spearman_r": r, "p_value": p})

    alignment_df = pd.DataFrame(alignment_rows)
    alignment_df.to_csv(DATA_DIR / "musicological_alignment.csv", index=False)
    print("\n── Musicological Alignment ──")
    print(alignment_df.to_string(index=False))

    # Figure: alignment heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    available = alignment_df["genre"].tolist()
    band_names = list(BAND_RANGES.keys())

    heat_data = []
    for genre in available:
        sv = np.abs(shap_results[genre]["shap_values"])
        row_vals = []
        for band in band_names:
            lo, hi = BAND_RANGES[band]
            row_vals.append(sv[lo:hi, :].mean())
        heat_data.append(row_vals)

    heat_df = pd.DataFrame(heat_data, index=[GENRE_DISPLAY.get(g, g.capitalize()) for g in available], columns=band_names)
    sns.heatmap(heat_df, ax=ax, cmap="YlOrRd", annot=True, fmt=".3f",
                cbar_kws={"label": "Mean |SHAP|"})
    ax.set_title("Musicological Alignment: SHAP Attribution per Band", fontweight="bold")
    savefig("05_musicological_alignment", fig)

    # ══════════════════════════════════════════════════════════════════════════
    # METRIC 2: Faithfulness — Top-K SHAP removal test
    # ══════════════════════════════════════════════════════════════════════════
    k_fractions = [0.05, 0.1, 0.2, 0.3, 0.5]
    faith_rows = []

    for genre in tqdm(GENRES, desc="Faithfulness test"):
        if genre not in shap_results:
            continue
        res = shap_results[genre]
        spec_np, spec_tensor = load_spectrogram(res["filepath"])
        spec_tensor = spec_tensor.to(device)

        # Original prediction confidence
        with torch.no_grad():
            orig_probs = F.softmax(model(spec_tensor), dim=1).cpu().numpy()[0]
        pred_idx = le.transform([genre])[0]
        orig_conf = orig_probs[pred_idx]

        abs_sv = np.abs(res["shap_values"])
        flat_idx = np.argsort(abs_sv.ravel())[::-1]

        for k_frac in k_fractions:
            n_remove = int(k_frac * abs_sv.size)
            masked = spec_np.copy()
            top_k = flat_idx[:n_remove]
            rows_k, cols_k = np.unravel_index(top_k, abs_sv.shape)
            masked[rows_k, cols_k] = 0.0

            masked_tensor = torch.tensor(masked, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                new_probs = F.softmax(model(masked_tensor), dim=1).cpu().numpy()[0]
            new_conf = new_probs[pred_idx]

            faith_rows.append({
                "genre": genre,
                "k_fraction": k_frac,
                "orig_conf": orig_conf,
                "masked_conf": new_conf,
                "conf_drop": orig_conf - new_conf,
            })

    faith_df = pd.DataFrame(faith_rows)
    faith_df.to_csv(DATA_DIR / "faithfulness_results.csv", index=False)

    # Figure: faithfulness curves
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

    # ══════════════════════════════════════════════════════════════════════════
    # METRIC 3: Inter-method agreement (SHAP vs LIME)
    # ══════════════════════════════════════════════════════════════════════════
    agree_rows = []
    common_genres = [g for g in GENRES if g in shap_results and g in lime_results]

    for genre in common_genres:
        sv = shap_results[genre]["shap_values"]

        # Build LIME importance map from segments
        lres = lime_results[genre]
        segments = lres["segments"]
        local_exp = lres["local_exp"]
        lime_map = np.zeros_like(lres["spectrogram"])
        for seg_id, weight in local_exp.items():
            lime_map[segments == seg_id] = weight

        # Ensure same shape
        min_h = min(sv.shape[0], lime_map.shape[0])
        min_w = min(sv.shape[1], lime_map.shape[1])
        sv_crop = sv[:min_h, :min_w].ravel()
        lime_crop = lime_map[:min_h, :min_w].ravel()

        pr, _ = stats.pearsonr(sv_crop, lime_crop)
        sr, _ = stats.spearmanr(sv_crop, lime_crop)

        # Top-K overlap (IoU of top 10% pixels)
        k = int(0.1 * len(sv_crop))
        shap_topk = set(np.argsort(np.abs(sv_crop))[-k:])
        lime_topk = set(np.argsort(np.abs(lime_crop))[-k:])
        iou = len(shap_topk & lime_topk) / len(shap_topk | lime_topk)

        agree_rows.append({
            "genre": genre,
            "pearson_r": pr,
            "spearman_r": sr,
            "top10_iou": iou,
        })

    agree_df = pd.DataFrame(agree_rows)
    agree_df.to_csv(DATA_DIR / "method_agreement.csv", index=False)
    print("\n── Inter-method Agreement (SHAP vs LIME) ──")
    print(agree_df.to_string(index=False))

    # Figure: inter-method agreement
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
    axes[0].set_xticklabels([GENRE_DISPLAY.get(g, g.capitalize()) for g in agree_df["genre"]], rotation=30, ha="right")
    axes[0].set_ylabel("Score")
    axes[0].set_title("A  SHAP vs LIME Agreement per Genre")
    axes[0].legend()

    # Scatter of Pearson vs Spearman
    axes[1].scatter(agree_df["pearson_r"], agree_df["spearman_r"], s=100,
                    c=[GENRE_COLORS[g] for g in agree_df["genre"]], edgecolors="black", linewidth=1)
    for _, row in agree_df.iterrows():
        axes[1].annotate(GENRE_DISPLAY.get(row["genre"], row["genre"].capitalize()), (row["pearson_r"], row["spearman_r"]),
                         fontsize=9, ha="left", va="bottom")
    axes[1].set_xlabel("Pearson r")
    axes[1].set_ylabel("Spearman ρ")
    axes[1].set_title("B  Correlation Agreement Scatter")

    fig.suptitle("Inter-Method Agreement: SHAP vs LIME", fontweight="bold")
    savefig("05_method_agreement", fig)

    # ══════════════════════════════════════════════════════════════════════════
    # METRIC 4: Spurious Correlation Detection
    # ══════════════════════════════════════════════════════════════════════════
    spurious_rows = []
    for genre in [g for g in GENRES if g in shap_results]:
        sv = np.abs(shap_results[genre]["shap_values"])
        spec = shap_results[genre]["spectrogram"]

        # Silent region = bottom 10% energy of spectrogram
        energy_threshold = np.percentile(spec, 10)
        silence_mask = spec <= energy_threshold
        high_energy_mask = spec > np.percentile(spec, 90)

        silence_shap = sv[silence_mask].mean() if silence_mask.any() else 0.0
        high_shap = sv[high_energy_mask].mean() if high_energy_mask.any() else 0.0

        # Edge regions (first/last 5 time bins)
        edge_width = 5
        edge_shap = sv[:, :edge_width].mean() + sv[:, -edge_width:].mean()
        center_shap = sv[:, edge_width:-edge_width].mean() if sv.shape[1] > 2 * edge_width else sv.mean()

        spurious_rows.append({
            "genre": genre,
            "silence_shap": silence_shap,
            "high_energy_shap": high_shap,
            "silence_ratio": silence_shap / (high_shap + 1e-8),
            "edge_shap": edge_shap / 2,
            "center_shap": center_shap,
            "edge_ratio": (edge_shap / 2) / (center_shap + 1e-8),
        })

    spurious_df = pd.DataFrame(spurious_rows)
    spurious_df.to_csv(DATA_DIR / "spurious_correlation.csv", index=False)
    print("\n── Spurious Correlation Detection ──")
    print(spurious_df[["genre", "silence_ratio", "edge_ratio"]].to_string(index=False))

    # Figure: spurious correlation
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    x = np.arange(len(spurious_df))
    w = 0.35

    axes[0].bar(x - w / 2, spurious_df["silence_shap"], w, label="Silence regions", color="#F44336")
    axes[0].bar(x + w / 2, spurious_df["high_energy_shap"], w, label="High-energy regions", color="#4CAF50")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([GENRE_DISPLAY.get(g, g.capitalize()) for g in spurious_df["genre"]], rotation=30, ha="right")
    axes[0].set_ylabel("Mean |SHAP|")
    axes[0].set_title("A  Attribution: Silence vs High-Energy")
    axes[0].legend()

    axes[1].bar(x - w / 2, spurious_df["edge_shap"], w, label="Edge regions", color="#FF9800")
    axes[1].bar(x + w / 2, spurious_df["center_shap"], w, label="Center regions", color="#2196F3")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([GENRE_DISPLAY.get(g, g.capitalize()) for g in spurious_df["genre"]], rotation=30, ha="right")
    axes[1].set_ylabel("Mean |SHAP|")
    axes[1].set_title("B  Attribution: Edge vs Center")
    axes[1].legend()

    fig.suptitle("Spurious Correlation Detection", fontweight="bold")
    savefig("05_spurious_correlation", fig)

    # ── Summary table ─────────────────────────────────────────────────────────
    summary_rows = []
    for genre in [g for g in GENRES if g in shap_results]:
        align = alignment_df[alignment_df.genre == genre]
        agree = agree_df[agree_df.genre == genre] if genre in agree_df["genre"].values else pd.DataFrame()
        spur = spurious_df[spurious_df.genre == genre]

        faith_50 = faith_df[(faith_df.genre == genre) & (faith_df.k_fraction == 0.5)]

        summary_rows.append({
            "genre": genre,
            "alignment_r": align["spearman_r"].values[0] if len(align) else np.nan,
            "faithfulness_50": faith_50["conf_drop"].values[0] if len(faith_50) else np.nan,
            "shap_lime_pearson": agree["pearson_r"].values[0] if len(agree) else np.nan,
            "spurious_ratio": spur["silence_ratio"].values[0] if len(spur) else np.nan,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(DATA_DIR / "xai_summary_table.csv", index=False)
    print("\n── XAI Summary ──")
    print(summary_df.to_string(index=False))

    print("\n✓ Quantitative validation complete.")
    print(f"  Results saved to {DATA_DIR}")


if __name__ == "__main__":
    main()
