"""Step 4 — LIME explanations for the trained CNN.

Usage:
    python -m gtzan_probe.explain_lime [--n-samples 500] [--n-segments 40]
"""

import argparse
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path
from tqdm.auto import tqdm
from skimage.segmentation import slic, mark_boundaries
from lime import lime_image

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    DATA_DIR, FIGURES_DIR, MODELS_DIR, GENRES, GENRE_COLORS,
    SR, N_MELS, N_FFT, HOP, BAND_RANGES, FIXED_W,
    setup_plotting, savefig, get_device,
)
from .model import MusicCNN
from .dataset import load_spectrogram


def parse_args():
    p = argparse.ArgumentParser(description="LIME explanations")
    p.add_argument("--n-samples", type=int, default=500)
    p.add_argument("--n-segments", type=int, default=40)
    return p.parse_args()


def main():
    args = parse_args()
    setup_plotting()
    device = get_device()
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    model = MusicCNN().to(device)
    model.load_state_dict(torch.load(MODELS_DIR / "cnn_gtzan_best.pt", map_location=device))
    model.eval()

    with open(MODELS_DIR / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    # ── LIME prediction function ──────────────────────────────────────────────
    def lime_predict_fn(images_np):
        """
        images_np: (N, H, W, 3) float array from LIME (uses first channel only).
        Returns softmax probabilities (N, 10).
        """
        batch = torch.tensor(images_np[:, :, :, 0], dtype=torch.float32).unsqueeze(1).to(device)
        with torch.no_grad():
            logits = model(batch)
            probs = F.softmax(logits, dim=1)
        return probs.cpu().numpy()

    # ── Compute LIME explanations ─────────────────────────────────────────────
    preds_df = pd.read_csv(DATA_DIR / "test_predictions.csv")
    lime_results = {}

    explainer = lime_image.LimeImageExplainer()

    for genre in tqdm(GENRES, desc="Computing LIME"):
        correct = preds_df[
            (preds_df.true_genre == genre) &
            (preds_df.pred_genre == genre) &
            (preds_df.fault == "clean")
        ]
        if correct.empty:
            print(f"  No correct predictions for {genre}, skipping")
            continue

        row = correct.iloc[0]
        spec_np, _ = load_spectrogram(row.filepath)

        # LIME expects (H, W, 3) RGB-like input
        spec_rgb = np.stack([spec_np] * 3, axis=-1)

        explanation = explainer.explain_instance(
            spec_rgb,
            lime_predict_fn,
            top_labels=10,
            hide_color=0,
            num_samples=args.n_samples,
            segmentation_fn=lambda img: slic(
                img, n_segments=args.n_segments, compactness=10, sigma=1
            ),
        )

        pred_idx = le.transform([genre])[0]
        temp, mask = explanation.get_image_and_mask(
            pred_idx, positive_only=False, num_features=10, hide_rest=False
        )

        # Get per-segment importance
        local_exp = dict(explanation.local_exp.get(pred_idx, []))

        lime_results[genre] = {
            "image": temp,
            "mask": mask,
            "spectrogram": spec_np,
            "segments": explanation.segments,
            "local_exp": local_exp,
            "filename": row.filename,
            "filepath": row.filepath,
        }

    with open(DATA_DIR / "lime_results.pkl", "wb") as f:
        pickle.dump(lime_results, f)
    print(f"LIME results saved to {DATA_DIR / 'lime_results.pkl'}")

    # ── Figure: LIME superpixel gallery ───────────────────────────────────────
    available = [g for g in GENRES if g in lime_results]
    n = len(available)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    if rows == 1:
        axes = [axes]
    axes_flat = [ax for row_axes in axes for ax in (row_axes if hasattr(row_axes, "__len__") else [row_axes])]

    lime_cmap = LinearSegmentedColormap.from_list("lime_div", ["#2196F3", "#FFFFFF", "#4CAF50"], N=256)

    for i, genre in enumerate(available):
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

    # ── Figure: LIME frequency band importance ────────────────────────────────
    band_names = list(BAND_RANGES.keys())
    band_data = {genre: [] for genre in available}

    for genre in available:
        res = lime_results[genre]
        segments = res["segments"]
        local_exp = res["local_exp"]

        # Build per-pixel importance from segments
        importance_map = np.zeros_like(res["spectrogram"])
        for seg_id, weight in local_exp.items():
            importance_map[segments == seg_id] = weight

        abs_imp = np.abs(importance_map)
        for band_name in band_names:
            lo, hi = BAND_RANGES[band_name]
            band_data[genre].append(abs_imp[lo:hi, :].mean())

    band_df = pd.DataFrame(band_data, index=band_names).T

    fig, ax = plt.subplots(figsize=(14, 7))
    band_df.plot(kind="bar", ax=ax, colormap="viridis", edgecolor="white", linewidth=0.5)
    ax.set_xticklabels([g.capitalize() for g in available], rotation=30, ha="right")
    ax.set_ylabel("Mean |LIME| Attribution")
    ax.set_title("LIME Frequency Band Importance per Genre", fontweight="bold")
    ax.legend(title="Frequency Band")
    savefig("04_lime_frequency_bands", fig)

    print("✓ LIME explanations complete.")


if __name__ == "__main__":
    main()
