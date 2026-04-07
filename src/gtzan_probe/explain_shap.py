"""Step 3 — SHAP explanations for the trained CNN.

Usage:
    python -m gtzan_probe.explain_shap [--n-background 50]
"""

import argparse
import pickle

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from tqdm.auto import tqdm

import torch
import torch.nn as nn

from .config import (
    DATA_DIR, FIGURES_DIR, MODELS_DIR, GENRES, GENRE_COLORS,
    SR, N_MELS, N_FFT, HOP, SEG_S, BAND_RANGES,
    setup_plotting, savefig, get_device,
)
from .model import MusicCNN
from .dataset import load_spectrogram


def parse_args():
    p = argparse.ArgumentParser(description="SHAP explanations")
    p.add_argument("--n-background", type=int, default=50)
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

    # ── Load data ─────────────────────────────────────────────────────────────
    meta = pd.read_csv(DATA_DIR / "gtzan_metadata.csv")
    preds_df = pd.read_csv(DATA_DIR / "test_predictions.csv")

    # Build background dataset (stratified sample)
    bg_spectrograms = []
    for genre in GENRES:
        genre_clips = meta[(meta.genre == genre) & (meta.fault == "clean")].head(
            args.n_background // len(GENRES)
        )
        for _, row in genre_clips.iterrows():
            _, tensor = load_spectrogram(row.filepath)
            bg_spectrograms.append(tensor.squeeze(0))  # (1, 128, W)

    background = torch.stack(bg_spectrograms).to(device)
    print(f"Background set: {background.shape}")

    # ── SHAP DeepExplainer ────────────────────────────────────────────────────
    explainer = shap.DeepExplainer(model, background)

    shap_results = {}
    for genre in tqdm(GENRES, desc="Computing SHAP"):
        correct = preds_df[
            (preds_df.true_genre == genre) &
            (preds_df.pred_genre == genre) &
            (preds_df.fault == "clean")
        ]
        if correct.empty:
            print(f"  No correct predictions for {genre}, skipping")
            continue

        row = correct.iloc[0]
        spec_np, spec_tensor = load_spectrogram(row.filepath)
        spec_tensor = spec_tensor.to(device)

        shap_values = explainer.shap_values(spec_tensor)
        pred_idx = le.transform([genre])[0]

        # shap_values shape: (1, 1, 128, 128, n_classes) — classes on last axis
        sv = shap_values[0, 0, :, :, pred_idx]  # (128, 128)

        shap_results[genre] = {
            "shap_values": sv,
            "spectrogram": spec_np,
            "filename": row.filename,
            "filepath": row.filepath,
        }

    with open(DATA_DIR / "shap_values.pkl", "wb") as f:
        pickle.dump(shap_results, f)
    print(f"SHAP values saved to {DATA_DIR / 'shap_values.pkl'}")

    # ── Figure: SHAP overlay gallery ──────────────────────────────────────────
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
    axes_flat = [ax for row_axes in axes for ax in (row_axes if hasattr(row_axes, "__len__") else [row_axes])]

    for i, genre in enumerate(available):
        ax = axes_flat[i]
        res = shap_results[genre]
        sv = res["shap_values"]
        vmax = np.abs(sv).max()

        ax.imshow(res["spectrogram"], aspect="auto", origin="lower", cmap="gray_r", alpha=0.4)
        im = ax.imshow(sv, aspect="auto", origin="lower", cmap=shap_cmap,
                       alpha=0.7, vmin=-vmax, vmax=vmax)
        ax.set_title(f"{genre.capitalize()}", fontsize=14, fontweight="bold", color=GENRE_COLORS[genre])
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Mel bin", fontsize=12)
        ax.tick_params(labelsize=10)

    # Hide unused axes
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.subplots_adjust(wspace=0.35, hspace=0.4)
    fig.suptitle("SHAP Explanation Overlays per Genre", fontsize=16, fontweight="bold")
    savefig("03_shap_gallery", fig)

    # ── Figure: Mean |SHAP| per frequency band per genre ──────────────────────
    band_names = list(BAND_RANGES.keys())
    band_data = {genre: [] for genre in available}

    for genre in available:
        sv = np.abs(shap_results[genre]["shap_values"])
        for band_name in band_names:
            lo, hi = BAND_RANGES[band_name]
            band_data[genre].append(sv[lo:hi, :].mean())

    band_df = pd.DataFrame(band_data, index=band_names).T

    fig, ax = plt.subplots(figsize=(14, 7))
    band_df.plot(kind="bar", ax=ax, colormap="viridis", edgecolor="white", linewidth=0.5)
    ax.set_xticklabels([g.capitalize() for g in available], rotation=30, ha="right")
    ax.set_ylabel("Mean |SHAP| Attribution")
    ax.set_title("Mean Absolute SHAP per Frequency Band per Genre", fontweight="bold")
    ax.legend(title="Frequency Band")
    savefig("03_shap_frequency_bands", fig)

    print("✓ SHAP explanations complete.")


if __name__ == "__main__":
    main()
