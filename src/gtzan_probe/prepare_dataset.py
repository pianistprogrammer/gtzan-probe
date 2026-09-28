"""Step 1 — Download GTZAN, audit faults, extract features, produce EDA figures.

Usage:
    python -m gtzan_probe.prepare_dataset
"""

import os
import subprocess
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedShuffleSplit

from .config import (
    DATA_DIR, FIGURES_DIR, GTZAN_ROOT, GENRES, GENRE_COLORS,
    SR, N_MELS, HOP, KNOWN_FAULTS, setup_plotting, savefig,
)


def download_gtzan():
    """Download GTZAN via Kaggle API if not already present."""
    if GTZAN_ROOT.exists() and any(GTZAN_ROOT.iterdir()):
        print(f"GTZAN already exists at {GTZAN_ROOT}")
        return

    print("GTZAN not found. Downloading via Kaggle API...")
    gtzan_parent = GTZAN_ROOT.parent
    gtzan_parent.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", "andradaolteanu/gtzan-dataset-music-genre-classification",
            "-p", str(gtzan_parent), "--unzip",
        ],
        check=False,
    )
    if result.returncode != 0:
        # Try alternate path structure after unzip
        alt = DATA_DIR / "gtzan" / "Data" / "genres_original"
        if alt.exists():
            print(f"Found at alternate path: {alt}")
            return
        raise RuntimeError(
            "Kaggle download failed. Place kaggle.json in ~/.kaggle/ "
            "or set GTZAN_ROOT env var to your local GTZAN path."
        )
    print(f"Downloaded to {gtzan_parent}")


def build_metadata() -> pd.DataFrame:
    """Scan GTZAN clips and build metadata CSV with fault flags."""
    # Support alternate unzip structure
    root = GTZAN_ROOT
    if not root.exists():
        alt = DATA_DIR / "gtzan" / "Data" / "genres_original"
        if alt.exists():
            root = alt
        else:
            raise FileNotFoundError(f"GTZAN not found at {GTZAN_ROOT} or {alt}")

    records = []
    for genre in tqdm(GENRES, desc="Scanning clips"):
        genre_dir = root / genre
        if not genre_dir.exists():
            print(f"  Warning: {genre_dir} not found")
            continue
        for wav in sorted(genre_dir.glob("*.wav")):
            fault = KNOWN_FAULTS.get(wav.name, "clean")
            rec = {
                "filepath": str(wav),
                "filename": wav.name,
                "genre": genre,
                "fault": fault,
                "file_size_kb": wav.stat().st_size / 1024,
            }
            try:
                rec["duration_s"] = round(librosa.get_duration(path=str(wav)), 3)
            except Exception:
                rec["duration_s"] = None
                rec["fault"] = "corrupt"
            records.append(rec)

    meta = pd.DataFrame(records)
    meta.to_csv(DATA_DIR / "gtzan_metadata.csv", index=False)
    print(f"\nTotal clips: {len(meta)}")
    print(f"Fault breakdown:\n{meta['fault'].value_counts()}")
    return meta


def extract_features(filepath):
    """Extract acoustic features from a single audio file."""
    try:
        y, sr = librosa.load(filepath, sr=SR, duration=30.0)
    except Exception:
        return None

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    feats = {}
    for i, coeff in enumerate(mfcc):
        feats[f"mfcc_{i}_mean"] = coeff.mean()
        feats[f"mfcc_{i}_std"] = coeff.std()
    feats["chroma_mean"] = chroma.mean()
    feats["chroma_std"] = chroma.std()
    feats["contrast_mean"] = contrast.mean()
    feats["zcr_mean"] = zcr.mean()
    feats["zcr_std"] = zcr.std()
    feats["rms_mean"] = rms.mean()
    feats["rms_std"] = rms.std()
    feats["centroid_mean"] = centroid.mean()
    feats["bandwidth_mean"] = bandwidth.mean()
    feats["rolloff_mean"] = rolloff.mean()
    feats["tempo"] = float(np.asarray(tempo).item())
    return feats


def build_feature_matrix(meta: pd.DataFrame) -> pd.DataFrame:
    """Extract features for all non-corrupt clips."""
    all_feats = []
    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Extracting features"):
        if row.fault == "corrupt":
            continue
        f = extract_features(row.filepath)
        if f:
            f["genre"] = row.genre
            f["filename"] = row.filename
            f["fault"] = row.fault
            all_feats.append(f)

    feat_df = pd.DataFrame(all_feats)
    feat_df.to_csv(DATA_DIR / "gtzan_features.csv", index=False)
    print(f"Feature matrix: {feat_df.shape}")
    return feat_df


def make_splits(feat_df: pd.DataFrame):
    """Create fault-aware stratified train/test split."""
    clean_df = feat_df[feat_df.fault == "clean"].copy().reset_index(drop=True)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in sss.split(clean_df, clean_df["genre"]):
        train_df = clean_df.iloc[train_idx]
        test_df = clean_df.iloc[test_idx]

    train_df.to_csv(DATA_DIR / "train_features.csv", index=False)
    test_df.to_csv(DATA_DIR / "test_features.csv", index=False)
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")


# ── EDA Figures ───────────────────────────────────────────────────────────────

def fig_dataset_composition(meta):
    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # Panel A: clips per genre
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

    # Panel B: duration distribution
    ax2 = fig.add_subplot(gs[1])
    valid = meta.dropna(subset=["duration_s"])
    ax2.hist(valid["duration_s"], bins=30, color="#2196F3", edgecolor="white", linewidth=0.5)
    ax2.axvline(30.0, color="#F44336", linestyle="--", linewidth=1.5, label="Expected 30s")
    ax2.set_xlabel("Duration (seconds)")
    ax2.set_ylabel("Count")
    ax2.set_title("B  Clip Duration Distribution")
    ax2.legend()

    # Panel C: fault breakdown by genre
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


def fig_spectrogram_gallery(meta):
    fig, axes = plt.subplots(2, 5, figsize=(20, 7))
    axes_flat = axes.flatten()
    img = None
    for i, genre in enumerate(GENRES):
        clips = meta[(meta.genre == genre) & (meta.fault == "clean")].head(1)
        if clips.empty:
            continue
        path = clips.iloc[0]["filepath"]
        y, sr = librosa.load(path, sr=SR, duration=30.0)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP)
        S_db = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(
            S_db, sr=sr, hop_length=HOP, x_axis="time", y_axis="mel",
            ax=axes_flat[i], cmap="magma",
        )
        axes_flat[i].set_title(genre.capitalize(), fontweight="bold", color=GENRE_COLORS[genre])
        axes_flat[i].set_xlabel("")
        axes_flat[i].set_ylabel("")

    if img is not None:
        fig.colorbar(img, ax=axes_flat, format="%+2.0f dB", shrink=0.6, pad=0.02, label="Power (dB)")
    fig.suptitle("Mel Spectrogram Gallery (one clip per genre)", fontweight="bold")
    savefig("01_spectrogram_gallery", fig)


def fig_feature_distributions(feat_df):
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


def fig_correlation_heatmap(feat_df):
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


def fig_tempo_boxplot(feat_df):
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


def fig_mfcc_heatmap(feat_df):
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


def main():
    print("LEGACY WORKFLOW: read docs/REPRODUCIBILITY.md; this is not the revised independent-test protocol.")
    setup_plotting()

    # Step 1: Download
    download_gtzan()

    # Step 2: Build metadata
    meta = build_metadata()

    # Step 3: Extract features
    feat_df = build_feature_matrix(meta)

    # Step 4: Train/test split
    make_splits(feat_df)

    # Step 5: EDA figures
    print("\nGenerating EDA figures...")
    fig_dataset_composition(meta)
    fig_spectrogram_gallery(meta)
    fig_feature_distributions(feat_df)
    fig_correlation_heatmap(feat_df)
    fig_tempo_boxplot(feat_df)
    fig_mfcc_heatmap(feat_df)

    print("\n✓ Dataset preparation complete.")
    print(f"  Data saved to:    {DATA_DIR}")
    print(f"  Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
