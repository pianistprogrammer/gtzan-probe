"""Step 2b — Evaluate the trained MusicCNN and produce confusion matrix / metrics figures.

Usage:
    python -m gtzan_probe.evaluate
"""

import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix

from .config import (
    DATA_DIR, FIGURES_DIR, MODELS_DIR, GENRES, GENRE_COLORS,
    setup_plotting, savefig, get_device,
)
from .model import MusicCNN
from .dataset import GTZANDataset


def main():
    print("LEGACY WORKFLOW: read docs/REPRODUCIBILITY.md; this is not the revised independent-test protocol.")
    setup_plotting()
    device = get_device()
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    model = MusicCNN(n_classes=10).to(device)
    model.load_state_dict(torch.load(MODELS_DIR / "cnn_gtzan_best.pt", map_location=device))
    model.eval()

    with open(MODELS_DIR / "label_encoder.pkl", "rb") as f:
        le: LabelEncoder = pickle.load(f)

    # ── Build test set ────────────────────────────────────────────────────────
    meta = pd.read_csv(DATA_DIR / "gtzan_metadata.csv")
    clean_meta = meta[meta.fault == "clean"].copy()
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for _, test_idx in sss.split(clean_meta, clean_meta["genre"]):
        test_meta = clean_meta.iloc[test_idx]

    test_ds = GTZANDataset(test_meta, le, augment=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    # ── Inference ─────────────────────────────────────────────────────────────
    all_preds, all_labels, all_files = [], [], []
    with torch.no_grad():
        for X, y, fnames in test_loader:
            X = X.to(device)
            logits = model(X)
            preds = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
            all_files.extend(fnames)

    pred_genres = le.inverse_transform(all_preds)
    true_genres = le.inverse_transform(all_labels)

    report = classification_report(true_genres, pred_genres, target_names=GENRES, output_dict=True)
    print(classification_report(true_genres, pred_genres, target_names=GENRES))

    # ── Confusion matrix figure ───────────────────────────────────────────────
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

    # ── Per-class metrics bar chart ───────────────────────────────────────────
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

    # ── Save predictions for XAI steps ────────────────────────────────────────
    pred_df = pd.DataFrame({
        "filename": all_files,
        "true_genre": true_genres,
        "pred_genre": pred_genres,
        "correct": (np.array(true_genres) == np.array(pred_genres)).astype(int),
    })
    pred_df = pred_df.merge(
        pd.read_csv(DATA_DIR / "gtzan_metadata.csv")[["filename", "filepath", "fault"]],
        on="filename", how="left",
    )
    pred_df.to_csv(DATA_DIR / "test_predictions.csv", index=False)

    overall_acc = pred_df["correct"].mean()
    print(f"\nOverall test accuracy: {overall_acc:.4f}")
    print(f"Predictions saved to {DATA_DIR / 'test_predictions.csv'}")
    print("✓ Evaluation complete.")


if __name__ == "__main__":
    main()
