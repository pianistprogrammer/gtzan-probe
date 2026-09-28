"""Step 2 — Train the MusicCNN on GTZAN mel spectrograms.

Usage:
    python -m gtzan_probe.train [--epochs 120] [--batch-size 32] [--lr 1e-3]
"""

import argparse
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix

from .config import (
    DATA_DIR, FIGURES_DIR, MODELS_DIR, GENRES, GENRE_COLORS,
    setup_plotting, savefig, get_device,
)
from .model import MusicCNN
from .dataset import GTZANDataset, GTZANMultiCropDataset


# ── Mixup helper ──────────────────────────────────────────────────────────────
def mixup_data(x, y, alpha=0.3):
    """Returns mixed inputs, pairs of targets, and lambda."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def parse_args():
    p = argparse.ArgumentParser(description="Train MusicCNN on GTZAN")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--segments-per-clip", type=int, default=5,
                   help="Random 3s segments per 30s clip per epoch (training)")
    p.add_argument("--mixup-alpha", type=float, default=0.3,
                   help="Mixup interpolation strength (0 = disabled)")
    p.add_argument("--warmup-epochs", type=int, default=10)
    return p.parse_args()


def main():
    print("LEGACY WORKFLOW: read docs/REPRODUCIBILITY.md; this is not the revised independent-test protocol.")
    args = parse_args()
    setup_plotting()
    device = get_device()
    print(f"Device: {device}")

    # ── Load metadata & build label encoder ───────────────────────────────────
    meta = pd.read_csv(DATA_DIR / "gtzan_metadata.csv")
    clean_meta = meta[meta.fault == "clean"].copy()

    le = LabelEncoder()
    le.fit(GENRES)
    with open(MODELS_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    # ── Stratified split ──────────────────────────────────────────────────────
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in sss.split(clean_meta, clean_meta["genre"]):
        train_meta = clean_meta.iloc[train_idx]
        test_meta = clean_meta.iloc[test_idx]

    # Multi-segment training: 5 random crops per clip per epoch
    train_ds = GTZANDataset(train_meta, le, augment=True,
                            segments_per_clip=args.segments_per_clip)
    # Single-crop for per-batch validation (fast)
    test_ds = GTZANDataset(test_meta, le, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Train samples/epoch: {len(train_ds)} ({args.segments_per_clip} segs/clip) | "
          f"Val batches: {len(test_loader)}")

    # Multi-crop test set for TTA evaluation at the end
    print("Building multi-crop test set for TTA...")
    tta_ds = GTZANMultiCropDataset(test_meta, le)
    tta_loader = DataLoader(tta_ds, batch_size=64, shuffle=False, num_workers=0)
    print(f"TTA segments: {len(tta_ds)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MusicCNN(n_classes=10).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"MusicCNN: {total_params:,} parameters")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Linear warmup then cosine annealing
    warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=args.warmup_epochs)
    cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-6)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[args.warmup_epochs])

    # ── Training loop ─────────────────────────────────────────────────────────
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # Train with mixup
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for X, y, _ in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            if args.mixup_alpha > 0:
                X_mix, y_a, y_b, lam = mixup_data(X, y, args.mixup_alpha)
                logits = model(X_mix)
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
                # Accuracy tracked on original labels (approximate)
                t_correct += (logits.argmax(1) == y_a).sum().item() * lam
                t_correct += (logits.argmax(1) == y_b).sum().item() * (1 - lam)
            else:
                logits = model(X)
                loss = criterion(logits, y)
                t_correct += (logits.argmax(1) == y).sum().item()

            loss.backward()
            optimizer.step()
            t_loss += loss.item() * len(y)
            t_total += len(y)

        # Validate (single crop, fast)
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for X, y, _ in test_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                loss = criterion(logits, y)
                v_loss += loss.item() * len(y)
                v_correct += (logits.argmax(1) == y).sum().item()
                v_total += len(y)

        scheduler.step()

        t_acc = t_correct / t_total
        v_acc = v_correct / v_total
        history["train_loss"].append(t_loss / t_total)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss / v_total)
        history["val_acc"].append(v_acc)

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), MODELS_DIR / "cnn_gtzan_best.pt")

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Ep {epoch:03d} | "
                f"train_loss={t_loss / t_total:.4f} acc={t_acc:.3f} | "
                f"val_loss={v_loss / v_total:.4f} acc={v_acc:.3f}"
            )

    torch.save(model.state_dict(), MODELS_DIR / "cnn_gtzan.pt")
    print(f"\nBest single-crop val accuracy: {best_val_acc:.4f}")

    # ── Test-Time Aggregation (TTA) — average logits over all segments per clip ──
    model.load_state_dict(torch.load(MODELS_DIR / "cnn_gtzan_best.pt", weights_only=True))
    model.eval()
    clip_logits = {}   # clip_idx → list of logit tensors
    clip_labels = {}
    with torch.no_grad():
        for X, y, fnames, clip_idxs in tta_loader:
            X = X.to(device)
            logits = model(X).cpu()
            for i in range(len(X)):
                ci = clip_idxs[i].item()
                clip_logits.setdefault(ci, []).append(logits[i])
                clip_labels[ci] = y[i].item()

    tta_correct = 0
    for ci, logit_list in clip_logits.items():
        avg_logit = torch.stack(logit_list).mean(0)
        pred = avg_logit.argmax().item()
        if pred == clip_labels[ci]:
            tta_correct += 1
    tta_acc = tta_correct / len(clip_logits)
    print(f"TTA clip-level val accuracy: {tta_acc:.4f} ({len(clip_logits)} clips)")

    # ── Training curves figure ────────────────────────────────────────────────
    epochs_range = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(epochs_range, history["train_loss"], label="Train loss", color="#2196F3", linewidth=2)
    ax1.plot(epochs_range, history["val_loss"], label="Val loss", color="#F44336", linewidth=2, linestyle="--")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Cross-Entropy Loss"); ax1.set_title("A  Loss Curves"); ax1.legend()

    ax2.plot(epochs_range, [a * 100 for a in history["train_acc"]], label="Train acc", color="#2196F3", linewidth=2)
    ax2.plot(epochs_range, [a * 100 for a in history["val_acc"]], label="Val acc", color="#F44336", linewidth=2, linestyle="--")
    ax2.axhline(best_val_acc * 100, color="gray", linestyle=":", linewidth=1.2, label=f"Best val {best_val_acc * 100:.1f}%")
    ax2.axhline(tta_acc * 100, color="#4CAF50", linestyle="-.", linewidth=1.2, label=f"TTA {tta_acc * 100:.1f}%")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)"); ax2.set_title("B  Accuracy Curves"); ax2.legend()

    fig.suptitle("CNN Training History on GTZAN", fontweight="bold")
    savefig("02_training_curves", fig)

    print("\n✓ Training complete. Model saved to", MODELS_DIR)


if __name__ == "__main__":
    main()
