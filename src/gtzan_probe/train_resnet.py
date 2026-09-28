"""Train a second CNN architecture (MusicResNet) on the same GTZAN splits.

Uses identical preprocessing, splits (random_state=42), and training
hyperparameters as the original MusicCNN. Run AFTER the original training
to compare SHAP/LIME explanations across architectures.

Run from project root:
    uv run python -m gtzan_probe.train_resnet

Saves: models/resnet_gtzan_best.pt, models/resnet_gtzan_label_encoder.pkl
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import DATA_DIR, MODELS_DIR, GENRES, KNOWN_FAULTS, get_device
from .dataset import GTZANDataset
from .model_resnet import MusicResNet

BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 120
WARMUP_EPOCHS = 10
LABEL_SMOOTHING = 0.1
MIXUP_ALPHA = 0.3
RANDOM_STATE = 42  # same as MusicCNN to get identical splits
TEST_SIZE = 0.2


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.3
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha > 0:
        lam = float(np.random.beta(alpha, alpha))
    else:
        lam = 1.0
    batch_size = x.size(0)
    idx = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


def mixup_criterion(criterion: nn.Module, pred: torch.Tensor,
                    y_a: torch.Tensor, y_b: torch.Tensor,
                    lam: float) -> torch.Tensor:
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def main() -> None:
    device = get_device()
    print(f"Training MusicResNet on {device}")

    # ── Metadata ──────────────────────────────────────────────────────────────
    meta = pd.read_csv(DATA_DIR / "gtzan_metadata.csv")
    meta = meta[~meta["filename"].isin(KNOWN_FAULTS)]
    meta = meta[meta["genre"].isin(GENRES)].reset_index(drop=True)

    le = LabelEncoder()
    le.fit(GENRES)

    # Same stratified split as MusicCNN (random_state=42)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                                      random_state=RANDOM_STATE)
    train_idx, val_idx = next(splitter.split(meta, meta["genre"]))
    train_meta = meta.iloc[train_idx].reset_index(drop=True)
    val_meta = meta.iloc[val_idx].reset_index(drop=True)
    print(f"Train: {len(train_meta)}  Val: {len(val_meta)}")

    train_ds = GTZANDataset(train_meta, le, augment=True, segments_per_clip=5)
    val_ds = GTZANDataset(val_meta, le, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MusicResNet(n_classes=len(GENRES)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MusicResNet parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=EPOCHS - WARMUP_EPOCHS)

    best_val_acc = 0.0
    best_path = MODELS_DIR / "resnet_gtzan_best.pt"

    # ── Training loop ──────────────────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        # Linear warmup
        if epoch <= WARMUP_EPOCHS:
            for pg in optimizer.param_groups:
                pg["lr"] = LR * (epoch / WARMUP_EPOCHS)

        model.train()
        train_loss = 0.0
        for x, y, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}",
                             leave=False):
            x, y = x.to(device), y.to(device)
            x, y_a, y_b, lam = mixup_data(x, y, MIXUP_ALPHA)
            optimizer.zero_grad()
            logits = model(x)
            loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        if epoch > WARMUP_EPOCHS:
            scheduler.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        val_acc = correct / total

        if epoch % 10 == 0 or val_acc > best_val_acc:
            print(f"  Epoch {epoch:3d}: loss={train_loss/len(train_loader):.4f} "
                  f"val_acc={val_acc:.4f}"
                  + (" ← best" if val_acc > best_val_acc else ""))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)

    print(f"\nBest val acc: {best_val_acc:.4f} → {best_path}")

    # Save label encoder (same as MusicCNN, but saving separately for clarity)
    le_path = MODELS_DIR / "resnet_gtzan_label_encoder.pkl"
    with open(le_path, "wb") as f:
        pickle.dump(le, f)

    # Save val split membership for provenance
    val_filenames = val_meta["filename"].tolist()
    (DATA_DIR / "resnet_val_filenames.json").write_text(
        __import__("json").dumps({
            "val_filenames": val_filenames,
            "n_val": len(val_filenames),
            "n_train": len(train_meta),
            "random_state": RANDOM_STATE,
            "note": "Identical split to MusicCNN (same random_state=42).",
        }, indent=2) + "\n"
    )
    print("Done.")


if __name__ == "__main__":
    main()
