"""Step 6 — Fine-tune Audio Spectrogram Transformer (AST) and visualise attention.

Usage:
    python -m gtzan_probe.explain_attention [--epochs-head 10] [--epochs-full 20]
"""

import argparse
import pickle

import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import ASTFeatureExtractor, ASTForAudioClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report

from .config import (
    DATA_DIR, FIGURES_DIR, MODELS_DIR, GENRES, GENRE_COLORS,
    SR, setup_plotting, savefig, get_device,
)


AST_MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_SR = 16000  # AST was pre-trained at 16 kHz


class GTZANAudioDataset(Dataset):
    """Dataset that returns waveforms preprocessed for AST."""

    def __init__(self, meta_df, label_encoder, feature_extractor, duration=10.0, augment=False):
        self.records = meta_df.reset_index(drop=True)
        self.le = label_encoder
        self.fe = feature_extractor
        self.duration = duration
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        label = self.le.transform([row.genre])[0]

        try:
            y, _ = librosa.load(row.filepath, sr=AST_SR, duration=self.duration)
        except Exception:
            y = np.zeros(int(AST_SR * self.duration))

        # Random offset for augmentation
        target_len = int(self.duration * AST_SR)
        if self.augment and len(y) > target_len:
            start = np.random.randint(0, len(y) - target_len)
            y = y[start : start + target_len]
        else:
            if len(y) < target_len:
                y = np.pad(y, (0, target_len - len(y)))
            else:
                y = y[:target_len]

        inputs = self.fe(y, sampling_rate=AST_SR, return_tensors="pt")
        input_values = inputs["input_values"].squeeze(0)
        return input_values, label, row.filename


def parse_args():
    p = argparse.ArgumentParser(description="AST attention visualization")
    p.add_argument("--epochs-head", type=int, default=10)
    p.add_argument("--epochs-full", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr-full", type=float, default=1e-5)
    p.add_argument("--extract-only", action="store_true",
                   help="Skip training, load saved model, and only extract attention + generate figures")
    return p.parse_args()


def main():
    args = parse_args()
    setup_plotting()
    device = get_device()
    print(f"Device: {device}")

    # ── Load AST ──────────────────────────────────────────────────────────────
    feature_extractor = ASTFeatureExtractor.from_pretrained(AST_MODEL_ID)
    ast_model = ASTForAudioClassification.from_pretrained(
        AST_MODEL_ID,
        num_labels=10,
        ignore_mismatched_sizes=True,
        output_attentions=True,
    )

    hidden_size = ast_model.config.hidden_size
    ast_model.classifier = nn.Sequential(
        nn.LayerNorm(hidden_size),
        nn.Linear(hidden_size, 10),
    )
    ast_model = ast_model.to(device)
    print(f"AST loaded. Hidden size: {hidden_size}")

    # ── Dataloaders ───────────────────────────────────────────────────────────
    le = LabelEncoder()
    le.fit(GENRES)

    meta = pd.read_csv(DATA_DIR / "gtzan_metadata.csv")
    clean_meta = meta[meta.fault == "clean"].copy()

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in sss.split(clean_meta, clean_meta["genre"]):
        train_meta = clean_meta.iloc[train_idx]
        test_meta = clean_meta.iloc[test_idx]

    train_ds = GTZANAudioDataset(train_meta, le, feature_extractor, augment=True)
    test_ds = GTZANAudioDataset(test_meta, le, feature_extractor, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    history = []

    if args.extract_only:
        print("\n── Skipping training (--extract-only), loading saved model ──")
    else:
        # ── Phase 1: freeze backbone, train head ──────────────────────────────
        print("\n── Phase 1: Linear probing (head only) ──")
        for name, p in ast_model.named_parameters():
            if "classifier" not in name:
                p.requires_grad = False

        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, ast_model.parameters()), lr=args.lr_head)

        for epoch in range(1, args.epochs_head + 1):
            ast_model.train()
            t_loss, t_correct, t_total = 0.0, 0, 0
            for X, y, _ in train_loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                outputs = ast_model(X)
                loss = criterion(outputs.logits, y)
                loss.backward()
                optimizer.step()
                t_loss += loss.item() * len(y)
                t_correct += (outputs.logits.argmax(1) == y).sum().item()
                t_total += len(y)

            history.append({"phase": 1, "epoch": epoch, "loss": t_loss / t_total, "acc": t_correct / t_total})
            if epoch % 5 == 0:
                print(f"  Ep {epoch}: loss={t_loss / t_total:.4f} acc={t_correct / t_total:.3f}")

        # ── Phase 2: unfreeze all, fine-tune ──────────────────────────────────
        print("\n── Phase 2: Full fine-tune ──")
        for p in ast_model.parameters():
            p.requires_grad = True

        optimizer = optim.AdamW(ast_model.parameters(), lr=args.lr_full, weight_decay=1e-4)
        best_val_acc = 0.0

        for epoch in range(1, args.epochs_full + 1):
            ast_model.train()
            t_loss, t_correct, t_total = 0.0, 0, 0
            for X, y, _ in train_loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                outputs = ast_model(X)
                loss = criterion(outputs.logits, y)
                loss.backward()
                optimizer.step()
                t_loss += loss.item() * len(y)
                t_correct += (outputs.logits.argmax(1) == y).sum().item()
                t_total += len(y)

            # Validate
            ast_model.eval()
            v_correct, v_total = 0, 0
            with torch.no_grad():
                for X, y, _ in test_loader:
                    X, y = X.to(device), y.to(device)
                    outputs = ast_model(X)
                    v_correct += (outputs.logits.argmax(1) == y).sum().item()
                    v_total += len(y)

            v_acc = v_correct / v_total
            history.append({"phase": 2, "epoch": epoch, "loss": t_loss / t_total, "acc": v_acc})

            if v_acc > best_val_acc:
                best_val_acc = v_acc
                torch.save(ast_model.state_dict(), MODELS_DIR / "ast_gtzan_best.pt")

            if epoch % 5 == 0:
                print(f"  Ep {epoch}: loss={t_loss / t_total:.4f} val_acc={v_acc:.3f}")

        print(f"Best AST val accuracy: {best_val_acc:.4f}")

        # ── Training curves figure ────────────────────────────────────────────
        hist_df = pd.DataFrame(history)
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(range(1, len(hist_df) + 1), hist_df["acc"] * 100, "-o", color="#2196F3", markersize=3)
        ax.axvline(args.epochs_head + 0.5, color="red", linestyle="--", label="Phase 1→2 boundary")
        ax.set_xlabel("Epoch (combined)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("AST Training Curves", fontweight="bold")
        ax.legend()
        savefig("06_ast_training_curves", fig)

    # ── Extract attention maps ────────────────────────────────────────────────
    ast_model.load_state_dict(torch.load(MODELS_DIR / "ast_gtzan_best.pt", map_location=device))
    ast_model.eval()

    preds_df = pd.read_csv(DATA_DIR / "test_predictions.csv")
    attention_results = {}

    for genre in tqdm(GENRES, desc="Extracting attention"):
        correct = preds_df[
            (preds_df.true_genre == genre) &
            (preds_df.pred_genre == genre) &
            (preds_df.fault == "clean")
        ]
        if correct.empty:
            continue

        row = correct.iloc[0]
        try:
            y, _ = librosa.load(row.filepath, sr=AST_SR, duration=10.0)
        except Exception:
            continue

        target_len = int(10.0 * AST_SR)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))

        inputs = feature_extractor(y, sampling_rate=AST_SR, return_tensors="pt")
        input_values = inputs["input_values"].to(device)

        with torch.no_grad():
            outputs = ast_model(input_values, output_attentions=True)
            attentions = outputs.attentions  # tuple of (1, n_heads, seq_len, seq_len)

        # Average attention from CLS token across all heads in last layer
        last_attn = attentions[-1][0]  # (n_heads, seq_len, seq_len)
        cls_attn = last_attn[:, 0, 1:].mean(dim=0).cpu().numpy()  # skip CLS-to-CLS

        attention_results[genre] = {
            "cls_attention": cls_attn,
            "filename": row.filename,
            "all_layer_attns": [a[0, :, 0, 1:].mean(dim=0).cpu().numpy() for a in attentions],
        }

    # ── Figure: CLS attention gallery ─────────────────────────────────────────
    available = [g for g in GENRES if g in attention_results]
    n = len(available)
    cols = 5
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for i, genre in enumerate(available):
        ax = axes_flat[i]
        attn = attention_results[genre]["cls_attention"]
        # Reshape to approximate 2D (freq x time) if possible
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

    # ── Save attention results ────────────────────────────────────────────────
    with open(DATA_DIR / "attention_results.pkl", "wb") as f:
        pickle.dump(attention_results, f)

    print("✓ Attention visualization complete.")


if __name__ == "__main__":
    main()
