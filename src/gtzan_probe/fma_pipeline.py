"""FMA-small pipeline — train classifier, run SHAP/LIME, compare with GTZAN.

Addresses editor request E5/E6: apply the same explanation methodology to
FMA-small and compare band-attribution profiles with GTZAN results.

Usage:
    uv run python -m gtzan_probe.fma_pipeline --output results/fma_analysis

Requires:
  - FMA-small audio at data/fma_archive/fma_small/fma_small/
  - FMA metadata at data/fma_archive/fma_metadata/fma_metadata/tracks.csv

Label crosswalk: FMA-small top-level genres mapped to GTZAN-comparable labels.
Only genres with a clear 1-to-1 mapping are included; ambiguous labels are
excluded. Genre names are NOT assumed to be equivalent across datasets —
the mapping is an operational convenience for comparison, not proof of
equivalent annotation concepts.

Mapped genres (FMA top-level → GTZAN label):
  Hip-Hop  → hiphop
  Pop      → pop
  Rock     → rock
  (Folk, Experimental, International, Electronic, Instrumental: excluded)
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from skimage.color import gray2rgb
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .config import DATA_DIR, MODELS_DIR, get_device, setup_plotting, N_MELS, N_FFT, HOP, SR, FIXED_W, SEG_S
from .model import MusicCNN

# ── Explicit label crosswalk ──────────────────────────────────────────────────
# Only these three genres have a clear 1-to-1 correspondence.
# FMA genre names and GTZAN labels are annotation decisions, not universal
# categories; this mapping is for exploratory comparison only.
FMA_TO_GTZAN = {
    "Hip-Hop": "hiphop",
    "Pop":     "pop",
    "Rock":    "rock",
}
MAPPED_GENRES = sorted(FMA_TO_GTZAN.values())  # ['hiphop', 'pop', 'rock']
AUDIT_BANDS = {"B1": (0, 10), "B2": (10, 30), "B3": (30, 80), "B4": (80, 128)}
N_SHAP_BG = 30  # fewer backgrounds due to smaller dataset
N_LIME_PERTURB = 300
FMA_DURATION = 30.0  # FMA-small clips are 30 seconds


def load_fma_metadata(fma_root: Path) -> pd.DataFrame:
    """Load FMA-small track list with top-level genre labels."""
    tracks_path = fma_root / "fma_metadata" / "fma_metadata" / "tracks.csv"
    # FMA tracks.csv has a multi-level header (2 rows).
    tracks = pd.read_csv(tracks_path, index_col=0, header=[0, 1])
    # Filter to fma_small subset
    small = tracks[tracks[("set", "subset")] == "small"].copy()
    # Extract top-level genre and official split
    small_df = pd.DataFrame({
        "track_id": small.index,
        "genre_top": small[("track", "genre_top")],
        "fma_split": small[("set", "split")],
    }).dropna(subset=["genre_top"]).reset_index(drop=True)
    return small_df


def fma_audio_path(fma_audio_root: Path, track_id: int) -> Path:
    """Return path to FMA audio file from numeric track ID."""
    tid_str = str(track_id).zfill(6)
    folder = tid_str[:3]
    return fma_audio_root / folder / f"{tid_str}.mp3"


def load_fma_spectrogram(audio_path: Path) -> tuple[np.ndarray, torch.Tensor] | None:
    """Load FMA audio and return (spec_np, spec_tensor) matching GTZAN preprocessing.
    Returns None if the file cannot be decoded."""
    try:
        y, _ = librosa.load(str(audio_path), sr=SR, duration=FMA_DURATION)
    except Exception:
        return None
    # Use opening 3 seconds (same crop as GTZAN eval)
    seg_len = int(SEG_S * SR)
    if len(y) >= seg_len:
        y = y[:seg_len]
    else:
        y = np.pad(y, (0, seg_len - len(y)))
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP)
    S_db = librosa.power_to_db(S, ref=np.max)
    S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
    if S_db.shape[1] != FIXED_W:
        from skimage.transform import resize
        S_db = resize(S_db, (N_MELS, FIXED_W), anti_aliasing=True)
    t = torch.tensor(S_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return S_db, t


class FMADataset(Dataset):
    """Simple FMA-small dataset for 3-class training."""

    def __init__(self, df: pd.DataFrame, le: LabelEncoder,
                 audio_root: Path, augment: bool = False) -> None:
        self.df = df.reset_index(drop=True)
        self.le = le
        self.audio_root = audio_root
        self.augment = augment
        self.seg_len = int(SEG_S * SR)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        row = self.df.iloc[idx]
        track_id = int(row["track_id"])
        label = self.le.transform([row["genre_gtzan"]])[0]
        path = fma_audio_path(self.audio_root, track_id)
        try:
            y, _ = librosa.load(str(path), sr=SR, duration=FMA_DURATION)
        except Exception:
            y = np.zeros(self.seg_len)
        if self.augment and len(y) > self.seg_len:
            start = np.random.randint(0, len(y) - self.seg_len)
            y = y[start: start + self.seg_len]
        elif len(y) >= self.seg_len:
            y = y[:self.seg_len]
        else:
            y = np.pad(y, (0, self.seg_len - len(y)))
        S = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP)
        S_db = librosa.power_to_db(S, ref=np.max)
        S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
        if S_db.shape[1] != FIXED_W:
            from skimage.transform import resize
            S_db = resize(S_db, (N_MELS, FIXED_W), anti_aliasing=True)
        tensor = torch.tensor(S_db, dtype=torch.float32).unsqueeze(0)
        return tensor, label, track_id


def train_fma_model(train_df: pd.DataFrame, val_df: pd.DataFrame,
                    le: LabelEncoder, audio_root: Path,
                    device: str, save_path: Path) -> MusicCNN:
    """Train a 3-class MusicCNN on FMA-small mapped genres."""
    n_classes = len(MAPPED_GENRES)
    model = MusicCNN(n_classes=n_classes).to(device)

    train_ds = FMADataset(train_df, le, audio_root, augment=True)
    val_ds = FMADataset(val_df, le, audio_root, augment=False)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=90)

    best_acc = 0.0
    EPOCHS = 100
    for epoch in range(1, EPOCHS + 1):
        if epoch <= 10:
            for pg in optimizer.param_groups:
                pg["lr"] = 1e-3 * (epoch / 10)
        model.train()
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if epoch > 10:
            scheduler.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total += y.size(0)
        acc = correct / total
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path)
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}/{EPOCHS}: val_acc={acc:.4f} (best={best_acc:.4f})")

    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    print(f"FMA model best val acc: {best_acc:.4f}")
    return model


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--skip-training", action="store_true",
                   help="Load existing model from --model-path instead of training")
    p.add_argument("--model-path", type=Path,
                   default=MODELS_DIR / "fma_gtzan_best.pt")
    args = p.parse_args()

    if args.output.exists():
        raise FileExistsError(f"{args.output} already exists.")
    args.output.mkdir(parents=True)
    out = args.output

    device = get_device()
    print(f"Device: {device}")

    # ── Locate FMA data ───────────────────────────────────────────────────────
    fma_root = DATA_DIR / "fma_archive"
    fma_audio_root = fma_root / "fma_small" / "fma_small"
    if not fma_audio_root.exists():
        raise FileNotFoundError(
            f"FMA-small audio not found at {fma_audio_root}. "
            "Set GTZAN_ROOT or update DATA_DIR in config.py."
        )

    # ── Load and filter metadata ──────────────────────────────────────────────
    print("Loading FMA metadata...")
    meta = load_fma_metadata(fma_root)
    mapped = meta[meta["genre_top"].isin(FMA_TO_GTZAN)].copy()
    mapped["genre_gtzan"] = mapped["genre_top"].map(FMA_TO_GTZAN)
    print(f"Mapped tracks: {len(mapped)}")
    print(mapped["genre_gtzan"].value_counts().to_string())

    # ── Document crosswalk ────────────────────────────────────────────────────
    crosswalk = {
        "mapping": FMA_TO_GTZAN,
        "excluded_fma_genres": [
            g for g in meta["genre_top"].unique()
            if g not in FMA_TO_GTZAN and isinstance(g, str)
        ],
        "n_mapped": int(len(mapped)),
        "note": (
            "FMA genre labels are not equivalent to GTZAN labels. "
            "This mapping is an operational choice for exploratory comparison. "
            "Genre names alone do not justify treating annotations as identical."
        ),
    }
    (out / "label_crosswalk.json").write_text(
        json.dumps(crosswalk, indent=2) + "\n"
    )

    # ── Build label encoder (3 classes) ──────────────────────────────────────
    le = LabelEncoder()
    le.fit(MAPPED_GENRES)
    le_path = MODELS_DIR / "fma_gtzan_label_encoder.pkl"
    with open(le_path, "wb") as f:
        pickle.dump(le, f)

    # ── Train/val split (stratified, no artist-disjoint split — disclosed) ───
    # FMA-small provides an official split, but for fair comparison with
    # GTZAN's random 80/20 we replicate that protocol. An artist-aware split
    # is a pending improvement noted in the experiment plan.
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(mapped, mapped["genre_gtzan"]))
    train_df = mapped.iloc[train_idx].reset_index(drop=True)
    val_df = mapped.iloc[val_idx].reset_index(drop=True)
    print(f"Train: {len(train_df)}  Val: {len(val_df)}")

    split_doc = {
        "n_train": len(train_df),
        "n_val": len(val_df),
        "protocol": "random stratified 80/20 (random_state=42), not artist-disjoint",
        "limitation": "Artist replication not controlled; same limitation as GTZAN split.",
    }
    (out / "split_info.json").write_text(json.dumps(split_doc, indent=2) + "\n")

    # ── Train or load model ───────────────────────────────────────────────────
    model_path = args.model_path
    if not args.skip_training:
        print("Training FMA 3-class model...")
        model = train_fma_model(train_df, val_df, le, fma_audio_root, device, model_path)
    else:
        print(f"Loading model from {model_path}")
        model = MusicCNN(n_classes=len(MAPPED_GENRES)).to(device).eval()
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # ── Evaluate and get correct predictions ─────────────────────────────────
    print("Evaluating on validation set...")
    correct_rows = []
    for _, row in tqdm(val_df.iterrows(), total=len(val_df)):
        path = fma_audio_path(fma_audio_root, int(row["track_id"]))
        if not path.exists():
            continue
        result = load_fma_spectrogram(path)
        if result is None:
            continue
        spec_np, spec_t = result
        spec_t = spec_t.to(device)
        with torch.no_grad():
            pred_idx = model(spec_t).argmax(1).item()
        true_idx = int(le.transform([row["genre_gtzan"]])[0])
        correct_rows.append({
            "track_id": int(row["track_id"]),
            "genre_gtzan": row["genre_gtzan"],
            "pred_genre": le.inverse_transform([pred_idx])[0],
            "correct": int(pred_idx == true_idx),
        })

    eval_df = pd.DataFrame(correct_rows)
    eval_df.to_csv(out / "fma_predictions.csv", index=False)
    val_acc = eval_df["correct"].mean()
    print(f"Val accuracy: {val_acc:.4f} ({eval_df['correct'].sum()}/{len(eval_df)})")

    correct_df = eval_df[eval_df["correct"] == 1].copy()
    print(f"Correctly classified: {len(correct_df)}")

    # ── SHAP backgrounds from training data ───────────────────────────────────
    bg_tensors = []
    for genre in MAPPED_GENRES:
        bg_rows = train_df[train_df["genre_gtzan"] == genre].head(10)
        for _, row in bg_rows.iterrows():
            path = fma_audio_path(fma_audio_root, int(row["track_id"]))
            if not path.exists():
                continue
            result = load_fma_spectrogram(path)
            if result is None:
                continue
            _, t = result
            bg_tensors.append(t)          # keep [1,1,128,128]; cat → [N,1,128,128]
            if len(bg_tensors) >= N_SHAP_BG:
                break
        if len(bg_tensors) >= N_SHAP_BG:
            break
    bg = torch.cat(bg_tensors[:N_SHAP_BG], dim=0).to(device)
    explainer = shap.DeepExplainer(model, bg)

    # ── Per-recording SHAP ────────────────────────────────────────────────────
    print(f"Computing SHAP for {len(correct_df)} FMA clips...")
    records = []
    for _, row in tqdm(correct_df.iterrows(), total=len(correct_df)):
        path = fma_audio_path(fma_audio_root, int(row["track_id"]))
        if not path.exists():
            continue
        result = load_fma_spectrogram(path)
        if result is None:
            continue
        spec_np, spec_t = result
        spec_t = spec_t.to(device)
        true_idx = int(le.transform([row["genre_gtzan"]])[0])
        # SHAP 0.51.0: returns (batch, C, H, W, n_classes); do NOT use no_grad
        shap_vals = explainer.shap_values(spec_t)
        shap_map = shap_vals[0, 0, :, :, true_idx]  # (128, 128) signed
        rec = {
            "track_id": int(row["track_id"]),
            "genre_gtzan": row["genre_gtzan"],
        }
        for b, (lo, hi) in AUDIT_BANDS.items():
            rec[f"shap_abs_{b}"] = float(np.mean(np.abs(shap_map[lo:hi, :])))
            rec[f"shap_signed_{b}"] = float(np.mean(shap_map[lo:hi, :]))
            rec[f"input_{b}"] = float(np.mean(spec_np[lo:hi, :]))
        records.append(rec)

    per_rec = pd.DataFrame(records)
    per_rec.to_csv(out / "fma_shap_per_track.csv", index=False)

    # ── Aggregate by genre ────────────────────────────────────────────────────
    agg_rows = []
    for genre in MAPPED_GENRES:
        g = per_rec[per_rec["genre_gtzan"] == genre]
        n = len(g)
        if n == 0:
            continue
        row_agg = {"genre_gtzan": genre, "n": n}
        for b in AUDIT_BANDS:
            for prefix in ("shap_abs", "shap_signed", "input"):
                col = f"{prefix}_{b}"
                if col in g.columns:
                    row_agg[f"{col}_mean"] = float(g[col].mean())
                    row_agg[f"{col}_std"] = float(g[col].std())
        agg_rows.append(row_agg)
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(out / "fma_genre_aggregates.csv", index=False)

    # ── Run metadata ──────────────────────────────────────────────────────────
    (out / "run_metadata.json").write_text(json.dumps({
        "n_mapped_tracks": int(len(mapped)),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_correct": int(len(correct_df)),
        "val_accuracy": float(val_acc),
        "mapped_genres": MAPPED_GENRES,
        "fma_to_gtzan": FMA_TO_GTZAN,
        "note": (
            "3-class model trained on FMA-small subset. "
            "Random 80/20 split, not artist-disjoint. "
            "SHAP backgrounds from training data only. "
            "FMA and GTZAN label equivalence is NOT assumed."
        ),
    }, indent=2) + "\n")

    print(f"\nFMA analysis complete. Results in {out}")


if __name__ == "__main__":
    main()
