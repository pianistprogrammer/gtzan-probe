"""Cross-dataset transfer experiment.

Applies each trained model to data from the other dataset for the three
shared genre labels (hiphop, pop, rock):

  Direction A — GTZAN model on FMA-small data
  Direction B — FMA model on GTZAN data

Combined with the within-dataset results already computed, this produces a
2x2 design:
  (GTZAN model, GTZAN data)  — results/revision_audit/band_profiles.csv
  (GTZAN model, FMA data)    — results/cross_dataset/gtzan_model_on_fma.csv
  (FMA model,   FMA data)    — results/fma_analysis/fma_genre_aggregates.csv
  (FMA model,   GTZAN data)  — results/cross_dataset/fma_model_on_gtzan.csv

Run from project root:
    uv run python -m gtzan_probe.cross_dataset --output results/cross_dataset
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import shap
import torch
from tqdm.auto import tqdm

from .config import DATA_DIR, MODELS_DIR, SR, N_MELS, N_FFT, HOP, FIXED_W, SEG_S, get_device
from .dataset import load_spectrogram, resolve_audio_path
from .model import MusicCNN
from .fma_pipeline import fma_audio_path, load_fma_spectrogram

AUDIT_BANDS = {"B1": (0, 10), "B2": (10, 30), "B3": (30, 80), "B4": (80, 128)}
SHARED_GENRES = ["hiphop", "pop", "rock"]   # genres present in both datasets
N_SHAP_BG = 30
FMA_ROOT = DATA_DIR / "fma_archive" / "fma_small" / "fma_small"


def band_stats(records: list[dict], genre_col: str = "genre") -> pd.DataFrame:
    rows = []
    df = pd.DataFrame(records)
    for genre in SHARED_GENRES:
        g = df[df[genre_col] == genre]
        if len(g) == 0:
            continue
        row = {genre_col: genre, "n": len(g)}
        for b in AUDIT_BANDS:
            for prefix in ("shap_abs", "input"):
                col = f"{prefix}_{b}"
                if col in g.columns:
                    row[f"{col}_mean"] = float(g[col].mean())
                    row[f"{col}_std"] = float(g[col].std())
        rows.append(row)
    return pd.DataFrame(rows)


def build_bg(tensors: list[torch.Tensor], device: str, n: int = N_SHAP_BG) -> torch.Tensor:
    return torch.cat(tensors[:n], dim=0).to(device)


# ── Direction A: GTZAN model on FMA data ─────────────────────────────────────

def gtzan_model_on_fma(out: Path, device: str) -> None:
    print("\n=== Direction A: GTZAN model on FMA-small data ===")
    ckpt = MODELS_DIR / "cnn_gtzan_best.pt"
    with open(MODELS_DIR / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    model = MusicCNN(n_classes=10).to(device).eval()
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))

    # Load FMA metadata and restrict to shared genres
    fma_val = pd.read_csv(out.parent / "fma_analysis" / "fma_predictions.csv")
    fma_val = fma_val[fma_val["genre_gtzan"].isin(SHARED_GENRES)].reset_index(drop=True)

    # Evaluate GTZAN model on FMA clips
    records = []
    for _, row in tqdm(fma_val.iterrows(), total=len(fma_val), desc="A: evaluate"):
        path = fma_audio_path(FMA_ROOT, int(row["track_id"]))
        if not path.exists():
            continue
        result = load_fma_spectrogram(path)
        if result is None:
            continue
        spec_np, spec_t = result
        spec_t = spec_t.to(device)
        with torch.no_grad():
            pred_idx = model(spec_t).argmax(1).item()
        pred_label = le.inverse_transform([pred_idx])[0]
        correct = int(pred_label == row["genre_gtzan"])
        records.append({"track_id": int(row["track_id"]),
                        "genre": row["genre_gtzan"],
                        "pred": pred_label,
                        "correct": correct})

    eval_df = pd.DataFrame(records)
    eval_df.to_csv(out / "gtzan_model_on_fma_predictions.csv", index=False)
    acc = eval_df["correct"].mean()
    print(f"  Accuracy: {acc:.3f} ({eval_df['correct'].sum()}/{len(eval_df)})")
    print("  Per genre:", eval_df.groupby("genre")["correct"].mean().round(3).to_dict())

    correct_df = eval_df[eval_df["correct"] == 1]

    # SHAP backgrounds from GTZAN training data
    split_df = pd.read_csv(DATA_DIR.parent / "results" / "revision_audit" / "legacy_split.csv")[["filename","split"]]
    meta_df = pd.read_csv(DATA_DIR / "gtzan_metadata.csv").merge(split_df, on="filename", how="left")
    train_meta = meta_df[meta_df["split"] == "train"]
    bg_tensors = []
    for genre in SHARED_GENRES:
        for _, r in train_meta[train_meta["genre"] == genre].head(10).iterrows():
            _, t = load_spectrogram(r["filepath"])
            bg_tensors.append(t)
            if len(bg_tensors) >= N_SHAP_BG:
                break
    bg = build_bg(bg_tensors, device)
    explainer = shap.DeepExplainer(model, bg)

    shap_records = []
    for _, row in tqdm(correct_df.iterrows(), total=len(correct_df), desc="A: SHAP"):
        path = fma_audio_path(FMA_ROOT, int(row["track_id"]))
        if not path.exists():
            continue
        result = load_fma_spectrogram(path)
        if result is None:
            continue
        spec_np, spec_t = result
        spec_t = spec_t.to(device)
        true_idx = int(le.transform([row["genre"]])[0])
        shap_vals = explainer.shap_values(spec_t, check_additivity=False)
        shap_map = shap_vals[0, 0, :, :, true_idx]
        rec = {"track_id": int(row["track_id"]), "genre": row["genre"]}
        for b, (lo, hi) in AUDIT_BANDS.items():
            rec[f"shap_abs_{b}"] = float(np.mean(np.abs(shap_map[lo:hi, :])))
            rec[f"input_{b}"] = float(np.mean(spec_np[lo:hi, :]))
        shap_records.append(rec)

    pd.DataFrame(shap_records).to_csv(out / "gtzan_model_on_fma_shap.csv", index=False)
    agg = band_stats(shap_records, genre_col="genre")
    agg.to_csv(out / "gtzan_model_on_fma_aggregates.csv", index=False)
    print(f"  SHAP computed for {len(shap_records)} clips")

    (out / "direction_a_metadata.json").write_text(json.dumps({
        "model": "MusicCNN (GTZAN, 10 classes)",
        "data": "FMA-small (hip-hop, pop, rock)",
        "n_evaluated": len(records),
        "n_correct": int(eval_df["correct"].sum()),
        "accuracy": float(acc),
        "note": "GTZAN model applied to out-of-distribution FMA data. "
                "Genre label equivalence is not assumed."
    }, indent=2) + "\n")


# ── Direction B: FMA model on GTZAN data ─────────────────────────────────────

def fma_model_on_gtzan(out: Path, device: str) -> None:
    print("\n=== Direction B: FMA model on GTZAN data ===")
    ckpt = MODELS_DIR / "fma_gtzan_best.pt"
    with open(MODELS_DIR / "fma_gtzan_label_encoder.pkl", "rb") as f:
        le = pickle.load(f)  # 3-class: hiphop, pop, rock

    model = MusicCNN(n_classes=3).to(device).eval()
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))

    # Load GTZAN evaluation clips for shared genres only
    pred_df = pd.read_csv(DATA_DIR / "test_predictions.csv")
    gtzan_eval = pred_df[pred_df["true_genre"].isin(SHARED_GENRES)].reset_index(drop=True)

    records = []
    for _, row in tqdm(gtzan_eval.iterrows(), total=len(gtzan_eval), desc="B: evaluate"):
        _, spec_t = load_spectrogram(row["filepath"])
        spec_t = spec_t.to(device)
        with torch.no_grad():
            pred_idx = model(spec_t).argmax(1).item()
        pred_label = le.inverse_transform([pred_idx])[0]
        correct = int(pred_label == row["true_genre"])
        records.append({"filename": row["filename"],
                        "genre": row["true_genre"],
                        "pred": pred_label,
                        "correct": correct})

    eval_df = pd.DataFrame(records)
    eval_df.to_csv(out / "fma_model_on_gtzan_predictions.csv", index=False)
    acc = eval_df["correct"].mean()
    print(f"  Accuracy: {acc:.3f} ({eval_df['correct'].sum()}/{len(eval_df)})")
    print("  Per genre:", eval_df.groupby("genre")["correct"].mean().round(3).to_dict())

    correct_df = eval_df[eval_df["correct"] == 1]

    # SHAP backgrounds from FMA training data
    fma_shap = pd.read_csv(out.parent / "fma_analysis" / "fma_shap_per_track.csv")
    meta_df = pd.read_csv(DATA_DIR / "gtzan_metadata.csv")

    # Use GTZAN training data as backgrounds (FMA model applied to GTZAN domain)
    split_df = pd.read_csv(DATA_DIR.parent / "results" / "revision_audit" / "legacy_split.csv")[["filename","split"]]
    train_meta = meta_df.merge(split_df, on="filename", how="left")
    train_meta = train_meta[
        (train_meta["split"] == "train") & (train_meta["genre"].isin(SHARED_GENRES))
    ]
    bg_tensors = []
    for genre in SHARED_GENRES:
        for _, r in train_meta[train_meta["genre"] == genre].head(10).iterrows():
            _, t = load_spectrogram(r["filepath"])
            bg_tensors.append(t)
            if len(bg_tensors) >= N_SHAP_BG:
                break
    bg = build_bg(bg_tensors, device)
    explainer = shap.DeepExplainer(model, bg)

    shap_records = []
    for _, row in tqdm(correct_df.iterrows(), total=len(correct_df), desc="B: SHAP"):
        filepath = meta_df.loc[meta_df["filename"] == row["filename"], "filepath"].iloc[0]
        resolved = resolve_audio_path(filepath, filename=row["filename"], genre=row["genre"])
        spec_np, spec_t = load_spectrogram(resolved)
        spec_t = spec_t.to(device)
        true_idx = int(le.transform([row["genre"]])[0])
        shap_vals = explainer.shap_values(spec_t, check_additivity=False)
        shap_map = shap_vals[0, 0, :, :, true_idx]
        rec = {"filename": row["filename"], "genre": row["genre"]}
        for b, (lo, hi) in AUDIT_BANDS.items():
            rec[f"shap_abs_{b}"] = float(np.mean(np.abs(shap_map[lo:hi, :])))
            rec[f"input_{b}"] = float(np.mean(spec_np[lo:hi, :]))
        shap_records.append(rec)

    pd.DataFrame(shap_records).to_csv(out / "fma_model_on_gtzan_shap.csv", index=False)
    agg = band_stats(shap_records, genre_col="genre")
    agg.to_csv(out / "fma_model_on_gtzan_aggregates.csv", index=False)
    print(f"  SHAP computed for {len(shap_records)} clips")

    (out / "direction_b_metadata.json").write_text(json.dumps({
        "model": "MusicCNN (FMA-small, 3 classes: hiphop/pop/rock)",
        "data": "GTZAN (hiphop, pop, rock evaluation clips)",
        "n_evaluated": len(records),
        "n_correct": int(eval_df["correct"].sum()),
        "accuracy": float(acc),
        "note": "FMA model applied to out-of-distribution GTZAN data. "
                "Genre label equivalence is not assumed."
    }, indent=2) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    if args.output.exists():
        raise FileExistsError(f"{args.output} already exists.")
    args.output.mkdir(parents=True)
    out = args.output

    device = get_device()
    print(f"Device: {device}")

    gtzan_model_on_fma(out, device)
    fma_model_on_gtzan(out, device)

    print(f"\nCross-dataset results saved to {out}")


if __name__ == "__main__":
    main()
