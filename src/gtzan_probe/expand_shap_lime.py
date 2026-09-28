"""Experiment A — All-recordings SHAP/LIME expansion.

Runs SHAP (DeepExplainer) and LIME on every correctly classified evaluation
recording (all 166 clips), not just one per label. Produces per-recording
band-mean tables and per-genre aggregates (mean ± std) for B1-B4.

Run from project root:
    uv run python -m gtzan_probe.expand_shap_lime --output results/all_shap_lime

Requires: models/cnn_gtzan_best.pt, models/label_encoder.pkl,
          data/test_predictions.csv, GTZAN audio accessible via GTZAN_ROOT.
Takes ~60-90 minutes on Apple MPS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import torch
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.color import gray2rgb
from tqdm.auto import tqdm

from .config import (
    DATA_DIR, MODELS_DIR, N_MELS, SR, GENRES, get_device, setup_plotting
)
from .dataset import load_spectrogram
from .model import MusicCNN

AUDIT_BANDS = {"B1": (0, 10), "B2": (10, 30), "B3": (30, 80), "B4": (80, 128)}
N_SHAP_BG = 50
N_LIME_PERTURB = 500
LIME_SEGMENTS = 40


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def band_means(arr: np.ndarray) -> dict[str, float]:
    """Mean value in each B1-B4 group for a 2-D (mel, time) array."""
    return {b: float(np.mean(arr[lo:hi, :])) for b, (lo, hi) in AUDIT_BANDS.items()}


def abs_band_means(arr: np.ndarray) -> dict[str, float]:
    return {b: float(np.mean(np.abs(arr[lo:hi, :]))) for b, (lo, hi) in AUDIT_BANDS.items()}


def lime_explain(spec_np: np.ndarray, model: MusicCNN, device: str,
                 true_label: int, n_perturb: int = N_LIME_PERTURB,
                 n_segments: int = LIME_SEGMENTS) -> np.ndarray:
    """Return signed LIME attribution map (same shape as spec_np)."""
    img3 = gray2rgb(spec_np).astype(np.float64)

    def predict_fn(images: np.ndarray) -> np.ndarray:
        batch = torch.tensor(
            images[:, :, :, 0], dtype=torch.float32
        ).unsqueeze(1).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(batch), dim=1).cpu().numpy()
        return probs

    seg = SegmentationAlgorithm("slic", n_segments=n_segments,
                                  compactness=10, sigma=1)
    explainer = lime_image.LimeImageExplainer(verbose=False)
    explanation = explainer.explain_instance(
        img3, predict_fn, top_labels=1,
        hide_color=0, num_samples=n_perturb,
        segmentation_fn=seg, random_seed=42,
    )
    _, mask = explanation.get_image_and_mask(
        true_label, positive_only=False, num_features=100, hide_rest=False
    )
    # Build a full signed map by repeating each segment's weight.
    segments = explanation.segments
    local_exp = explanation.local_exp.get(true_label, [])
    seg_weights = {seg_id: w for seg_id, w in local_exp}
    lime_map = np.zeros_like(spec_np, dtype=np.float64)
    for seg_id, w in seg_weights.items():
        lime_map[segments == seg_id] = w
    return lime_map


def build_shap_explainer(model: MusicCNN, meta: pd.DataFrame,
                          device: str) -> shap.DeepExplainer:
    """Build SHAP background from first 5 training entries per genre."""
    split_path = DATA_DIR.parent / "results" / "revision_audit" / "legacy_split.csv"
    split_df = pd.read_csv(split_path)[["filename", "split"]]
    meta = meta.merge(split_df, on="filename", how="left")
    train_meta = meta[meta["split"] == "train"]
    bg_rows = []
    for g in GENRES:
        rows = train_meta[train_meta["genre"] == g].head(5)
        bg_rows.append(rows)
    bg_meta = pd.concat(bg_rows).head(N_SHAP_BG)
    tensors = []
    for _, row in bg_meta.iterrows():
        _, t = load_spectrogram(row["filepath"])
        tensors.append(t)          # shape [1, 1, 128, 128]; cat → [N, 1, 128, 128]
    bg = torch.cat(tensors, dim=0).to(device)
    return shap.DeepExplainer(model, bg)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--shap-only", action="store_true",
                   help="Skip LIME (faster; useful for a quick test run)")
    args = p.parse_args()

    if args.output.exists():
        raise FileExistsError(
            f"{args.output} already exists. Choose a new directory."
        )
    args.output.mkdir(parents=True)
    out = args.output

    device = get_device()
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt = MODELS_DIR / "cnn_gtzan_best.pt"
    le_path = MODELS_DIR / "label_encoder.pkl"
    model = MusicCNN().to(device).eval()
    model.load_state_dict(
        torch.load(ckpt, map_location=device, weights_only=True)
    )
    with open(le_path, "rb") as f:
        le = pickle.load(f)

    # ── Load predictions and metadata ─────────────────────────────────────────
    pred_df = pd.read_csv(DATA_DIR / "test_predictions.csv")
    meta_df = pd.read_csv(DATA_DIR / "gtzan_metadata.csv")
    # Merge split info
    meta_merged = meta_df.merge(
        pred_df[["filename", "correct"]], on="filename", how="left"
    )
    correct_df = pred_df[pred_df["correct"] == 1].copy()
    print(f"Correctly classified: {len(correct_df)} / {len(pred_df)}")

    # ── Build SHAP explainer (training-only backgrounds) ─────────────────────
    print("Building SHAP explainer with training backgrounds...")
    explainer = build_shap_explainer(model, meta_df, device)

    # ── Per-recording loop ────────────────────────────────────────────────────
    records = []
    t0 = time.time()
    for i, row in tqdm(correct_df.iterrows(), total=len(correct_df),
                       desc="Explaining clips"):
        spec_np, spec_t = load_spectrogram(row["filepath"])
        spec_t = spec_t.to(device)

        # SHAP (requires grad — do NOT wrap in torch.no_grad())
        # SHAP 0.51.0 returns shape (batch, C, H, W, n_classes); class axis is last.
        shap_vals = explainer.shap_values(spec_t)
        true_idx = le.transform([row["true_genre"]])[0]
        shap_map = shap_vals[0, 0, :, :, true_idx]  # (128, 128) signed

        # LIME (optional)
        lime_map = None
        if not args.shap_only:
            lime_map = lime_explain(spec_np, model, device, true_idx)

        rec = {
            "filename": row["filename"],
            "genre": row["true_genre"],
        }
        for b, (lo, hi) in AUDIT_BANDS.items():
            rec[f"shap_abs_{b}"] = float(np.mean(np.abs(shap_map[lo:hi, :])))
            rec[f"shap_signed_{b}"] = float(np.mean(shap_map[lo:hi, :]))
            rec[f"input_{b}"] = float(np.mean(spec_np[lo:hi, :]))
            if lime_map is not None:
                rec[f"lime_abs_{b}"] = float(np.mean(np.abs(lime_map[lo:hi, :])))
                rec[f"lime_signed_{b}"] = float(np.mean(lime_map[lo:hi, :]))

        records.append(rec)

    elapsed = time.time() - t0
    print(f"Finished {len(records)} clips in {elapsed/60:.1f} min")

    per_recording = pd.DataFrame(records)
    per_recording.to_csv(out / "per_recording.csv", index=False)
    print(f"Saved {out/'per_recording.csv'}")

    # ── Aggregate by genre ────────────────────────────────────────────────────
    agg_rows = []
    for genre in GENRES:
        g = per_recording[per_recording["genre"] == genre]
        n = len(g)
        if n == 0:
            continue
        row_agg = {"genre": genre, "n": n}
        for b in AUDIT_BANDS:
            for prefix in ("shap_abs", "shap_signed", "input"):
                col = f"{prefix}_{b}"
                if col in g.columns:
                    row_agg[f"{col}_mean"] = float(g[col].mean())
                    row_agg[f"{col}_std"] = float(g[col].std())
            if not args.shap_only:
                for prefix in ("lime_abs", "lime_signed"):
                    col = f"{prefix}_{b}"
                    if col in g.columns:
                        row_agg[f"{col}_mean"] = float(g[col].mean())
                        row_agg[f"{col}_std"] = float(g[col].std())
        agg_rows.append(row_agg)

    agg = pd.DataFrame(agg_rows)
    agg.to_csv(out / "genre_aggregates.csv", index=False)
    print(f"Saved {out/'genre_aggregates.csv'}")

    # ── Save run metadata ─────────────────────────────────────────────────────
    meta_run = {
        "n_explained": len(records),
        "n_correct_total": int(pred_df["correct"].sum()),
        "shap_backgrounds": N_SHAP_BG,
        "lime_perturbations": N_LIME_PERTURB if not args.shap_only else "skipped",
        "lime_segments": LIME_SEGMENTS if not args.shap_only else "skipped",
        "elapsed_seconds": elapsed,
        "checkpoint_sha256": sha256(ckpt),
        "device": device,
        "note": (
            "Training-data backgrounds used for SHAP; "
            "backgrounds verified against reconstructed split. "
            "Evaluation partition reused as test set (validation overlap unresolved)."
        ),
    }
    (out / "run_metadata.json").write_text(
        __import__("json").dumps(meta_run, indent=2) + "\n"
    )

    # ── Quick figure: mean SHAP B1-B4 per genre ───────────────────────────────
    _plot_aggregates(agg, out, args.shap_only)
    print("Done.")


def _plot_aggregates(agg: pd.DataFrame, out: Path, shap_only: bool) -> None:
    setup_plotting()
    bands = list(AUDIT_BANDS.keys())
    genres = GENRES
    n_genres = len(genres)
    x = np.arange(n_genres)
    width = 0.2

    fig, axes = plt.subplots(1, 2 if not shap_only else 1,
                             figsize=(14 if not shap_only else 8, 5))
    if shap_only:
        axes = [axes]

    for ax, prefix, title in zip(
        axes,
        ["shap_abs", "lime_abs"] if not shap_only else ["shap_abs"],
        ["Mean |SHAP| per band (all correct clips)", "Mean |LIME| per band (all correct clips)"],
    ):
        for i, b in enumerate(bands):
            col = f"{prefix}_{b}_mean"
            err_col = f"{prefix}_{b}_std"
            vals = [
                float(agg.loc[agg["genre"] == g, col].iloc[0])
                if g in agg["genre"].values and col in agg.columns else 0.0
                for g in genres
            ]
            errs = [
                float(agg.loc[agg["genre"] == g, err_col].iloc[0])
                if g in agg["genre"].values and err_col in agg.columns else 0.0
                for g in genres
            ]
            ax.bar(x + i * width, vals, width, label=b, yerr=errs, capsize=3)

        ax.set_xticks(x + 1.5 * width)
        ax.set_xticklabels(genres, rotation=30, ha="right", fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("Mean absolute attribution")
        ax.legend(title="Band")
        n_vals = agg["n"].tolist() if "n" in agg.columns else []
        if n_vals:
            ax.set_xlabel(
                f"Genre (N per genre: {', '.join(f'{g}={n}' for g, n in zip(genres, n_vals))})",
                fontsize=9
            )

    fig.tight_layout()
    path = out / "genre_band_aggregates.pdf"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved figure: {path}")


if __name__ == "__main__":
    main()
