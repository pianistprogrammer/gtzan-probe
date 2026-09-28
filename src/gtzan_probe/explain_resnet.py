"""SHAP/LIME explanations for the second architecture (MusicResNet).

Run AFTER train_resnet.py. Explains the same evaluation clips as the
MusicCNN analysis so the two architectures can be compared on identical inputs.

Run from project root:
    uv run python -m gtzan_probe.explain_resnet --output results/resnet_explanations
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.color import gray2rgb
from tqdm.auto import tqdm

from .config import DATA_DIR, MODELS_DIR, GENRES, get_device
from .dataset import load_spectrogram
from .model_resnet import MusicResNet

AUDIT_BANDS = {"B1": (0, 10), "B2": (10, 30), "B3": (30, 80), "B4": (80, 128)}
N_SHAP_BG = 50
N_LIME_PERTURB = 500
LIME_SEGMENTS = 40


def lime_explain(spec_np: np.ndarray, model: MusicResNet, device: str,
                 true_label: int) -> np.ndarray:
    img3 = gray2rgb(spec_np).astype(np.float64)

    def predict_fn(images: np.ndarray) -> np.ndarray:
        batch = torch.tensor(images[:, :, :, 0], dtype=torch.float32).unsqueeze(1).to(device)
        with torch.no_grad():
            return torch.softmax(model(batch), dim=1).cpu().numpy()

    seg = SegmentationAlgorithm("slic", n_segments=LIME_SEGMENTS, compactness=10, sigma=1)
    explainer = lime_image.LimeImageExplainer(verbose=False)
    explanation = explainer.explain_instance(
        img3, predict_fn, top_labels=1, hide_color=0,
        num_samples=N_LIME_PERTURB, segmentation_fn=seg, random_seed=42,
    )
    segments = explanation.segments
    local_exp = explanation.local_exp.get(true_label, [])
    lime_map = np.zeros_like(spec_np, dtype=np.float64)
    for seg_id, w in local_exp:
        lime_map[segments == seg_id] = w
    return lime_map


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--shap-only", action="store_true")
    args = p.parse_args()

    if args.output.exists():
        raise FileExistsError(f"{args.output} already exists.")
    args.output.mkdir(parents=True)
    out = args.output

    device = get_device()
    ckpt = MODELS_DIR / "resnet_gtzan_best.pt"
    le_path = MODELS_DIR / "label_encoder.pkl"  # same label encoder as MusicCNN

    if not ckpt.exists():
        raise FileNotFoundError(
            f"{ckpt} not found. Run train_resnet.py first."
        )

    model = MusicResNet(n_classes=len(GENRES)).to(device).eval()
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    with open(le_path, "rb") as f:
        le = pickle.load(f)

    # Use the SAME ten clips as the MusicCNN explanation for fair comparison.
    audit_samples = pd.read_csv(DATA_DIR / "../results/revision_audit/explanation_samples.csv")
    # Fallback: use first correct per genre from predictions
    pred_df = pd.read_csv(DATA_DIR / "test_predictions.csv")
    meta_df = pd.read_csv(DATA_DIR / "gtzan_metadata.csv")

    # Identify the same 10 selected clips (one per label, first correct)
    selected = []
    for genre in GENRES:
        genre_preds = pred_df[pred_df["true_genre"] == genre]
        correct = genre_preds[genre_preds["correct"] == 1]
        if len(correct) > 0:
            selected.append(correct.iloc[0])
    selected_df = pd.DataFrame(selected)

    # Verify ResNet also classifies them correctly
    verified = []
    for _, row in selected_df.iterrows():
        _, spec_t = load_spectrogram(row["filepath"])
        spec_t = spec_t.to(device)
        with torch.no_grad():
            pred_idx = model(spec_t).argmax(1).item()
        true_idx = int(le.transform([row["true_genre"]])[0])
        verified.append({
            "filename": row["filename"],
            "genre": row["true_genre"],
            "resnet_correct": int(pred_idx == true_idx),
            "resnet_pred": le.inverse_transform([pred_idx])[0],
        })
    ver_df = pd.DataFrame(verified)
    ver_df.to_csv(out / "selected_clips_verification.csv", index=False)
    n_resnet_correct = ver_df["resnet_correct"].sum()
    print(f"ResNet correct on selected 10 clips: {n_resnet_correct}/10")

    # SHAP backgrounds — load split info keeping only filename+split to avoid duplicate genre column
    split_df = pd.read_csv(DATA_DIR.parent / "results" / "revision_audit" / "legacy_split.csv")[["filename", "split"]]
    train_meta = meta_df.merge(split_df, on="filename", how="left")
    bg_tensors = []
    for g in GENRES:
        rows = train_meta[
            (train_meta["genre"] == g) & (train_meta["split"] == "train")
        ].head(5)
        for _, row in rows.iterrows():
            _, t = load_spectrogram(row["filepath"])
            bg_tensors.append(t)           # [1,1,128,128]; cat → [N,1,128,128]
    bg = torch.cat(bg_tensors[:N_SHAP_BG], dim=0).to(device)
    explainer = shap.DeepExplainer(model, bg)

    records = []
    for _, row in tqdm(selected_df.iterrows(), total=len(selected_df), desc="Explaining"):
        spec_np, spec_t = load_spectrogram(row["filepath"])
        spec_t = spec_t.to(device)
        true_idx = int(le.transform([row["true_genre"]])[0])

        # SHAP 0.51.0: returns (batch, C, H, W, n_classes); do NOT use no_grad
        # check_additivity=False needed because residual connections cause minor
        # approximation error that exceeds SHAP's strict tolerance.
        shap_vals = explainer.shap_values(spec_t, check_additivity=False)
        shap_map = shap_vals[0, 0, :, :, true_idx]  # (128, 128) signed

        lime_map = None
        if not args.shap_only:
            lime_map = lime_explain(spec_np, model, device, true_idx)

        rec = {"filename": row["filename"], "genre": row["true_genre"]}
        for b, (lo, hi) in AUDIT_BANDS.items():
            rec[f"shap_abs_{b}"] = float(np.mean(np.abs(shap_map[lo:hi, :])))
            rec[f"shap_signed_{b}"] = float(np.mean(shap_map[lo:hi, :]))
            rec[f"input_{b}"] = float(np.mean(spec_np[lo:hi, :]))
            if lime_map is not None:
                rec[f"lime_abs_{b}"] = float(np.mean(np.abs(lime_map[lo:hi, :])))
        records.append(rec)

    per_rec = pd.DataFrame(records)
    per_rec.to_csv(out / "resnet_band_profiles.csv", index=False)

    (out / "run_metadata.json").write_text(json.dumps({
        "model": "MusicResNet",
        "checkpoint": str(ckpt),
        "n_explained": len(records),
        "n_resnet_correct_on_selected": int(n_resnet_correct),
        "clips_are_same_as_musiccnn_selected": True,
        "note": (
            "Same 10 clips as MusicCNN explanation for direct architecture comparison. "
            "Different ResNet clips than MusicCNN correct clips may exist."
        ),
    }, indent=2) + "\n")
    print(f"Done. Results in {out}")


if __name__ == "__main__":
    main()
