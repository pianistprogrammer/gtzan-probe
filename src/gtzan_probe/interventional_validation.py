"""Interventional validation and cross-dataset correlation (GTZAN vs FMA-small).

Usage:
    python -m gtzan_probe.interventional_validation [--download-fma]
"""

from __future__ import annotations

import argparse
import ast
import pickle
import urllib.request
import zipfile
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
from tqdm.auto import tqdm

from .config import (
    BAND_RANGES,
    DATA_DIR,
    FIGURES_DIR,
    GENRES,
    MODELS_DIR,
    N_FFT,
    N_MELS,
    HOP,
    SR,
    setup_plotting,
    savefig,
    get_device,
)
from .dataset import load_spectrogram, resolve_audio_path
from .model import MusicCNN

FMA_SMALL_URL = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
FMA_META_URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--download-fma",
        action="store_true",
        help="Download FMA-small + metadata if missing.",
    )
    p.add_argument(
        "--fma-root",
        type=Path,
        default=Path(DATA_DIR / "fma" / "fma_small"),
        help="Path to extracted fma_small audio root.",
    )
    p.add_argument(
        "--fma-meta-root",
        type=Path,
        default=Path(DATA_DIR / "fma" / "fma_metadata"),
        help="Path to extracted fma_metadata root.",
    )
    p.add_argument("--max-fma-per-genre", type=int, default=80)
    p.add_argument("--bootstrap", type=int, default=2000)
    return p.parse_args()


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True)
    means = samples.mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _predict_probs(model: torch.nn.Module, spec_tensor: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        probs = F.softmax(model(spec_tensor), dim=1).cpu().numpy()[0]
    return probs


def _mask_silence(spec_np: np.ndarray, low_q: float = 10.0) -> np.ndarray:
    masked = spec_np.copy()
    thr = np.percentile(spec_np, low_q)
    masked[spec_np <= thr] = 0.0
    return masked


def _mask_band(spec_np: np.ndarray, lo: int, hi: int) -> np.ndarray:
    masked = spec_np.copy()
    masked[lo:hi, :] = 0.0
    return masked


def _to_model_tensor(spec_np: np.ndarray, device: str) -> torch.Tensor:
    return torch.tensor(spec_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)


def _band_profile(spec_np: np.ndarray) -> dict[str, float]:
    return {band: float(spec_np[lo:hi, :].mean()) for band, (lo, hi) in BAND_RANGES.items()}


def _download_zip(url: str, out_zip: Path):
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    if out_zip.exists():
        # Guard against interrupted partial downloads.
        if zipfile.is_zipfile(out_zip):
            return
        out_zip.unlink(missing_ok=True)
    print(f"Downloading {url} -> {out_zip}")
    urllib.request.urlretrieve(url, out_zip)


def ensure_fma_available(fma_root: Path, fma_meta_root: Path, download_fma: bool):
    small_ok = fma_root.exists() and any(fma_root.rglob("*.mp3"))
    meta_ok = (fma_meta_root / "tracks.csv").exists()
    if small_ok and meta_ok:
        return
    if not download_fma:
        raise FileNotFoundError(
            f"FMA not found. Expected audio at {fma_root} and metadata at {fma_meta_root}. "
            "Run with --download-fma to fetch official FMA-small files."
        )

    fma_base = Path(DATA_DIR / "fma")
    small_zip = fma_base / "fma_small.zip"
    meta_zip = fma_base / "fma_metadata.zip"
    _download_zip(FMA_SMALL_URL, small_zip)
    _download_zip(FMA_META_URL, meta_zip)

    if not fma_root.exists():
        print(f"Extracting {small_zip} ...")
        with zipfile.ZipFile(small_zip, "r") as zf:
            zf.extractall(fma_base)
    if not fma_meta_root.exists():
        print(f"Extracting {meta_zip} ...")
        with zipfile.ZipFile(meta_zip, "r") as zf:
            zf.extractall(fma_base)


def _map_fma_to_gtzan(genre_top: str | float) -> str | None:
    if not isinstance(genre_top, str):
        return None
    g = genre_top.lower().strip()
    mapping = {
        "hip-hop": "hiphop",
        "hip hop": "hiphop",
        "classical": "classical",
        "country": "country",
        "jazz": "jazz",
        "metal": "metal",
        "pop": "pop",
        "reggae": "reggae",
        "rock": "rock",
        "blues": "blues",
        "disco": "disco",
    }
    for k, v in mapping.items():
        if k in g:
            return v
    return None


def _parse_genre_ids(value: str | float) -> list[int]:
    if not isinstance(value, str):
        return []
    try:
        parsed = ast.literal_eval(value)
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    out = []
    for x in parsed:
        try:
            out.append(int(x))
        except Exception:
            continue
    return out


def load_fma_tracks(fma_meta_root: Path) -> pd.DataFrame:
    tracks_path = fma_meta_root / "tracks.csv"
    genres_path = fma_meta_root / "genres.csv"
    tracks = pd.read_csv(tracks_path, index_col=0, header=[0, 1], low_memory=False)
    genres = pd.read_csv(genres_path)
    genres["genre_id"] = genres["genre_id"].astype(int)
    id_to_title = {int(r["genre_id"]): str(r["title"]).lower() for _, r in genres.iterrows()}
    df = pd.DataFrame({
        "track_id": tracks.index.astype(int),
        "subset": tracks[("set", "subset")].astype(str),
        "genre_top": tracks[("track", "genre_top")],
        "genres_all": tracks[("track", "genres_all")].astype(str),
    })
    df["gtzan_genre"] = df["genre_top"].apply(_map_fma_to_gtzan)
    df["mapping_source"] = np.where(df["gtzan_genre"].notna(), "genre_top", "")
    df["mapping_evidence"] = np.where(df["gtzan_genre"].notna(), df["genre_top"].astype(str), "")

    # Expanded, still conservative mapping using explicit subgenre tags.
    # Specific tags (e.g., metal, reggae) take priority over broad top-level labels.
    priority_rules = [
        ("metal", "metal"),
        ("reggae", "reggae"),
    ]

    all_ids = df["genres_all"].apply(_parse_genre_ids)
    all_titles = all_ids.apply(lambda ids: [id_to_title.get(i, "") for i in ids])
    for fma_kw, target in priority_rules:
        mask = all_titles.apply(lambda ts: any(fma_kw in t for t in ts))
        df.loc[mask, "gtzan_genre"] = target
        df.loc[mask, "mapping_source"] = "genres_all_keyword"
        df.loc[mask, "mapping_evidence"] = fma_kw

    return df


def fma_track_path(fma_root: Path, track_id: int) -> Path:
    tid = f"{int(track_id):06d}"
    return fma_root / tid[:3] / f"{tid}.mp3"


def compute_gtzan_empirical_profiles(test_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="GTZAN empirical profiles"):
        genre = row["true_genre"]
        path = resolve_audio_path(row["filepath"], filename=row.get("filename"), genre=genre)
        try:
            spec_np, _ = load_spectrogram(path)
        except Exception:
            continue
        prof = _band_profile(spec_np)
        prof["genre"] = genre
        rows.append(prof)
    prof_df = pd.DataFrame(rows)
    return prof_df.groupby("genre", as_index=False).mean(numeric_only=True)


def compute_fma_empirical_profiles(
    fma_root: Path,
    tracks_df: pd.DataFrame,
    max_per_genre: int,
) -> pd.DataFrame:
    rows = []
    keep = tracks_df[(tracks_df["subset"].str.lower() == "small") & tracks_df["gtzan_genre"].notna()].copy()
    for genre in GENRES:
        gdf = keep[keep["gtzan_genre"] == genre].head(max_per_genre)
        for _, tr in tqdm(gdf.iterrows(), total=len(gdf), desc=f"FMA {genre}"):
            p = fma_track_path(fma_root, int(tr["track_id"]))
            if not p.exists():
                continue
            try:
                y, _ = librosa.load(str(p), sr=SR, duration=3.0)
            except Exception:
                continue
            if len(y) < int(3.0 * SR):
                y = np.pad(y, (0, int(3.0 * SR) - len(y)))
            S = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP)
            S_db = librosa.power_to_db(S, ref=np.max)
            S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
            from skimage.transform import resize
            S_db = resize(S_db, (N_MELS, 128), anti_aliasing=True)
            prof = _band_profile(S_db)
            prof["genre"] = genre
            rows.append(prof)
    if not rows:
        return pd.DataFrame(columns=["genre", *BAND_RANGES.keys()])
    prof_df = pd.DataFrame(rows)
    return prof_df.groupby("genre", as_index=False).mean(numeric_only=True)


def run_interventions(args):
    setup_plotting()
    device = get_device()
    print(f"Device: {device}")

    model = MusicCNN().to(device)
    model.load_state_dict(torch.load(MODELS_DIR / "cnn_gtzan_best.pt", map_location=device))
    model.eval()

    with open(MODELS_DIR / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    test_df = pd.read_csv(DATA_DIR / "test_predictions.csv")
    if "true_genre" not in test_df.columns:
        raise ValueError("Expected true_genre column in test_predictions.csv")

    silence_rows = []
    band_rows = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Interventional ablations"):
        true_genre = row["true_genre"]
        filepath = resolve_audio_path(row["filepath"], filename=row.get("filename"), genre=true_genre)
        try:
            spec_np, spec_tensor = load_spectrogram(filepath)
        except Exception:
            continue
        spec_tensor = spec_tensor.to(device)
        probs = _predict_probs(model, spec_tensor)

        true_idx = int(le.transform([true_genre])[0])
        pred_idx = int(np.argmax(probs))
        conf_true = float(probs[true_idx])
        conf_pred = float(probs[pred_idx])

        # Silence ablation
        silence_masked = _mask_silence(spec_np)
        probs_sil = _predict_probs(model, _to_model_tensor(silence_masked, device))
        silence_rows.append({
            "filename": row["filename"],
            "genre": true_genre,
            "true_conf_orig": conf_true,
            "true_conf_silence_masked": float(probs_sil[true_idx]),
            "delta_true_conf_silence": conf_true - float(probs_sil[true_idx]),
            "pred_conf_orig": conf_pred,
            "pred_conf_silence_masked": float(probs_sil[pred_idx]),
            "delta_pred_conf_silence": conf_pred - float(probs_sil[pred_idx]),
        })

        # Frequency-band ablation
        for band, (lo, hi) in BAND_RANGES.items():
            masked = _mask_band(spec_np, lo, hi)
            probs_band = _predict_probs(model, _to_model_tensor(masked, device))
            band_rows.append({
                "filename": row["filename"],
                "genre": true_genre,
                "band": band,
                "true_conf_orig": conf_true,
                "true_conf_band_masked": float(probs_band[true_idx]),
                "delta_true_conf_band": conf_true - float(probs_band[true_idx]),
                "pred_conf_orig": conf_pred,
                "pred_conf_band_masked": float(probs_band[pred_idx]),
                "delta_pred_conf_band": conf_pred - float(probs_band[pred_idx]),
            })

    silence_df = pd.DataFrame(silence_rows)
    band_df = pd.DataFrame(band_rows)
    silence_df.to_csv(DATA_DIR / "intervention_silence_ablation.csv", index=False)
    band_df.to_csv(DATA_DIR / "intervention_band_ablation.csv", index=False)
    print("Saved intervention_silence_ablation.csv and intervention_band_ablation.csv")

    # Bootstrap CIs
    boot_rows = []
    for genre, gdf in silence_df.groupby("genre"):
        vals = gdf["delta_true_conf_silence"].values
        lo, hi = bootstrap_ci(vals, n_boot=args.bootstrap, seed=42)
        boot_rows.append({
            "metric": "silence_ablation_delta_true_conf",
            "group": genre,
            "mean": float(np.mean(vals)),
            "ci_low": lo,
            "ci_high": hi,
            "n": len(vals),
        })
    for (genre, band), gdf in band_df.groupby(["genre", "band"]):
        vals = gdf["delta_true_conf_band"].values
        lo, hi = bootstrap_ci(vals, n_boot=args.bootstrap, seed=42)
        boot_rows.append({
            "metric": "band_ablation_delta_true_conf",
            "group": f"{genre}:{band}",
            "mean": float(np.mean(vals)),
            "ci_low": lo,
            "ci_high": hi,
            "n": len(vals),
        })

    # Existing reported metrics: resample genres to get global CIs.
    align_df = pd.read_csv(DATA_DIR / "musicological_alignment.csv")
    agree_df = pd.read_csv(DATA_DIR / "method_agreement.csv")
    faith_df = pd.read_csv(DATA_DIR / "faithfulness_results.csv")
    for metric_name, values in [
        ("alignment_spearman_r_global", align_df["spearman_r"].values),
        ("agreement_pearson_global", agree_df["pearson_r"].values),
        ("agreement_iou_global", agree_df["top10_iou"].values),
    ]:
        lo, hi = bootstrap_ci(np.asarray(values), n_boot=args.bootstrap, seed=123)
        boot_rows.append({
            "metric": metric_name,
            "group": "all_genres",
            "mean": float(np.mean(values)),
            "ci_low": lo,
            "ci_high": hi,
            "n": len(values),
        })

    for k, gdf in faith_df.groupby("k_fraction"):
        vals = gdf["conf_drop"].values
        lo, hi = bootstrap_ci(vals, n_boot=args.bootstrap, seed=99)
        boot_rows.append({
            "metric": "faithfulness_conf_drop",
            "group": f"k={k}",
            "mean": float(np.mean(vals)),
            "ci_low": lo,
            "ci_high": hi,
            "n": len(vals),
        })

    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(DATA_DIR / "bootstrap_confidence_intervals.csv", index=False)
    print("Saved bootstrap_confidence_intervals.csv")

    # Data-driven empirical baseline on GTZAN test clips.
    gtzan_prof = compute_gtzan_empirical_profiles(test_df)
    gtzan_prof.to_csv(DATA_DIR / "gtzan_empirical_band_profiles.csv", index=False)
    print("Saved gtzan_empirical_band_profiles.csv")

    # Compare SHAP band profile vs empirical profile per genre
    with open(DATA_DIR / "shap_values.pkl", "rb") as f:
        shap_results = pickle.load(f)
    align_rows = []
    for genre in GENRES:
        if genre not in shap_results:
            continue
        sv = np.abs(shap_results[genre]["shap_values"])
        shap_vec = np.array([sv[lo:hi, :].mean() for lo, hi in BAND_RANGES.values()], dtype=float)
        emp_row = gtzan_prof[gtzan_prof["genre"] == genre]
        if emp_row.empty:
            continue
        emp_vec = emp_row[list(BAND_RANGES.keys())].values[0].astype(float)
        rho, p = stats.spearmanr(shap_vec, emp_vec)
        align_rows.append({"genre": genre, "spearman_r": rho, "p_value": p})
    emp_align_df = pd.DataFrame(align_rows)
    emp_align_df.to_csv(DATA_DIR / "alignment_empirical_vs_shap.csv", index=False)
    print("Saved alignment_empirical_vs_shap.csv")

    # FMA cross-dataset correlation
    try:
        ensure_fma_available(args.fma_root, args.fma_meta_root, args.download_fma)
        fma_tracks = load_fma_tracks(args.fma_meta_root)

        cov = (
            fma_tracks[fma_tracks["subset"].str.lower() == "small"]
            .groupby(["gtzan_genre", "mapping_source"], dropna=False)
            .size()
            .reset_index(name="n_tracks")
            .sort_values(["n_tracks", "gtzan_genre"], ascending=[False, True])
        )
        cov.to_csv(DATA_DIR / "fma_gtzan_mapping_coverage.csv", index=False)
        print("Saved fma_gtzan_mapping_coverage.csv")

        fma_prof = compute_fma_empirical_profiles(args.fma_root, fma_tracks, args.max_fma_per_genre)
        fma_prof.to_csv(DATA_DIR / "fma_empirical_band_profiles.csv", index=False)

        common = sorted(set(gtzan_prof["genre"]).intersection(set(fma_prof["genre"])))
        corr_rows = []
        for genre in common:
            gvec = gtzan_prof[gtzan_prof["genre"] == genre][list(BAND_RANGES.keys())].values[0].astype(float)
            fvec = fma_prof[fma_prof["genre"] == genre][list(BAND_RANGES.keys())].values[0].astype(float)
            rho, p = stats.spearmanr(gvec, fvec)
            corr_rows.append({
                "genre": genre,
                "spearman_r": rho,
                "p_value": p,
            })

        if corr_rows:
            corr_df = pd.DataFrame(corr_rows)
            corr_df.to_csv(DATA_DIR / "fma_gtzan_band_correlation.csv", index=False)
            print("Saved fma_gtzan_band_correlation.csv")

            fig, ax = plt.subplots(figsize=(10, 4.5))
            ax.bar(corr_df["genre"], corr_df["spearman_r"], color="#4CAF50")
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set_ylabel("Spearman ρ")
            ax.set_title("GTZAN vs FMA-small Band-Profile Correlation by Genre", fontweight="bold")
            ax.tick_params(axis="x", labelrotation=30)
            savefig("07_fma_gtzan_correlation", fig)
    except Exception as e:
        # Keep GTZAN interventions reproducible even if FMA fetch is unavailable.
        print(f"FMA correlation step skipped: {e}")

    print("✓ Interventional validation complete.")
    print(f"Outputs written to {DATA_DIR} and {FIGURES_DIR}.")


def main():
    args = parse_args()
    run_interventions(args)


if __name__ == "__main__":
    main()
