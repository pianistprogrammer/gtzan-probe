"""Shared configuration, paths, and plotting setup for gtzan-probe."""

import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# ── Project root (the directory containing pyproject.toml) ────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# All data lives ON the external drive, inside the project
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

# GTZAN can be overridden via env var; defaults to data/gtzan/genres_original
GTZAN_ROOT = Path(
    os.environ.get("GTZAN_ROOT", str(DATA_DIR / "gtzan" / "genres_original"))
)

# Ensure output directories exist
for d in (DATA_DIR, FIGURES_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Audio constants ───────────────────────────────────────────────────────────
SR = 22050
DURATION = 30.0
SEG_S = 3.0
N_MELS = 128
HOP = 512
N_FFT = 2048
FIXED_W = 128  # fixed time-axis width for CNN input

# ── Genre list & colours ─────────────────────────────────────────────────────
GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]

GENRE_PALETTE = [
    "#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0",
    "#00BCD4", "#E91E63", "#8BC34A", "#FF5722", "#607D8B",
]

GENRE_COLORS = dict(zip(GENRES, GENRE_PALETTE))

# ── Legacy partial exclusion list (NOT a complete Sturm 2013 audit) ──────────────────────────────────────────
KNOWN_FAULTS = {
    "blues.00068.wav": "corrupt",
    "jazz.00054.wav": "corrupt",
    "jazz.00055.wav": "duplicate",
    "jazz.00056.wav": "duplicate",
    "jazz.00057.wav": "duplicate",
    "jazz.00058.wav": "duplicate",
    "rock.00081.wav": "mislabeled",
    "classical.00049.wav": "mislabeled",
}

# ── Legacy arbitrary spectral priors: NOT musicological ground truth ─────────────────────────
BAND_RANGES = {
    "sub-bass": (0, 10),
    "bass/rhythm": (10, 30),
    "harmony": (30, 80),
    "timbre": (80, 128),
}

GT_BANDS = {
    "blues":     {"bass/rhythm": 1.0, "harmony": 0.7, "timbre": 0.4, "sub-bass": 0.3},
    "classical": {"harmony": 1.0, "timbre": 0.9, "sub-bass": 0.2, "bass/rhythm": 0.3},
    "country":   {"bass/rhythm": 0.8, "harmony": 0.9, "timbre": 0.6, "sub-bass": 0.2},
    "disco":     {"bass/rhythm": 1.0, "sub-bass": 0.8, "harmony": 0.5, "timbre": 0.4},
    "hiphop":    {"sub-bass": 1.0, "bass/rhythm": 0.9, "harmony": 0.3, "timbre": 0.3},
    "jazz":      {"harmony": 1.0, "bass/rhythm": 0.8, "timbre": 0.7, "sub-bass": 0.2},
    "metal":     {"timbre": 1.0, "bass/rhythm": 0.9, "harmony": 0.6, "sub-bass": 0.5},
    "pop":       {"bass/rhythm": 0.8, "harmony": 0.8, "timbre": 0.6, "sub-bass": 0.4},
    "reggae":    {"sub-bass": 1.0, "bass/rhythm": 0.9, "harmony": 0.4, "timbre": 0.3},
    "rock":      {"bass/rhythm": 1.0, "timbre": 0.9, "harmony": 0.5, "sub-bass": 0.4},
}

# ── Publication-grade matplotlib config ───────────────────────────────────────
FIGURE_DPI = 600  # high-resolution raster fallback

def setup_plotting():
    """Apply publication-quality matplotlib defaults."""
    plt.rcParams.update({
        "figure.dpi": FIGURE_DPI,
        "savefig.dpi": FIGURE_DPI,
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.titlesize": 16,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.prop_cycle": matplotlib.cycler(color=GENRE_PALETTE),
        # PDF/vector defaults
        "pdf.fonttype": 42,       # TrueType fonts in PDF (editable text)
        "ps.fonttype": 42,
        "svg.fonttype": "none",   # keep text as text in SVG
    })


def savefig(name, fig=None, tight=True):
    """Save figure as PDF (vector, for LaTeX) + high-DPI PNG (preview).

    PDFs use \\includegraphics{figures/<name>.pdf} directly in LaTeX.
    """
    if fig is None:
        fig = plt.gcf()
    if tight:
        fig.tight_layout()

    # Primary: PDF vector — crisp at any zoom, ideal for \includegraphics
    pdf_path = FIGURES_DIR / f"{name}.pdf"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor="white")
    print(f"  ✓ Saved {pdf_path}")

    # Secondary: high-DPI PNG for quick preview / non-LaTeX use
    png_path = FIGURES_DIR / f"{name}.png"
    fig.savefig(png_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    print(f"  ✓ Saved {png_path} ({FIGURE_DPI} DPI)")

    plt.close(fig)


# ── Device selection ──────────────────────────────────────────────────────────
def get_device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
