# GTZAN-Probe — XAI for Music Genre Classification

Explainability research on the GTZAN dataset: train a CNN classifier, then interrogate it with SHAP, LIME, and Transformer attention to determine whether predictions are driven by musicological features or dataset artifacts.

## Prerequisites

- **Python 3.13+**
- **[uv](https://docs.astral.sh/uv/)** package manager
- **GTZAN dataset** (~1.2 GB of `.wav` files)

## Setup

```bash
# 1. Create virtual environment & install all dependencies
uv venv
uv add numpy pandas librosa soundfile matplotlib seaborn \
       scikit-learn scikit-image tqdm torch torchvision \
       shap lime transformers scipy kaggle

# 2. Install the project package (editable)
uv pip install -e .
```

## Get the Dataset

**Option A — Kaggle API (automated):**

```bash
# Place your Kaggle token
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# The prepare step will auto-download via Kaggle API
```

**Option B — Manual download:**

1. Download from [Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
2. Unzip so the structure is:

```
data/gtzan/genres_original/
├── blues/
│   ├── blues.00000.wav
│   └── ...
├── classical/
├── country/
├── disco/
├── hiphop/
├── jazz/
├── metal/
├── pop/
├── reggae/
└── rock/
```

> **Tip:** Override the dataset path with `export GTZAN_ROOT=/your/path/to/genres_original` if it lives elsewhere.

## Pipeline

Run each step in order. Every command uses the project's venv automatically:

```bash
# Step 1 — Prepare dataset (download, audit faults, extract features, EDA figures)
.venv/bin/python -m gtzan_probe.prepare_dataset

# Step 2 — Train the CNN classifier
.venv/bin/python -m gtzan_probe.train

# Step 3 — Evaluate (confusion matrix, per-class metrics, save predictions)
.venv/bin/python -m gtzan_probe.evaluate

# Step 4 — SHAP explanations
.venv/bin/python -m gtzan_probe.explain_shap

# Step 5 — LIME explanations
.venv/bin/python -m gtzan_probe.explain_lime

# Step 6 — Quantitative validation (alignment, faithfulness, agreement, spurious detection)
.venv/bin/python -m gtzan_probe.validate

# Step 7 — Fine-tune AST & attention visualization
.venv/bin/python -m gtzan_probe.explain_attention

# Step 8 — Generate final publication figures & LaTeX tables
.venv/bin/python -m gtzan_probe.generate_figures
```

### Training options

```bash
.venv/bin/python -m gtzan_probe.train --epochs 60 --batch-size 32 --lr 1e-3
```

## Outputs

| Directory  | Contents                                              |
|------------|-------------------------------------------------------|
| `data/`    | Metadata CSVs, feature matrices, SHAP/LIME pickles, LaTeX tables |
| `models/`  | `cnn_gtzan_best.pt`, `ast_gtzan_best.pt`, label encoder |
| `figures/` | PDF (vector) + 600 DPI PNG for every figure            |

All outputs are written to the **project directory** (stays on the external drive).

### Using figures in LaTeX

Every figure is saved as both PDF and PNG. For LaTeX:

```latex
\includegraphics[width=\columnwidth]{figures/FINAL_01_shap_hero.pdf}
```

## Project Structure

```
src/gtzan_probe/
├── config.py              # Paths, constants, plot setup, savefig()
├── model.py               # MusicCNN architecture
├── dataset.py             # GTZANDataset, load_spectrogram()
├── prepare_dataset.py     # Step 1: download, audit, features, EDA
├── train.py               # Step 2: CNN training loop
├── evaluate.py            # Step 3: test-set evaluation & metrics
├── explain_shap.py        # Step 4: SHAP DeepExplainer
├── explain_lime.py        # Step 5: LIME superpixel explanations
├── validate.py            # Step 6: quantitative XAI metrics
├── explain_attention.py   # Step 7: AST fine-tune + attention maps
└── generate_figures.py    # Step 8: camera-ready figures & LaTeX
```
