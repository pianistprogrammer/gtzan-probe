# GTZAN-Probe

An exploratory comparison of SHAP and LIME for a GTZAN MusicCNN, with historical AST attention visualizations. This repository is being corrected following an audit of the submitted study. It does **not** establish that the classifier learned genre concepts, or that low-energy attribution proves shortcut use.

Start with [reproducibility and known limitations](docs/REPRODUCIBILITY.md), the [revision experiment plan](docs/EXPERIMENT_PLAN.md), and [audit evidence](results/revision_audit/summary.json). The new [exploratory filtering probe](results/filter_probe/summary.json) covers all 199 historical evaluation recordings. Historical results and newly recomputed measurements are labeled separately. Planned experiments are not reported as completed.

## Setup

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync --frozen
uv run python -m unittest discover -s tests -v
```

Use the checked-in lockfile; do not run `uv add` to reconstruct the environment. Obtain GTZAN independently under its applicable terms and set `GTZAN_ROOT=/path/to/genres_original`. Audio is not redistributed. A model/artifact download is not yet supplied, so source installation alone does not reproduce the historical findings.

## Audit existing trusted artifacts

```sh
uv run python -m gtzan_probe.revision_audit --output results/my_audit
```

The required local artifact names and provenance limitations are listed in [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). This command recalculates from existing files without replacing them and records their SHA-256 hashes. It is not a training replication.

## Repository layout

- `src/gtzan_probe/`: historical pipeline plus separate revision audit/experiments.
- `tests/`: numerical/protocol checks for new revision code.
- `results/revision_audit/`: small sanitized tables and provenance for the audit.
- `docs/`: reproducibility limitations and prospective experiment design.
- `notebooks/`: output-free entry points; read their warnings before running.
- `scripts/export_public.py`: creates an inspectable source/evidence ZIP without Git history.

The historical modules (`prepare_dataset`, `train`, `evaluate`, `explain_shap`, `explain_lime`, `validate`, `explain_attention`, `generate_figures`, `interventional_validation`) preserve the original workflow for inspection. They can overwrite local artifacts and do not implement the proposed corrected experimental design. Preserve old outputs and use a separate checkout for reruns.

Manuscripts, correspondence, audio, model binaries and large generated artifacts stay local. Licensed under the MIT License — see [LICENSE](LICENSE). Audio files are not redistributed; obtain GTZAN independently under its applicable terms. See the provenance document before reusing findings.
