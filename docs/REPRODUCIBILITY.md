# Reproducibility and provenance

## What the historical code actually did

- A partial eight-entry exclusion list was described as a GTZAN audit. It is not Sturm's complete fault inventory and it is not an artist filter. The literal legacy `clean` flag means only “not on that list / not flagged by the local loader.”
- `train.py` partitions that metadata with StratifiedShuffleSplit(test_size=0.2, random_state=42). Validation selects the best checkpoint and the same partition is evaluated. Training RNG and LIME RNG seeds were not recorded; the split seed does not imply seeded training.
- Five random 3-second crops per training track per epoch are correlated observations. Evaluation and explanation use the opening 3 seconds. `train.py` additionally prints multi-crop accuracy; `evaluate.py` saves single-crop predictions. These results must not be conflated.
- `explain_shap.py` and `explain_lime.py` choose the first correctly classified row per genre: ten examples, not all test clips. SHAP backgrounds use the first five eligible metadata rows per genre, not training-only rows. Audit output lists reconstructed validation overlap; actual historical background IDs were not stored.
- SHAP explains logits; LIME's prediction function returns probabilities. Their attribution values and perturbation geometry have different meanings. The LIME default is 500 samples, although the old paper stated 1,000. Historical invocation logs are absent.
- Four mel slices are used: [0:10], [10:30], [30:80], [80:128]. The old names sub-bass/bass-rhythm/harmony/timbre and the five-band manuscript description were misleading. Historical code retains its keys to preserve compatibility; use B1-B4 and the audited filter-center/support table when interpreting it.
- “Musicological alignment” is a correlation against hand-written weights without supporting genre-specific evidence. It is not validation. Low-energy attribution is not a silence/artifact detector. Edge bins are first/last five time columns, not first/last ten percent.
- `regen_all_figures.py` replaces a supposed alignment-score graphic with an attribution heatmap under the same filename. That legacy naming collision is now fixed for future runs; the heatmap is saved as `05_legacy_band_attribution_heatmap`. The revision uses separate artifact names and captions.
- FMA processing averages normalized log-mel input values, then correlates four ranks. It does not train/test classifiers across corpora or apply SHAP/LIME to FMA.

- The legacy AST gallery takes a square reshape of the attention vector and can truncate patches, rather than reconstructing the model's actual frequency/time patch grid. It must not support frequency-specific interpretations; the revised manuscript removes it.

## Reproduce the revision audit

Install Python 3.13 and uv, then run `uv sync --frozen`. On this workstation the old .venv interpreter was unusable; the completed audit used the existing dependency packages with a working Python 3.13.13 interpreter; a fresh `/tmp/gtzan-revision-env` was then installed for tests and the filtering probe. An attempted `.venv-revision` install on the external volume was interrupted because it was slow. All recorded scientific dependency versions match the original lockfile. This patch version is disclosed in the run metadata.

Trusted local inputs required in data/: gtzan_metadata.csv, test_predictions.csv, shap_values.pkl, lime_results.pkl, faithfulness_results.csv. Required in models/: cnn_gtzan_best.pt, label_encoder.pkl. Check expected hashes in results/revision_audit/input_checksums.json before comparing. Pickles must only be loaded from trusted sources.

```
uv run python -m gtzan_probe.revision_audit --output results/my_audit
uv run python -m unittest discover -s tests -v
```

The audit writes a new directory and refuses to overwrite it. It exports relative identifiers, input hashes, exact legacy split membership, selected examples, recalculated summaries, and saved-versus-current-checkpoint discrepancies. It does not repair missing historical lineage. Published evidence tables are derived data, not independent replication. Model and explanation inputs still need a licensed, documented artifact release; the source ZIP alone cannot reproduce training or every historical number.

## Completed exploratory waveform probe

```
uv run python -m gtzan_probe.filter_probe --output results/my_filter_probe
uv run python scripts/summarize_filter_probe.py results/my_filter_probe
uv run python scripts/revision_figures.py --audit results/revision_audit --output figures/revision
```

The completed run contains 199 recordings, 4,179 condition rows, zero failures, all ten genre labels and all class logits/probabilities. No audio is modified on disk. It discloses exact filter coefficients, checkpoint and audio hashes, RMS/peak changes, and limitations. Initially-correct subgroup sizes are 19 hip-hop and 13 disco recordings. There is no independent-test or artist-disjoint inference claim, and no naive population confidence interval. Per-recording raw output is retained even when it contradicts the original spectral narrative.

## Environment packaging correction

The old lockfile recorded the project as virtual despite pyproject.toml defining a buildable package. Its root entry was changed to editable; dependency versions are unchanged. `uv sync --frozen` now installs the project itself. The six numerical/protocol tests check band support, exact top-k behavior with ties, non-finite inputs, fail-fast decoding and targeted filtering/sham identity.

## New runs

Use a separate checkout and output root; do not overwrite historical artifacts. Follow EXPERIMENT_PLAN.md. The legacy training commands are preserved for inspection and are not a corrected artist-disjoint protocol. No completed multi-seed, artist-disjoint, FMA-medium, or second-architecture explanation replication is claimed.

## Public release

`python scripts/export_public.py` creates a source/evidence ZIP with no .git history, manuscript, correspondence, audio, model checkpoints, or pickle caches. Inspect EXPORT_SHA256.json inside it. The source code is released under the MIT License. The author must still verify distribution rights for model checkpoints and result artifacts before claiming a complete open-source/model/data release; audio is not redistributed.

Paper/, data/ and figures/ have been removed from the current Git index but are kept locally. Earlier Git commits still contain them. The configured GitHub remote was already public when checked. No history rewrite, commit, push, repository visibility change, or journal submission was performed. A future publication of this checkout must address existing history; .gitignore alone does not remove published files.
