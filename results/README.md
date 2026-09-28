# Evidence from the revision

These tables are derived from trusted local historical artifacts and one new exploratory intervention. They are not an independent reproduction of model training.

- `revision_audit/summary.json`: actual sample counts, reconstructed partition checks, environment versions and provenance gaps.
- `revision_audit/input_checksums.json`: SHA-256 of the local model, label encoder, attribution pickles and historical CSV inputs.
- `revision_audit/legacy_split.csv`: filename-level training/validation membership reconstructed from the original algorithm. Validation was also used for reported evaluation.
- `revision_audit/band_definitions.csv`: precise indices, filter-center ranges and overlapping supports for B1-B4.
- `revision_audit/band_profiles.csv`: same-excerpt input, absolute SHAP and absolute LIME group means, one excerpt per label.
- `revision_audit/agreement.csv`: signed and absolute agreement, explicit stable-tie top-k overlap and a percentile-threshold sensitivity variant. These are not interchangeable scores.
- `revision_audit/faithfulness_comparison.csv`: all 50 saved and recomputed masking observations. Max discrepancy is approximately 3.7e-7.
- `filter_probe/config.json`: exact filter coefficients and hashes for the frozen checkpoint/source-prediction file and experiment code.
- `filter_probe/per_track.csv`: all 4,179 output rows (199 tracks × 21 conditions), with logits/probabilities for every class. Changes are **perturbed minus original**; the legacy masking table uses the opposite sign.
- `filter_probe/audio_manifest.csv`: recording filenames and file hashes, without local absolute paths or audio content.
- `filter_probe/failures.json`: empty in the completed run; failures must never be silently replaced with fabricated silence.
- `filter_probe/subgroup_summary.csv`: every genre, intervention and initial-correctness subgroup, including results not highlighted in the manuscript.

The fixed-target sham changes are below 1e-6. Baseline decisions match all 199 archived predictions. The low- and high-frequency filters are not matched for bandwidth/perturbation strength; RMS controls are largely canceled by spectrogram normalization. Artist/recording dependence is unresolved, so the summaries are descriptive and contain no claim of independent-track confidence intervals or causal genre identification. The raw attribution/model artifacts are not included; their release remains necessary for complete external verification.
