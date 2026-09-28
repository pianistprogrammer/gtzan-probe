# Revision experiment plan

Status: prospective protocol except for the explicitly completed exploratory filter probe below. Freeze this plan and all analysis choices before new runs. Deadline in the decision letter: 23 November 2026.

## Completed exploratory probe (not a confirmatory replication)

The new `filter_probe` run processes all 199 historical evaluation tracks (4,179 condition rows), with zero failures and all 199 baseline predictions matching the archived decisions. Results and exact filter coefficients are in `results/filter_probe/`. Low-frequency attenuation reduces hip-hop probability, but the higher-frequency comparison gives a slightly larger mean reduction; two of 13 initially correct disco predictions change to hip-hop under +12 dB low-component boosting. The filters are not matched for perturbation magnitude. RMS rescaling is largely canceled by the normalization. Group-aware uncertainty, retraining and independent test data remain pending. The plan below describes the stronger study still needed.

## Priority 1: establish valid data and evaluation units

- Replace the eight-item hand-maintained exclusion list with an attributed, versioned GTZAN fault/artist/recording manifest based on Sturm's audit. Distinguish exact duplicate, repeated recording, artist repetition, label dispute, and decode failure. Record source and uncertainty per item; unknown artists must not be invented.
- Verify PCM fingerprints for exact duplicates, then audit recording groups. Keep each connected artist/recording/duplicate group in only one partition. Publish filenames, group IDs, exclusions, checksums, split seed and generator. Provide class counts and unresolved metadata rates. Group-aware train/validation/test partitioning may require balance compromises.
- Use validation only for checkpoint selection; reserve test until the protocol is fixed. The historical 80/20 partition reuses validation as test and cannot retrospectively become an independent test set.
- Train at least three declared seeds on the same fixed partitions. Report each run, failures and stopping rules. Separate training-seed uncertainty from track/artist uncertainty. Do not count crops as independent tracks.

## Priority 2: interventions that can falsify the narrative

Hypotheses are motivated by the existing exploratory results, not preregistered discoveries. On all eligible evaluation tracks, test whether reducing low-frequency energy reduces hip-hop class probability, and whether increasing it changes disco decisions toward hip-hop. Include initially incorrect predictions and separately report an initially-correct subgroup.

- Filter waveforms, not normalized mel pixels, using a documented response below 250 Hz and fixed gains -12, -6, 0, +6, +12 dB. Include a sham operation and a matched high-frequency comparison (2-6 kHz). Apply the identical preprocessing afterward.
- Record original/perturbed probabilities and logits for a fixed target, all class predictions, transition counts, RMS/peak changes, exact coefficients, track checksums, and crop offsets. Include both unmatched and RMS-matched variants to distinguish spectral from overall level effects. Avoid clipping; record any playback scaling separately from model input.
- Report paired effects and confidence intervals at the independent track or artist-group level; analyze dose response. Do not extrapolate from ten examples to genres. Filtering changes several musical and production properties and does not identify a cultural genre concept.
- Compare attribution-guided deletion with random and energy-ranked deletion at equal area. Test both SHAP and LIME, multiple LIME seeds, alternative partitions, and zero/training-mean/noise baselines. Keep targets, units (logit versus probability), crops and model hashes identical. Report negative changes without clipping them away.

## Priority 3: test whether explanations add to input spectra

- Compute mean normalized log-mel and linear-power profiles using the same tracks/crops as the explanation samples, and separately the full evaluation set. Normalized log-mel means are not energy fractions.
- Compare signed and absolute attribution summaries separately to these baselines. Report 128-bin curves as well as four historical bands and five physical bands (0-60, 60-250, 250-2000, 2000-6000, 6000-11025 Hz). Document filter-center assignment and band sizes; a five-band rerun is a new analysis.
- Quantify within-track versus between-track and within-method versus between-method variability. Publish local surrogate fidelity and SHAP additivity residuals. Run weight-randomization sanity checks. Define ties explicitly when computing top-k overlap.

## Priority 4: architecture and dataset robustness requested by the editor

- Repeat matched SHAP/LIME and intervention protocols on MusicCNN and a materially different model (for example a residual CNN) trained on identical splits. AST attention alone is not this experiment. Report all seeds and classification failures.
- Apply the same analysis to FMA-small and FMA-medium using artist-aware partitions and documented available labels. FMA-small is nested inside FMA-medium: it is not an independent external test of a medium-trained model unless overlaps are excluded.
- Publish an explicit label crosswalk with original FMA IDs/titles, ambiguity and exclusions. Similar label strings do not establish equivalent annotation concepts. Evaluate transfer in both directions only on a justified shared task, separately from within-corpus performance; report unmatched classes and domain shifts.
- The saved FMA band-profile rank correlations are input statistics, not cross-dataset classifier or explanation validation. Do not present them as fulfilled robustness experiments.

## Resubmission gates

All completed claims must link to immutable input manifests, model/config hashes, raw per-track results, and a script that builds each reported table/figure. Resolve the saved-checkpoint/attribution provenance conflict; regenerate affected results if needed. Reconcile the exact submitted PDF with the review. Recheck bibliography against primary sources. Provide accessible, anonymized review materials if required by the journal, plus a separately prepared public release. Never describe planned work as completed in the response letter.
