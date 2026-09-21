# Seizure information preservation on SWEC-ETHZ

This experiment trains **separate clean-only EEGNet, GRU and ChronoNet decoders
for each patient**, freezes them, then compares matched clean, contaminated and
artifact-cancelled test windows. It reads all 18 patient HDF5 exports directly;
the old root-level/windowed NPY files are **not** used because their training/test
windows overlap and do not contain recording provenance.

Run from the project root with Python 3.10+ and NumPy, SciPy, h5py, PyTorch,
scikit-learn, matplotlib and threadpoolctl installed. The current environment
already supplies these dependencies. Tests additionally use pytest.

```bash
# Inspect all 18 patient splits; no signal extraction or model training.
python -m experiments.seizure_preservation.run --plan-only \
  --output-dir results/seizure_preservation_plan

# Real ID04 data, all three decoders and eight conditions, tiny one-epoch check.
# These are integration-test outputs, NOT scientific performance estimates.
python -m experiments.seizure_preservation.run --smoke --device cpu

# Full experiment: 18 patients, clean validation checkpoint/LR selection.
python -m experiments.seizure_preservation.run --device cuda:0

# Resume after interruption; finished patients are skipped, partial ones restart.
python -m experiments.seizure_preservation.run --device cuda:0 --resume

python -m pytest -q tests/test_downstream_task.py
```

To use **100 windows total per patient** (normally 60 train / 20 validation /
20 test), including LRR and all three decoders:

```bash
python -m experiments.seizure_preservation.run --device cuda:0 \
  --samples-per-patient 100 \
  --output-dir results/seizure_preservation_n100
```

Replace 100 with any integer >=6. A sample here is one multichannel window
(default two seconds), not an individual time point. `samples_per_patient` can
also be set in config.json; `null` retains the original per-class caps.
This option overrides `max_windows_per_class`, allocates the total budget using
`split_ratios`, and targets equal seizure/non-seizure counts within each split.
Scarce-class quotas are filled from the other class without replacement; if a
whole split has too few eligible windows, fewer are returned. Negative-only
validation remains negative-only. Recording groups are still split first;
only selected window signals are read and upsampled. Selection is reproducible,
saved in manifests, and shared across all decoders/methods. LRR's separate
synthetic-only calibration windows are not included in this neural-window budget.
Balanced subsampling changes test prevalence and very small test sets give
coarse metrics; it is useful for reducing computation.

All settings are in [config.json](config.json). CLI overrides and smoke overrides
are saved in the exact run configuration. Choose a new output directory when
changing configuration or code; resume verifies both configuration and source
hashes. The output directory must be dedicated to this experiment.

## Splits, sampling, and labels

- `load_patient_data` reads `IDxx/EEG` (channels × samples), `fs_target`, and
  seizure onset/offset seconds converted to half-open native sample intervals.
  The local HDF5 signals are already downsampled to **256 Hz**. They are the
  artifact-free reference in this experiment, not the original 512/1024 Hz
  acquisition samples; upsampling cannot restore lost bandwidth.
- `split_patient_data` assigns whole recording groups approximately 60/20/20
  using a patient-specific fixed seed. The export lacks per-file lengths.
  Hourly offsets are **inferred** from the sequential `IDxx_1h.mat`, etc. names;
  filename sequence, file count and total duration are validated. The last
  recording can be shorter. This assumption is recorded in each split manifest.
- Recordings touched by one seizure plus the configured 60-second margin are
  united into one group before assignment. Thus a seizure crossing an hourly
  boundary cannot enter multiple splits. Groups are stratified by seizure
  presence, not randomly split by windows. The 60/20/20 target is by group
  count; seizure counts/durations can differ substantially when seizures cluster.
- ID01, ID02, ID11, ID15 and ID17 have only two independent seizure groups.
  It is impossible to have positives in three disjoint splits. The explicit
  default is one positive group for training, one for testing, and independent
  seizure-free recordings for validation. Their clean validation loss cannot
  measure seizure sensitivity. These cases are flagged; set
  `two_seizure_policy="error"` to refuse them instead. No seizure is subdivided
  to manufacture a third positive split.
- `extract_windows` makes two-second, nonoverlapping windows by default, wholly
  inside assigned groups with a two-second boundary guard. Any seizure overlap
  gives label 1; otherwise label 0. All split windows use the same length,
  stride and label rule. Exact native starts/stops and group IDs are saved.
- Each selected window is independently polyphase-resampled **256 → 1024 Hz**.
  Filtering never reads across split boundaries. Full recordings stay on disk;
  extracted arrays use memory-mapped NPY files.
- Default caps retain all positive windows and seed-subsample up to **2,000
  negatives per split** for tractability. This changes prevalence: accuracy,
  weighted F1 and kappa describe the selected test sample, not continuous
  deployment prevalence. Set both class caps to `null` to retain every eligible
  window, or choose a preregistered cap. Candidate and selected counts are saved.
  Sampling happens only after group splitting and never mixes splits.

## Reuse and decoder training

- `data_loading.Dataset`, `pipeline.PipelineConfig`, `pipeline.run_pipeline`,
  and the shared filtering/normalization/tracking code provide cancellation.
- The **existing EEGNet** is loaded from
  `/home/jalali/seizure_detection/models/eegnet.py`. Its lazy spatial convolution
  and classifier are materialized before the optimizer is built. Its existing
  temporal kernels are retained, now acting on the 1024 Hz grid.
- The **existing synthetic generator** is loaded from
  `/home/jalali/seizure_detection/utils/artifact_generator.py`.
  Both source paths are configurable, hashed, and copied into the run snapshot.
  Missing sources cause an explicit error, not a silent replacement.
- No GRU or ChronoNet decoder exists in TBME_clean_code. `models.py` adds a
  stacked unidirectional GRU with final-state readout and a ChronoNet adaptation:
  three inception blocks (parallel kernels 2/4/8, stride 2), then four densely
  connected GRUs and a last-state binary readout. Variable channel count,
  two-second inputs, and dropout are explicit adaptations of
  [Roy et al., ChronoNet](https://arxiv.org/abs/1802.00308).
- Per-channel mean/SD are fitted once on **clean training data only**. Those
  exact statistics transform validation and every test condition. `y_clean`
  in the shared Dataset interface is reserved for reference waveforms; binary
  decoder labels are stored separately.
- AdamW and class-weighted cross-entropy use training class counts only.
  Learning-rate candidates, early stopping and best checkpoint selection use
  minimum **clean validation loss**. No test condition influences training,
  early stopping, thresholds, or method parameters. Prediction is fixed argmax.
- The selected decoder is set to eval mode with all parameter gradients disabled;
  evaluation asserts this. BatchNorm statistics remain frozen. Checkpoints
  include architecture config, normalization, seed, epoch and selection loss.
- DAST with derivative order one omits the first sample. To avoid inventing a
  replacement, every decoder is trained/validated/evaluated on samples `[1:]`
  of the same 2048-sample windows (2047 samples, still at 1024 Hz). Other
  derivative orders use the corresponding shared offset. Labels are attached
  to the original fixed windows; the exact offset is saved with predictions.

## Matched artifacts and cancellation methods

Test artifacts and independent LRR calibration artifacts are generated **only after all three clean decoders are selected**.
The reused generator operates on zeros at 10240 Hz, and its artifact is
anti-aliased to 1024 Hz before addition to the upsampled clean test signal.
Stimulation channel, rate, phase, current, pulse frequency, decay realizations,
and amplitude are sampled from the configured held-out seed namespace. Artifact
RMS is scaled to a random 2–8 times the patient’s **training** RMS by default;
there is no test-label or test-window-energy scaling. No artifacts are applied to decoder training/validation data. Removal hyperparameters
are fixed; LRR alone requires fitting weights on the independent synthetic-only
calibration described below.

Every window is generated once. All methods receive a copy of the **same**
immutable contaminated array, checked by SHA-256 after every call. Neither the
clean test reference nor labels are passed to cancellation. Parameters,
seeds, triggers and clean/artifact/contaminated hashes are saved per window.

| Output name | Implementation |
|---|---|
| Clean | Held-out clean reference |
| Contaminated | Exactly `clean + artifact` |
| DAST | Existing derivative-driven PASTd pipeline, default rank 1; per-window state reset, normalized tracking and denormalized output |
| SVD | Existing `benchmarks.methods.window_svd`, centered spatial SVD in merged known pulse windows |
| ERAASR | New explicit **single-trial Python adaptation**: sequential channel then pulse PCA regression, with continuity reconstruction |
| LRR | Existing `fit_lrr_trials` and fixed-weight, sample-by-sample `lrr` benchmark |
| linear_interpolation | Existing benchmark implementation using known pulse markers |
| template_subtraction | Existing backward template implementation using the previous three contaminated pulse epochs |

LRR uses 16 independent synthetic-artifact-only windows per patient by default
(`cancellation.lrr.calibration_windows`). Calibration contains zero neural signal;
its amplitudes use training RMS and its seeds use a separate `lrr_calibration`
namespace. It never uses clean, contaminated, or labelled test windows for fitting.
The existing `fit_lrr_trials` selects known pulse periods, fits the channel-wise
regression weights once, and freezes them. All matched test samples then pass
through the existing `OnlineLRR.process_sample` path with those fixed weights.
This baseline assumes access to a synthetic-only artifact calibration source;
it is not a calibration-free method. Decoder training remains clean-only.
`IDxx/lrr_weights.npy` and `IDxx/lrr_calibration.json` save weights, parameters,
seeds, pulse-window settings, and hashes. LRR is included automatically in
restoration metrics, patient aggregates, figures, and Holm-corrected DAST-versus-LRR
comparisons. Use a new output directory when adding LRR to an older run.

The existing `window_ica`, `pulse`/`low_rank_tv` are optional entries in `methods`.
Parameters are fixed before test evaluation. Known synthetic pulse markers are
available to windowed methods, representing an oracle-timing comparison.

**ERAASR scope:** this repository contained ERAASR example data but no algorithm.
The adaptation follows the channel/pulse stages in the author's
[`cleanArtifactTensor`](https://github.com/djoshea/eraasr/blob/b5af88ae6388ccd57a4c7f28fa2c6af7b944c439/%2BERAASR/cleanArtifactTensor.m)
and [`cleanMatrixViaPCARegression`](https://github.com/djoshea/eraasr/blob/b5af88ae6388ccd57a4c7f28fa2c6af7b944c439/%2BERAASR/cleanMatrixViaPCARegression.m).
Default `pca_only_omitted=false` uses globally fitted PCs with the target and
neighbor coefficients removed before regression, an upstream-supported mode.
`true` refits PCA on the non-omitted columns (slower). The across-trial stage is
explicitly disabled (`nPC_trials=0`) because each window is an independent
realization, not a repeated identical stimulation trial. The adaptation uses
known, integer-period markers, no high-pass filter, no post-stimulation PCR,
and no alignment search. It is **not** a claim of bitwise equivalence to the
complete original MATLAB ERAASR pipeline. Report this variant in publications.

## Outputs and inference

- `results.csv`: patient × decoder × method accuracy, weighted F1, kappa,
  and three unclipped restoration ratios.
- `aggregated_results.csv`: patient mean, **sample SD (`ddof=1`)**, formatted
  mean ± SD, and valid-patient count for every metric. NaNs are excluded with
  their available count exposed; an absent patient is never imputed.
- `paired_statistics.csv`: two-sided paired Wilcoxon signed-rank comparisons
  of DAST against every other non-clean condition, separately for each decoder
  and raw metric. **Patients are the independent units.** Global Holm correction
  spans all decoder × comparator × metric tests. Raw/corrected p-values, pair
  IDs, sample sizes, nonzero differences and mean paired differences are saved.
  Differences are rounded to 12 decimals before ranking; zeros use `wilcox`.
  All-zero differences yield p=1; fewer than two valid patients yield NaN.
- `figures/{accuracy,weighted_f1,kappa,restoration}.{pdf,png}`: paired patient
  points/lines, patient mean ± SD, and unbounded restoration plots. Restoration
  has panels for each metric and decoder, with reference lines at 0 and 1.
- `IDxx/splits.json`, input fingerprints, window manifests and class counts.
- `IDxx/{EEGNet,GRU,ChronoNet}/best.pt`, candidate-best checkpoints, epoch
  histories and clean-validation selection metadata.
- `IDxx/predictions/*.npz`: per-window probabilities, true labels, window IDs,
  sampling rate and common sample offset.
- `IDxx/artifact_parameters.jsonl`, training normalization, exact configuration,
  source snapshots/hashes, environment versions and git working-tree diff.

Restoration is `(method-artifact)/(clean-artifact)`, unchanged for negative gaps
and **never clipped**. Gaps with absolute value at most `restoration_epsilon`,
or nonfinite inputs, yield NaN. Negative clean-artifact gaps should be interpreted
with care: inspect raw metrics as well; the formula does not imply improvement
when the clean reference itself scores below contamination.

Default storage retains exact contaminated arrays and artifacts. Regenerable
clean and cancelled signal caches are removed after successful patient evaluation;
window manifests and predictions are always retained. Enable the respective
`storage` flags to retain all arrays. Several GB per patient may be needed for
cached 1024 Hz signals, and full continuous-window extraction is much larger.

Use the full run for scientific results. A smoke run exercises all paths but has
one epoch, reduced architectures, 16 training and 8 test windows on one patient;
its saved config and completion marker explicitly identify it as a smoke test.
