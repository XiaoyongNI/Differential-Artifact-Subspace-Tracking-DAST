# Online PASTd identification of the artifact direction

```bash
python tests/plot_pastd_u1_convergence.py --dataset all
```

The experiment measures whether differentiation improves recursive estimation
of the dominant artifact spatial direction from the contaminated signal.
It reuses `run_datasets` from the energy-ratio experiment, including the same
patient loading, recording-group splits, clip selection, synthetic generator,
training-reference amplitude calibration and PBS scaling. Existing source
files are not modified.

## Matched tracking protocol

For each continuous clean/artifact pair, construct `X = neural + artifact`.
Initialize two independent instances of the existing `algorithms.PASTd` with
rank one, beta 0.999, `U0 = e1`, and `d0 = 1`. Both states reset once per clip.
There are no resets at the preceding energy experiment's 100-ms boundaries.

The trackers receive:

```text
No differentiation:       X[:, 1:]
First-order difference:   derivative_time(X)
```

Both therefore have T−1 updates and the same latest sample at each update.
No fictitious leading zero difference is inserted. The stored curve at sample
zero is the shared initialization; sample `i > 0` is the estimate immediately
after updating at original sample `i`. Time is `i / fs` from clip start, not
from stimulation onset. Synthetic onset times vary according to the existing
generator and remain in the metadata. PBS crops may begin during stimulation.

No extra filtering, channel normalization, centering, warm start or truth-based
initialization is introduced. `--beta`, `--initial-power` and
`--initialization random` allow explicitly labeled sensitivity analyses. A
random initial vector uses a recorded seed and is shared by both conditions.
The initial power is identical in both arms; absolute sample amplitude can
affect PASTd startup, so the physical mixing scale is preserved.

The tracker is online with respect to the prepared sample sequence. Existing
clip extraction and resampling are offline preparation; this experiment does
not establish end-to-end hardware latency or a causal resampling pipeline.

## Ground truth and similarity

The primary reference is the leading left singular vector of the **original
artifact** `A[:, 1:]`, computed with the stable-subspace experiment's spatial
SVD helper. Uncentered SVD matches the second moment tracked by raw PASTd.
The same reference is used for both conditions:

```text
similarity(t) = abs(dot(tracked_u1(t), artifact_u1))
```

Vectors are normalized, so similarity lies in [0, 1] and is invariant to the
arbitrary SVD sign. Full-clip artifact SVD is used only for evaluation, never
for tracker updates or initialization. Tests explicitly verify that future
samples cannot change earlier PASTd estimates.

Differentiation may also change the artifact's own dominant spatial direction,
especially for a multi-component PBS artifact. To distinguish reference
rotation from tracking error, save:

- Original versus differentiated artifact-u1 absolute cosine.
- Leading energy fractions and relative eigenvalue gaps.
- Secondary traces using the raw artifact reference for raw tracking and the
  differentiated artifact reference for differential tracking. These traces
  have **different targets**, so they supplement the shared-reference result.

A zero or numerically tied original artifact direction excludes the whole
pair. A missing or tied differentiated reference marks the secondary trace
unavailable while retaining a valid primary comparison. ERAASR and dictionary
recordings are excluded because isolated artifact ground truth is unavailable.

## Convergence measures and inference

The default evaluation horizon and early plot extent are both 200 ms, set
before inspecting outcomes. The script also saves full-clip plots and traces.
Horizons round to samples at the configured rate (default: 1024 Hz).

- **Early mean similarity:** trapezoidal area under the similarity curve,
  divided by the actual early horizon. Higher means better early identification.
- **Sustained convergence:** similarity at least 0.95 continuously for at least
  20 ms. Both the start and the **confirmation** of the first qualifying run
  are saved. Confirmation determines convergence time. A transient crossing
  does not qualify; later regressions are visible in the full trace.
- **Restricted convergence delay:** confirmation time if confirmed within the
  common early horizon, otherwise the horizon. Non-converged observations
  retain their event/censoring flags; the cap is not a successful convergence
  time. Lower restricted delay is better. The corresponding convergence rate
  is reported alongside it. Full-clip event times are also saved, with missing
  times for non-converged clips.

Trials are averaged within patient/configuration. Synthetic independent units
are patients. PBS first averages patients equally within each configuration;
units are configurations, not reused copies of a PBS recording. Configuration
independence is assumed, and inference is conditional on the sampled neural
cohort. This matches the previous energy-ratio experiment.

Paired two-sided Wilcoxon tests compare early mean similarity and restricted
delay across independent units. Holm correction covers both metrics and both
datasets. Differences are rounded to 12 decimal places for numerical ties;
all-zero differences give p=1 and fewer than two units give unavailable p.
The experiment reports the observed direction, including slower convergence
or poorer alignment if encountered.

Plots show mean ± sample SD across independent units, using the existing
3.5 × 3.0-inch TBME style, 13-pt serif/STIX labels, 1.4-pt lines and project
colors. The dotted line is the similarity threshold. Full curves stop at the
shortest included clip in each dataset so the contributing cohort does not
change with time. Untruncated individual traces are retained in the archive.

## Outputs

Default directory: `figures/derivative/pastd_u1_convergence/`.

- `pastd_u1_convergence_swec_synthetic.pdf/png` and
  `pastd_u1_convergence_swec_pbs.pdf/png`: primary early-time plots.
- Corresponding `_full` plots: full common follow-up.
- Corresponding `_condition_matched` plots: secondary reference diagnostic.
- `clip_metrics.csv`: matched trial metrics, confirmation times, event flags.
- `independent_unit_metrics.csv`, `convergence_summary.csv`,
  `paired_statistics.csv`: aggregation and paired inference.
- `curve_summary.csv`: per-sample mean and SD of each curve and reference type.
- `artifact_reference_diagnostics.csv`: reference rotation and identifiability.
- `trace_index.csv` and `tracking_traces.npz`: full-resolution paired traces.
  Each trace key has `_primary` and `_condition_matched` arrays of shape
  `(2, T)`, with conditions `[none, derivative]`, plus `_references` with rows
  `[original, derivative]`. Sample zero is initialization, not an update.
- `representative_<dataset>.npz`: the first valid pair's actual tracked vectors
  `(2, T, channels)`, shared reference, similarities, sample and time axes.
- `analysis_metadata.json`, `sampling/`, `skipped_datasets.csv`,
  `excluded_clips.csv`: configuration, source parameters, selection manifests
  and explicit exclusions.

Dataset/path/amplitude controls match the energy script: `--dataset`,
`--patient-config`, `--data-root`, `--artifact-source`, `--root`,
`--max-patients`, `--max-trials`, `--calibration-trials`, `--seed`,
`--pbs-configurations`, and `--pbs-scale`. Tracking controls are `--beta`,
`--initial-power`, `--initialization`, `--evaluation-ms`, `--plot-ms`,
`--similarity-threshold` and `--hold-ms`. Use separate `--output-dir` values
for sensitivity analyses to avoid mixing artifacts from different runs.

Quick check:

```bash
python tests/plot_pastd_u1_convergence.py --dataset all \
  --max-patients 2 --max-trials 2 --calibration-trials 2 \
  --pbs-configurations 0 1 --output-dir /tmp/pastd-check
python -m pytest tests/test_pastd_u1_convergence.py -q
```
