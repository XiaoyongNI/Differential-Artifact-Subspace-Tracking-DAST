# Differentiation and artifact–neural spatial overlap

Run from the repository root:

```bash
python tests/plot_differentiation_subspace_separation.py --dataset all
```

Select `swec_synthetic`, `swec_pbs`, `eraasr`, or `dictionary` individually. Defaults
are seed 42, 95% energy ranks, both normalization modes, nonoverlapping 100-ms
SWEC windows, all configured patients, up to six SWEC clips per patient, and all
14 PBS configurations. For a small run:

```bash
python tests/plot_differentiation_subspace_separation.py \
  --dataset swec_synthetic --max-patients 2 --max-trials 2 \
  --output-dir /tmp/subspace-check
```

Patient paths, rates, clip lengths and generator settings come from the existing
`experiments/seizure_preservation/config.json`. Use `--patient-config`,
`--data-root`, or `--artifact-source` to override them. `--root` controls the
existing PBS/ERAASR/dictionary loaders. No neural reference is synthesized or
inferred from contaminated stimulation data.

## Ground-truth protocol

The existing patient loader, recording-level split and window extractor select
up to six two-second clips per patient from the test partition, with seeded
class-stratified selection. A class shortage can yield fewer clips; sampling
manifests preserve labels and native sample positions. The split is a reusable
selection convention; this experiment does not train a model. `--max-trials`
applies to SWEC clips, while `--trials-per-block` controls the real recordings.

The existing synthetic generator produces an artifact once for each clip. Its
fixed unit amplitude reference is sufficient because scalar magnitude does not
affect a spatial subspace. No clean signal is used to optimize artifact
parameters. PBS artifacts use the stable-subspace experiment's isolated PBS
loader (existing 0.1–1.0-second crop and resampling), then are resampled to the
patient analysis rate and paired with the first eight neural channels. PBS
traces are reused across patients without counting those copies as independent
artifact recordings.

The configured analysis rate defaults to 1024 Hz. Neural recordings are
resampled from their native rate with the existing extractor, synthetic
artifacts are generated at 10240 Hz and downsampled with the existing adapter,
and PBS is resampled to 1024 Hz. This protocol differs from the unidentified
single-recording precomputed archive in `load_swec_synthetic`; it enables actual
patient-level inference. Upsampling does not restore unrecorded neural
bandwidth. At 1024 Hz a requested 100-ms window rounds to 102 samples (99.61 ms).

Both conditions use exactly the same segments and channels. `derivative_time`
produces T−1 samples from each T-sample segment. The undifferentiated condition
retains all T samples. Each channel is centered within its segment after the
condition transform. Tails, nonfinite inputs, zero-energy matrices and invalid
SVDs are excluded in matched pairs, with reasons saved.

Two coordinate choices are reported:

- `none`: centered but unscaled voltage.
- `global_per_channel`: the same centered data divided by neural-reference
  channel standard deviations fitted once across the full parent clip (or full
  prestimulation recording for real data). The scale is identical for artifact
  and neural data, and fixed before differentiation. Near-constant neural
  channels reject the normalized comparison rather than changing channels.

This uses `normalize_channels` to obtain the fixed scales. It intentionally
avoids separately standardizing artifact and neural matrices, which would put
their bases in different coordinates. Both modes use mean centering, unlike
the stable-subspace script's *uncentered* `none` mode. The primary figure uses
centered, unscaled data; normalized results are a sensitivity analysis.

The stable-subspace SVD and cumulative-energy helpers select each component's
smallest rank reaching the threshold. The equal-rank overlap helper is reused;
the unequal-rank case keeps **all selected columns** and divides squared
Frobenius projection overlap by the smaller rank. The CSV also includes the
unnormalized numerator. These are two distinct uses of “normalization”:
channel scaling is optional, while the primary overlap metric always includes
rank normalization.

## Real-recording estimates

ERAASR and dictionary data use explicitly provided prestimulation baselines.
The existing stimulation-locked estimator forms pulse averages in disjoint
trial groups, with recorded detected landmarks. The artifact matrix is a short
average pulse epoch, not a 100-ms isolated-artifact recording; the neural
matrix uses the corresponding trials' rest intervals. Each continuous rest
trial is centered/differentiated separately before concatenation, so trial
boundaries cannot introduce false transients.

These estimates may retain evoked neural responses and are **not ground
truth**. Their sampling differs from the SWEC window analysis and their figures
are separate. Each named recording is one independent unit. Its blocks are
descriptive observations, so a single recording yields no Wilcoxon p-value.
Different dictionary files are not assumed to represent independent animals.
Datasets without usable references or pulse estimates are skipped explicitly.

## Inference and outputs

Synthetic overlap is averaged across windows within patient. PBS overlap is
first averaged within patient/configuration, then equally across patients for
each configuration. This avoids weighting patients by their valid-window
counts. PBS inference assumes independent configurations and is conditional
on the sampled neural cohort; it does not establish biological independence
between configurations.

Two-sided paired Wilcoxon tests use these independent-unit averages, with
differences rounded to 12 decimal places for numerical ties. All-zero
differences give p=1; fewer than two units give an unavailable p-value. Holm
correction spans all testable datasets and both normalization modes. Negative
differences mean differentiation reduced overlap. Results are not constrained
to support that hypothesis.

Outputs default to `figures/derivative/`:

- `artifact_neural_subspace_overlap.pdf` and PNG: unscaled ground-truth results,
  independent-unit pairs and mean ± sample SD, y-axis [0, 1].
- `artifact_neural_subspace_overlap_global_per_channel.*`: scaled sensitivity.
- `exploratory_subspace_overlap*.pdf/png`: individual real-recording estimates.
- `window_overlaps.csv`: ranks, both overlap measures, sample counts and keys.
- `independent_unit_overlaps.csv`: values used for inference and the main plots.
- `overlap_summary.csv`: independent-unit mean, SD, median, IQR and counts;
  also includes window counts and window-weighted means.
- `paired_statistics.csv`: sample counts, mean/median changes, raw and corrected
  p-values, and status.
- `excluded_windows.csv`, `skipped_datasets.csv`, `real_recording_events.csv`:
  exclusion and extraction audit trails (events exist only when estimated).
- `singular_value_diagnostic_*.pdf/png/json`: four relative singular spectra
  for the first valid SWEC pair and its identity. Spectra are normalized by
  their leading singular value, because the reused SVD rescales each input.
- `analysis_metadata.json` and `sampling/`: resolved settings, generator
  parameters, source provenance and selected patient intervals.

Figures use the stable-subspace plot's 3.5 × 3.0-inch size, serif/STIX fonts,
13-pt labels, 1.4-pt lines, 9.5-pt frameless legends, project palette and
600-dpi PNG/embedded-font PDF export.

Run the targeted analytical and aggregation tests with:

```bash
python -m pytest tests/test_differentiation_subspace_separation.py -q
```
