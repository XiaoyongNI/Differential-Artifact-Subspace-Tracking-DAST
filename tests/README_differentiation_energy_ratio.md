# Artifact-to-neural energy ratio

```bash
python tests/plot_differentiation_energy_ratio.py --dataset all
```

The experiment compares separately known artifact `A` and neural `N` components
within exactly the same channels, clips and nonoverlapping windows:

```text
R_none       = sum(A**2) / sum(N**2)
R_derivative = sum(diff(A)**2) / sum(diff(N)**2)
gain         = R_derivative / R_none
gain_db      = 10 * log10(gain)
```

Sums include all channels and samples. Differentiation calls DAST's existing
`algorithms.derivative_time` along the time axis. Both differentiated components
have T−1 samples, so the ratio of energies also equals the ratio of mean powers.
No derivative is taken across separate windows or trial boundaries. A gain
above one (positive dB change) means differentiation emphasizes artifact energy.
This metric does not itself test spatial orthogonality or recovery accuracy.

The default measures raw energy, including DC, without channel normalization
or extra filtering. `--demean` explicitly removes each channel's mean within
each condition to measure fluctuation energy instead; use a separate output
directory for that sensitivity analysis. Artifact and neural components are
never independently standardized, nor rescaled between conditions. Nonfinite,
zero-energy and incomplete windows are excluded as whole matched pairs and
logged; denominators are not regularized with arbitrary epsilons.

## Data and amplitude conventions

The existing seizure-preservation configuration supplies patient IDs, HDF5
paths, sampling rates, clip lengths, recording-group splits and generator
settings. All configured patients (18 by default) contribute up to six test
clips, selected with the existing seeded class-stratified extractor. Defaults
are seed 42 and 100-ms windows. At 1024 Hz, 100 ms rounds to 102 samples.
Sampling manifests preserve the native clip positions and labels. Upsampling
neural recordings does not add neural bandwidth.

- **SWEC + Synthetic:** reuse the existing artifact generator and its
  training-reference RMS convention. The reference is estimated from up to six
  separately selected training clips per patient, using recording/seizure
  groups disjoint from the test clips. The generator's existing random RMS
  multiplier and all other parameters are recorded. The artifact is generated
  once per test clip and reused unchanged in both conditions.
- **SWEC + PBS:** reuse `load_pbs_artifact` from the stable-subspace experiment,
  resample to the configured patient rate and pair with the first eight neural
  channels. The default amplitude scale is read from `load_swec_pbs` (10000).
  PBS traces are reused across patients, but those copies are not counted as
  independent artifact recordings. Absolute energy ratios depend on this
  mixing scale and neural units; paired gain is invariant to constant scalar
  scaling of either component.
- **ERAASR / Dictionary:** excluded because pulse averages are estimated
  artifacts, not separately available artifact ground truth. The `all` run
  records this reason explicitly.

The synthetic amplitude convention intentionally differs from the preceding
subspace experiment's fixed unit RMS: amplitude does not affect spatial
orientation, but it does affect an absolute energy ratio. The paired energy
gain is unchanged by that constant scaling.

## Aggregation and inference

Window energy ratios are converted to dB and averaged within each
patient/configuration. Their linear equivalents are geometric means, which
avoid arithmetic ratio averages dominated by a few very small denominators.
Synthetic independent units are patients. For PBS, patient dB means are
averaged with equal weights within each configuration; the resulting units are
configurations. PBS inference assumes independent configurations and is
conditional on the sampled neural cohort.

The manuscript table shows median [Q1, Q3] across independent-unit ratios for
both conditions and separately for their paired gains. The median gain need
not equal the ratio of the two marginal medians. Paired two-sided Wilcoxon
tests use unit dB changes, with differences rounded to 12 decimals for numerical
ties. Holm correction covers the evaluated ground-truth datasets. All-zero
changes give p=1; fewer than two units give an unavailable p-value.

## Outputs and options

Default output: `figures/derivative/energy_ratio/`.

- `energy_ratio_table.tex` / `.md`: manuscript-ready comparison table. LaTeX
  uses `booktabs` and `graphicx`.
- `energy_ratio_summary.csv`: median, quartiles, independent-unit/window counts,
  mean and median dB changes, Wilcoxon and Holm p-values.
- `window_energy_ratios.csv`: both conditions, component energies, ratios in
  linear and dB units, and matching sample/channel/window identifiers.
- `paired_energy_gains.csv`: linear and dB gain per matched window.
- `independent_unit_energy_ratios.csv`: values used for the table and inference.
- `artifact_neural_energy_ratio.pdf` / `.png`: paired unit dB ratios and
  median/IQR summaries using the stable-subspace TBME plotting style.
- `excluded_windows.csv`, `skipped_datasets.csv`, `analysis_metadata.json`,
  `sampling/`: exclusion reasons, provenance, resolved configuration, artifact
  parameters and selected calibration/test intervals.

Run either dataset individually with `--dataset swec_synthetic` or
`--dataset swec_pbs`. Other controls include `--max-patients`, `--max-trials`,
`--calibration-trials`, `--window-ms`, `--seed`, `--pbs-configurations`,
`--pbs-scale`, `--demean`, `--patient-config`, `--data-root`,
`--artifact-source`, `--root`, and `--output-dir`. Use a fresh output directory
when changing the dataset selection or protocol so old figures are not mixed
with new outputs.

For a quick check:

```bash
python tests/plot_differentiation_energy_ratio.py --dataset all \
  --max-patients 2 --max-trials 2 --calibration-trials 2 \
  --pbs-configurations 0 1 --output-dir /tmp/energy-ratio-check
python -m pytest tests/test_differentiation_energy_ratio.py -q
```
