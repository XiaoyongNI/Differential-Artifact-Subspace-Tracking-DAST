# TBME artifact-removal benchmarks

Run from `TBME_clean_code` using the existing NumPy/SciPy environment. ICA also
requires scikit-learn (tested with the project's `py3.10ni` environment).

```bash
python -m benchmarks.run --dataset synthetic
python -m benchmarks.run --dataset ERAASR --first-pulse-sample 20 --n-test 1
```

The second command is an interface example: **20 is illustrative**, not a
verified ERAASR pulse onset. Supply actual indices relative to the loader's
cropped `dataset.X` with `--stim-times triggers.npy`, or a verified first pulse
with `--first-pulse-sample`. Shared 1D triggers or per-trial lists are accepted
by the Python API; the CLI `.npy` input supports numeric rectangular arrays.
No stimulation phase is inferred from stimulation rate alone. Periodic CLI
schedules round each absolute pulse time, allowing noninteger sample periods.

```python
from data_loading import load_dataset
from benchmarks import BenchmarkConfig, run_benchmark
from pipeline import PipelineConfig, run_pipeline

dataset = load_dataset("ERAASR")
# triggers: integer sample indices for each trial, relative to dataset.X
result = run_benchmark(dataset, BenchmarkConfig(method="window_svd", rank=1),
                       stim_times=triggers)
cleaned = result.cleaned  # same shape and sample grid as dataset.X
artifact = result.artifact  # dataset.X - cleaned
past = run_pipeline(dataset, PipelineConfig(n_test=0))
# PAST with first derivative omits sample 0: compare cleaned[:, :, 1:] with
# past.stages["cleaned"]. The runner saves this offset explicitly.
```

## Methods and assumptions

- `lrr`: offline-trained linear regression reference with fixed weights; every
  test sample is processed independently through `OnlineLRR.process_sample`.
  See the training and online API below.
- `linear_interpolation`: replace merged pulse windows with lines between the
  immediately adjacent untouched samples. Edge windows lacking two anchors
  remain unchanged and are counted in diagnostics.
- `template_subtraction`: subtract the average of the previous three complete
  raw pulse epochs per trial by default, following upstream backward template
  subtraction. The first pulse remains unchanged. Set `template_history=None`
  and supply `training_data`/`training_stim_times` to average separate training
  pulses. CLI `--template-history 0` uses preceding trials to train that fixed
  template. Overlapping template estimates are averaged before subtraction.
- `window_svd`: fit a centered channels-by-time SVD independently in each merged
  artifact window, subtracting the leading `rank` spatial components. Channel
  means are retained. This is local SVD, rather than upstream pooled targeted PCA.
- `window_ica`: fit FastICA independently in each merged window, removing `rank`
  components ranked by absolute excess kurtosis (default) or reconstructed
  component energy. Source variance is unsuitable with unit-variance whitening.
  Subtract only those components, preserving contributions outside the fitted
  subspace and channel means. Convergence warnings are reported; constant or
  insufficient-rank windows are skipped. Use `ica_components` to limit fitting.
- `pulse` / `low_rank_tv`: the PULSE repository's marker-free **LowRankTV** method.
  There is no class named PULSE in the referenced methods directory. This alias
  identifies LowRankTV explicitly, rather than asserting it is a separately
  published PULSE method. The upstream solver is vendored unchanged except for
  a minimal base-class shim; it returns the TV component as the neural estimate.
  Its objective is `min nuclear_norm(L) + lambda_tv * TV(S), X = L + S`.
  `lambda_tv`, `rho`, tolerance and iteration limit are configurable. The
  original stopping rule and TV routine are retained, including their numerical
  limitations; outputs are benchmark reproductions, not guaranteed optima.
- `past`: run the existing TBME pipeline on the same selected trials. Runner
  defaults to no input filtering; `--past-preprocessing lpf` selects the usual
  low-pass stage. PAST parameters otherwise follow `PipelineConfig` defaults.

Windows use `[trigger - pre_samples, trigger + post_samples)` with durations
configured in milliseconds. SVD/ICA/interpolation merge overlapping windows;
all window methods preserve samples outside processed windows. The window methods are offline
benchmarks: SVD/ICA fit the current window and fixed template averaging requires
explicit training data for held-out evaluation. No clean labels are used to fit
any method. Artifact-dominant rank/kurtosis assumptions can remove neural signal.

## Sample-by-sample linear regression reference

LRR follows the channel-specific weighted-reference approach of
[Young et al., J Neural Eng 2018, 15:026014](https://doi.org/10.1088/1741-2552/aa9ee8).
This baseline fits uncentered ordinary least squares (`np.linalg.lstsq`, no
intercept) using only explicitly marked artifact samples from training data.
Each target uses every other simultaneous channel; the weight diagonal is zero.

```python
from benchmarks import fit_lrr, OnlineLRR, apply_lrr_batch

W = fit_lrr(X_train, artifact_mask)  # (C, T_train), boolean (T_train,)
lrr = OnlineLRR(W)
for t in range(X_test.shape[1]):
    x_clean_t = lrr.process_sample(X_test[:, t])  # exactly (C,)
    # consume/store x_clean_t immediately
lrr.reset()  # no-op: there is no temporal state
```

Inference performs `x_t - W @ x_t` on every sample, including non-artifact
periods. It has no buffering, windows, future data, trigger input, adaptation,
or matrix inversion. Weights are copied on initialization; `lrr.W` returns a
snapshot. Persistent weight storage is `8*C*C` bytes (float64), plus O(C)
temporary sample storage; work is O(C²) per call. `apply_lrr_batch(X, W)` is
provided only for offline numerical verification and is not used by the runner.

```bash
python -m benchmarks.run --dataset synthetic --methods lrr --n-test 1 \
  --pre-ms 0 --post-ms 3 --out-dir plots/lrr
python -m unittest benchmarks.test_lrr benchmarks.test_methods benchmarks.test_metrics
```

The runner trains on preceding trials and tests on the last `--n-test` trials;
LRR requires at least one of each. `fit_lrr_trials` reuses the shared trigger,
pulse-interval and trial-concatenation helpers to construct the **offline training
mask only**. Choose `--pre-ms`/`--post-ms` to cover the training artifact duration.
Test triggers do not enter the LRR calculation. The Python benchmark API accepts
only pretrained weights: `run_benchmark(dataset, BenchmarkConfig(method="lrr"),
W=W)`. Training never defaults to the test recording.

CSV `training_seconds` reports fitting separately; `seconds` includes online
replay and benchmark validation/output allocation. `seconds_per_sample` divides
that replay time by the number of multichannel samples, so it is an amortized
runner measurement, not an isolated call-latency measurement. NPZ files save
`lrr_weights` and `training_trial_indices`. The recording adapter stores complete
outputs for existing metrics; the online class itself stores no recording.
Rest-preservation metrics reuse the same fixed W without refitting.

Independent channel activity is approximately preserved when many channels
provide a common artifact reference. Correlated neural activity can also be
subtracted; preservation is not guaranteed for every recording. Empty training
masks are rejected; rank-deficient fits use the minimum-norm solution. With one
channel there are no predictors and the output is unchanged.

The runner writes cleaned signals, removed components, config, selected trial
indices, stimulation times and sample offset to NPZ files, plus runtime and
reconstruction metrics in `summary.csv`. RMSE is computed only for a clean
reference with the same shape as `X`. In particular, ERAASR's prestimulation
`y_clean` is a separate interval and is **not** used as aligned ground truth.
Rest-period correlations and clean/rest energy loss are defined below. Use
synthetic data for reconstruction checks; a separate rest period is a similarity
reference and a clean neural-preservation test input.

## Upstream provenance

Read from https://github.com/spbui00/pulse/tree/main/sparc/methods at commit
`295d0c9fe1195efacb6454fe303109bb83f8c4a4`:
`interpolation/linear_interpolation.py`, `template_subtraction/base.py`,
`template_subtraction/backward_template_subtraction.py`, `decomposition/pca.py`,
`decomposition/ica.py`, `decomposition/local_ica.py`,
`decomposition/sparse_local_projection.py`, and `decomposition/low_rank_tv.py`.

Local window methods adapt these ideas to the TBME array interface and robust
window handling. SparseLocalProjection is a different upstream method requiring
a stimulation channel and markers; it is not the `pulse` alias here.
The vendored LowRankTV code is copyright 2025 Han Bui, MIT licensed; the full
notice is in `LICENSE-PULSE`.

```bash
python -m unittest benchmarks.test_methods
```

## Covariance-aware PAST removal

`projection` remains the default. The standalone `project_and_remove` function
and the pipeline's legacy projection expression are unchanged. Optional
covariance removal uses the same identified U, active k, differentiation,
normalization, adaptive-k rules, harmonic stage, and evaluation metrics.

```bash
# All three removal routes on the same synthetic trial and sample grid.
python -m benchmarks.run --dataset synthetic --methods past --compare-removal \
  --pulse-post-ms 3 --out-dir plots/covariance_comparison

# The same comparison on ERAASR with measured pulse markers (indices relative
# to load_eraasr().X, whose crop begins at original sample 1600).
python -m benchmarks.run --dataset ERAASR --methods past --compare-removal \
  --stim-times triggers.npy --past-preprocessing lpf \
  --past-normalization global_per_channel --out-dir plots/eraasr_covariance

# Select a single covariance route in the ordinary main runner.
python main.py --dataset ERAASR --removal-method covariance \
  --neural-cov-source rest --normalization global_per_channel
python main.py --dataset ERAASR --removal-method covariance \
  --neural-cov-source interpulse --stim-times triggers.npy \
  --pulse-pre-ms .1 --pulse-post-ms 1.1 --interpulse-middle-fraction .5
```

Python API: `run_pipeline(dataset, PipelineConfig(removal_method="covariance",
neural_cov_source="interpulse"), stim_times=triggers)`. `stim_times` may be shared
or per-trial, including ragged lists. The full dataset's triggers are supplied
before `n_test` selection. If omitted, `dataset.metadata["stim_times"]` can supply
them. Alternatively `first_pulse_sample` / `--first-pulse-sample` generates a
periodic schedule from an explicitly verified phase. Rate alone is insufficient.

Neural covariance sources:

- **rest**: use `dataset.baseline` preferentially, otherwise its provided clean
  `dataset.y_clean`. Paired rest trials follow the selected stimulation trials;
  independent rest recordings can have a different number of trials or samples.
  Apply the same LPF/BPF as the stimulation path, with no differentiation, then
  pool channel moments. Transform them with the stimulation signal's actual
  mean and scale, not independently fitted rest normalization. This includes the
  scalar per-trial convention and current streaming mean/scale histories.
- **interpulse**: use only samples between the end of one pulse window and the
  start of the next. Windows have the template subtraction convention
  `[trigger - pulse_pre_ms, trigger + pulse_post_ms)` in converted samples.
  Merge overlapping windows first. Retain the central
  `interpulse_middle_fraction` of each free gap (default .5), rounding the two
  excluded margins inward. No samples before the first pulse or after the last
  pulse are assumed safe. Pool these samples offline across selected trials,
  exclude samples cropped off by differentiation, and transform pooled moments
  into the same removal coordinates as for rest. Future gaps can participate;
  this estimator is not a causal real-time inter-pulse estimator.

Because `Rx` is the specified uncentered EMA of `x_t @ x_t.T`, `Rn` is also an
expected second moment in removal coordinates: scaled centered neural covariance
plus the outer product of its mean relative to the signal's centering offset.
This includes any nonzero residual mean and avoids subtracting centered variance
from uncentered signal power. No clean labels are used for subspace identification;
provided clean/rest data are used explicitly for the rest covariance route.

`Rx` resets to the initial `Rn` at each trial, then updates with
`covariance_beta` (default .99) using the normalized, non-differential signal
entering removal. Current active U columns give
`max(diag(U.T @ (Rx - Rn) @ U), 0)` on every sample. No artifact power is inferred
from projection coefficients. Suppression is
`Rn @ solve(Rn + covariance_lambda * U @ Lambda_a @ U.T + covariance_eps * I, x)`;
a Cholesky solve is used without an explicit matrix inverse. `covariance_eps` is
absolute in removal coordinates; tune it when input scales differ greatly.

Covariances are symmetrized and checked for PSD; tiny negative roundoff
eigenvalues are clipped and invalid matrices/dimensions/parameters rejected.
Fewer than `interpulse_min_samples` safe samples (default 20) trigger an explicit
rest fallback. If no usable rest exists either, removal is bypassed with a warning;
the tracker and later harmonic stage still run. A requested rest route with no
clean/rest data raises an explanatory error. Pulse width and middle fraction
must be chosen to exclude the actual pulse/transient duration, especially with
filters having long impulse responses.

The comparison NPZ files include Rn at the beginning/end of each trial, final Rx,
artifact powers, and inter-pulse masks. CSV `neural_cov_source` and saved covariance
diagnostics distinguish actual `interpulse`, `rest_fallback`, and `bypass` routes.
RMSE/relative RMSE definitions are unchanged; ERAASR still has no aligned
reconstruction reference in the runner, but now has rest similarity and
clean/rest preservation metrics.

```bash
python -m unittest test_covariance_suppression benchmarks.test_methods
```

## Label ERAASR pulse triggers

```bash
python label_eraasr_triggers.py --trial 0
python -m benchmarks.run --dataset ERAASR --methods past --compare-removal \
  --stim-times plots/eraasr_triggers/triggers.npy \
  --past-preprocessing lpf --past-normalization global_per_channel
```

The labeling script detects all trials independently and plots one zero-based
trial (default 0, plotted channel 15). It finds robust channel-median absolute
derivative peaks constrained by the known pulse period, then searches backward
to the transient's leading edge. Thresholds and search fractions are configurable;
`--detection-channel` selects a single channel instead of the median score.
It does not fill missing pulses with a regular schedule. These are signal-derived
onset estimates, not acquisition hardware timestamps.

Outputs are `triggers.npy` (numeric trials-by-pulses, when counts agree), a JSON
file with detection settings/diagnostics, a CSV with cropped and original sample
indices, and PNG/PDF plots with a waveform zoom and detection thresholds. Indices
are relative to `load_eraasr().X`; original file indices add 1600. For this dataset,
default settings detect 18 pulses per trial, with 89–91 sample spacing around the
expected 90. If counts disagree, ragged labels are preserved in JSON/CSV rather
than padded or replaced by a shared schedule; pass those lists through the Python
`stim_times` API. Inspect the plot before relying on the pulse-window assumptions.

## Rest-period performance metrics

`summary.csv` and each NPZ's `metrics_json` now include:

- `rest_correlation_time`: mean per-trial/channel Pearson correlation between
  stimulation-period output and the provided clean/rest signal. Pair interval
  starts after the same derivative sample offset; truncate to the shorter
  interval, with no lag search.
- `rest_correlation_freq`: mean per-trial/channel Pearson correlation of linear
  Welch PSDs, using identical frequency bins. Hann segments are 20 ms or the
  shorter recording length, with 50% overlap and constant detrending. DC is
  excluded; all remaining bins through Nyquist are retained.
- `energy_loss_fraction`: mean of the **per-trial** clean/rest loss fractions,
  using exactly the clean-loss script's convention:
  `removed_energy / original_energy = sum((rest - cleaned_rest)**2) / sum(rest**2)`.
- `energy_loss_db`: mean per-trial `10*log10(residual_energy/original_energy)`.
  Zero original/residual energy gives an undefined value, as in the existing
  clean-loss function. Undefined trial/channel values are omitted from means.

The mixed-signal correction fraction has been removed. The clean/rest energy
metric does **not** use stimulation-period `raw-output`, and is not
`1 - residual_energy/original_energy`. NPZs additionally save `rest_original`,
`rest_after_removal`, `rest_energy_rows`, and `rest_energy_row_columns` with the
same trial/original/removed/residual/fraction/dB fields as `clean_energy_loss`.
Individual trial/channel correlations and their rest reference are also saved.

Use `baseline` preferentially, otherwise the supplied `y_clean`. Paired rest
trials follow the same trial selection as stimulation. A single independent rest
trial may be shared; other unmatched trial counts provide mean-rest-PSD similarity
only, with time correlation/energy loss unavailable. Missing, constant, or too
short references give NaN where the corresponding metric is undefined. The rest
signal has the same LPF/BPF as a PAST output; the clean-loss reference is filtered
identically so filter attenuation itself is not counted as subspace loss.

For PAST, clean/rest preservation freezes the **final stimulation-learned** U, k,
Rn/Rx, and channel scales, then applies the linear removal to the neural signal.
It never retrains U or updates Rx on rest. It omits the stimulation signal's
centering offset in this component test: introducing that offset into a different
recording would measure an artificial DC correction rather than the action on a
clean neural component. The rank-one projection route uses `clean_energy_loss`
itself; covariance and other outputs use the identical per-trial energy equations.

For window SVD/ICA, reuse the stimulation-fitted removal basis/operator in the
same relative sample windows, retaining the clean window's own channel mean.
Interpolation uses the same relative pulse positions on rest; template subtraction
replays its stimulation-estimated correction at those positions. These positions
are hypothetical removal locations in an OFF recording, not detected OFF pulses.
LowRankTV has no learned fit state and its configured solver runs directly on the
clean/rest recording. Nonorthogonal/template correction loss fractions may exceed
one; the metric measures correction power, not necessarily net energy attenuation.

Correlation with a separate rest interval is **similarity**, not aligned
reconstruction accuracy. Runtime measures stimulation processing only, excluding
rest metric evaluation. Existing RMSE metrics remain available for aligned clean
references such as the synthetic dataset.

```bash
python -m unittest benchmarks.test_metrics benchmarks.test_methods
```
