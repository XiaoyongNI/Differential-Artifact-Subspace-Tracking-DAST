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
all window methods preserve samples outside processed windows. These are offline
benchmarks: SVD/ICA fit the current window and fixed template averaging requires
explicit training data for held-out evaluation. No clean labels are used to fit
any method. Artifact-dominant rank/kurtosis assumptions can remove neural signal.

The runner writes cleaned signals, removed components, config, selected trial
indices, stimulation times and sample offset to NPZ files, plus runtime and
reconstruction metrics in `summary.csv`. RMSE is computed only for a clean
reference with the same shape as `X`. In particular, ERAASR's prestimulation
`y_clean` is a separate interval and is **not** used as aligned ground truth.
`removed_energy_fraction` measures the size of the correction, not neural loss.
Use synthetic data for reconstruction checks or a separately designed clean
period experiment for neural preservation.

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
