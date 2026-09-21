"""Artifact removal benchmarks using arrays shaped (trials, channels, time)."""
from __future__ import annotations

from dataclasses import dataclass, field
import warnings
import numpy as np
from utils import concatenate_trials
from .lrr import OnlineLRR, fit_lrr
from .asar import OnlineASAR, calibrate_asar


@dataclass
class BenchmarkConfig:
    method: str = "linear_interpolation"
    pre_ms: float = 0.1
    post_ms: float = 1.1
    rank: int = 1
    template_history: int | None = 3  # None: average all training pulse epochs
    ica_components: int | None = None
    ica_selection: str = "kurtosis"
    random_state: int = 0
    max_iters: int = 200
    tol: float = 1e-4
    lambda_tv: float = 1.0
    rho: float = 1.0
    asar_filter_length: int = 201
    asar_mu: float = 0.1
    asar_threshold: float = 5.0
    asar_epsilon: float = 1e-4
    asar_stats_samples: int = 500
    asar_reference_channels: tuple[int, ...] | None = None
    dictionary_min_points: int = 2
    dictionary_min_cluster_size: int = 3
    dictionary_outlier_threshold: float = .95
    dictionary_bracket: int = 6
    dictionary_baseline_samples: int = 3
    dictionary_normalize: str = 'preAverage'
    dictionary_match: str = 'corr'
    dictionary_onset_threshold: float = 1.5
    dictionary_detection_ms: float = 4.
    dictionary_min_duration_ms: float = .5
    dictionary_end_percentile: float = 75.
    pulse_epochs: int = 200
    pulse_batch_size: int = 1
    pulse_learning_rate: float = 1e-3
    pulse_device: str = 'auto'
    pulse_artifact_duration_ms: float = 40.
    pulse_blur_samples: int = 5
    pulse_predict_neural: bool = False
    pulse_uncertainty: bool = False
    pulse_f_cutoff: float = 10.
    pulse_w_cosine: float = 5.
    pulse_w_rank_a: float = 1.5
    pulse_w_rank_s: float = 1.
    pulse_w_spectral: float = 1.2
    pulse_w_spectral_slope: float = 5.


@dataclass
class BenchmarkResult:
    cleaned: np.ndarray
    artifact: np.ndarray
    diagnostics: dict = field(default_factory=dict)


def validate_signal(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 3 or any(size == 0 for size in x.shape):
        raise ValueError("Signals must be nonempty (trials, channels, time) arrays")
    if not np.isfinite(x).all():
        raise ValueError("Signals must be finite")
    return x


def pulse_times(stim_times, n_trials, n_samples):
    """Accept shared 1D sample indices or one sequence per trial (including ragged)."""
    if stim_times is None:
        raise ValueError("This method requires explicit stimulation sample indices")
    rows = list(stim_times)
    if not rows or np.isscalar(rows[0]):
        rows = [rows] * n_trials
    if len(rows) != n_trials:
        raise ValueError("Provide one stimulation-time sequence per trial")
    out = []
    for row in rows:
        a = np.asarray(row)
        if a.ndim != 1 or not np.isfinite(a).all() or np.any(a != np.floor(a)):
            raise ValueError("Stimulation times must be finite integer sample indices")
        a = np.unique(a.astype(int))
        if np.any(a < 0) or np.any(a >= n_samples):
            raise ValueError("Stimulation times must lie inside the signal")
        out.append(a)
    return out


def window_samples(fs, config):
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be positive and finite")
    if not np.isfinite([config.pre_ms, config.post_ms]).all() or config.pre_ms < 0 or config.post_ms <= 0:
        raise ValueError("pre_ms must be nonnegative and post_ms positive")
    return int(round(config.pre_ms * fs / 1000)), max(1, int(round(config.post_ms * fs / 1000)))


def windows(times, n_samples, pre, post):
    """Merge half-open intervals so overlapping artifacts are processed once."""
    merged = []
    for t in times:
        start, end = max(0, int(t) - pre), min(n_samples, int(t) + post)
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def linear_interpolation(x, stim_times, fs, config=None):
    config = config or BenchmarkConfig()
    x = validate_signal(x)
    pre, post = window_samples(fs, config)
    times = pulse_times(stim_times, x.shape[0], x.shape[-1])
    cleaned = x.copy()
    skipped = 0
    for trial, row in enumerate(times):
        for start, end in windows(row, x.shape[-1], pre, post):
            # Anchors must be outside the contaminated interval. Edge windows
            # without two clean anchors are explicitly left unchanged.
            if start == 0 or end == x.shape[-1]:
                skipped += 1
                continue
            weight = (np.arange(start, end) - (start - 1)) / (end - (start - 1))
            cleaned[trial, :, start:end] = (
                x[trial, :, start-1, None] * (1-weight)
                + x[trial, :, end, None] * weight
            )
    return BenchmarkResult(cleaned, x-cleaned, {"skipped_edge_windows": skipped})


def average_template_subtraction(x, stim_times, fs, config=None, training_data=None, training_stim_times=None):
    """Subtract average pulse epochs, using prior pulses or separate training data.

    history=None learns a fixed template from training_data (defaults to x).
    A finite history uses only previous complete raw epochs within each trial,
    matching upstream BackwardTemplateSubtraction. First pulses remain intact.
    Overlapping template estimates are averaged before subtraction.
    """
    config = config or BenchmarkConfig(method="template_subtraction")
    x = validate_signal(x)
    pre, post = window_samples(fs, config)
    times = pulse_times(stim_times, x.shape[0], x.shape[-1])
    history = config.template_history
    if history is not None and (not isinstance(history, int) or history < 1):
        raise ValueError("template_history must be a positive integer or None")
    template = None
    if history is None:
        train = x if training_data is None else validate_signal(training_data)
        if train.shape[1] != x.shape[1]:
            raise ValueError("Training and test channel counts must match")
        train_times = pulse_times(stim_times if training_stim_times is None else training_stim_times,
                                 train.shape[0], train.shape[-1])
        epochs = [train[tr, :, t-pre:t+post] for tr, row in enumerate(train_times)
                  for t in row if t >= pre and t+post <= train.shape[-1]]
        if not epochs:
            raise ValueError("No complete training pulse epochs for averaging")
        template = np.mean(epochs, axis=0)
    artifact = np.zeros_like(x)
    coverage = np.zeros((x.shape[0], x.shape[-1]))
    skipped = 0
    for trial, row in enumerate(times):
        previous = []
        for t in row:
            if t < pre or t+post > x.shape[-1]:
                skipped += 1
                continue
            epoch = x[trial, :, t-pre:t+post]
            estimate = template if history is None else (np.mean(previous[-history:], axis=0) if previous else None)
            if estimate is not None:
                artifact[trial, :, t-pre:t+post] += estimate
                coverage[trial, t-pre:t+post] += 1
            previous.append(epoch)
    artifact /= np.maximum(coverage[:, None, :], 1)
    return BenchmarkResult(x-artifact, artifact, {"template": template, "skipped_edge_epochs": skipped})


def _decomposition(x, stim_times, fs, config, use_ica):
    x = validate_signal(x)
    pre, post = window_samples(fs, config)
    times = pulse_times(stim_times, x.shape[0], x.shape[-1])
    if not isinstance(config.rank, int) or not 0 <= config.rank <= x.shape[1]:
        raise ValueError("rank must be an integer between zero and the channel count")
    cleaned = x.copy()
    components = []
    skipped = 0
    for trial, row in enumerate(times):
        for start, end in windows(row, x.shape[-1], pre, post):
            epoch = x[trial, :, start:end]
            mean = epoch.mean(axis=1, keepdims=True)
            centered = epoch - mean
            if config.rank == 0:
                continue
            if not use_ica:
                u, s, vt = np.linalg.svd(centered, full_matrices=False)
                r = min(config.rank, len(s))
                removed = (u[:, :r] * s[:r]) @ vt[:r]
                indices = np.arange(r)
            else:
                from sklearn.decomposition import FastICA
                from sklearn.exceptions import ConvergenceWarning
                from scipy.stats import kurtosis
                available = np.linalg.matrix_rank(centered)
                n = available if config.ica_components is None else min(config.ica_components, available)
                if n < 1 or end-start < 3:
                    skipped += 1
                    continue
                if config.ica_selection not in ("kurtosis", "energy"):
                    raise ValueError("ica_selection must be kurtosis or energy")
                ica = FastICA(n_components=n, whiten="unit-variance", random_state=config.random_state,
                              max_iter=config.max_iters, tol=config.tol)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    sources = ica.fit_transform(epoch.T)
                # Whitened ICA sources have equal variance: use reconstructed
                # component energy or absolute kurtosis, not source variance.
                score = (np.abs(kurtosis(sources, axis=0, fisher=True)) if config.ica_selection == "kurtosis"
                         else np.sum(sources**2, axis=0) * np.sum(ica.mixing_**2, axis=0))
                indices = np.argsort(np.nan_to_num(score, nan=-np.inf))[::-1][:min(config.rank, n)]
                removed = (sources[:, indices] @ ica.mixing_[:, indices].T).T
                if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                    warnings.warn(f"ICA did not converge in trial {trial}, window [{start}:{end}]", ConvergenceWarning)
            cleaned[trial, :, start:end] = epoch-removed
            component = {"trial": trial, "window": (start, end), "removed_components": indices.tolist()}
            if use_ica:
                component["removal_operator"] = ica.mixing_[:, indices] @ ica.components_[indices]
            else:
                component["removal_basis"] = u[:, :r].copy()
            components.append(component)
    return BenchmarkResult(cleaned, x-cleaned, {"components": components, "skipped_degenerate_windows": skipped})


def window_svd(x, stim_times, fs, config=None):
    """Fit centered spatial SVD per merged artifact window; remove leading ranks."""
    return _decomposition(x, stim_times, fs, config or BenchmarkConfig(method="window_svd"), False)


def window_ica(x, stim_times, fs, config=None):
    """Fit FastICA per merged artifact window and subtract selected contributions."""
    return _decomposition(x, stim_times, fs, config or BenchmarkConfig(method="window_ica"), True)


def svd_template_subtraction(x, stim_times, fs, config=None,
                             training_data=None, training_stim_times=None):
    """Window SVD followed by average template subtraction of its residual.

    Both stages reuse the same pulse windows/configuration. For fixed-template
    fitting, separate training recordings also pass through SVD first; no raw
    training template is subtracted from an SVD-cleaned test signal.
    """
    config = config or BenchmarkConfig(method='svd_template_subtraction')
    x = validate_signal(x)
    # Materialize once so both stages also support generator marker inputs.
    markers = None if stim_times is None else list(stim_times)
    times = pulse_times(markers, x.shape[0], x.shape[-1])
    train_cleaned = None
    train_times = None
    if config.template_history is None and training_data is not None:
        train = validate_signal(training_data)
        if train.shape[1] != x.shape[1]:
            raise ValueError('Training and test channel counts must match')
        train_times = pulse_times(markers if training_stim_times is None else training_stim_times,
                                  train.shape[0], train.shape[-1])
        train_cleaned = window_svd(train, train_times, fs, config).cleaned
    svd_result = window_svd(x, times, fs, config)
    template_result = average_template_subtraction(
        svd_result.cleaned, times, fs, config,
        training_data=train_cleaned, training_stim_times=train_times)
    return BenchmarkResult(template_result.cleaned, x-template_result.cleaned, {
        'stage_order': ('window_svd', 'template_subtraction'),
        'svd': svd_result, 'template_subtraction': template_result})


def low_rank_tv(x, stim_times=None, fs=None, config=None):
    """Upstream PULSE repository's marker-free LowRankTV (not a named PULSE class)."""
    from ._pulse_low_rank_tv import LowRankTV
    config = config or BenchmarkConfig(method="low_rank_tv")
    x = validate_signal(x)
    if config.rho <= 0 or config.lambda_tv < 0 or config.max_iters < 1 or config.tol <= 0:
        raise ValueError("rho, max_iters, tol must be positive; lambda_tv nonnegative")
    model = LowRankTV(lambda_tv=config.lambda_tv, rho=config.rho, max_iters=config.max_iters,
                      tol=config.tol, verbose=False)
    cleaned = model.fit(x).transform(x)
    return BenchmarkResult(cleaned, x-cleaned, {"implementation": "upstream LowRankTV",
                           "commit": "295d0c9fe1195efacb6454fe303109bb83f8c4a4",
                           "artifact_definition": "input minus returned neural estimate"})


def fit_lrr_trials(training_data, training_stim_times, fs, config=None):
    """Offline adapter: pool artifact-period samples from explicit training trials."""
    train = validate_signal(training_data)
    pre, post = window_samples(fs, config or BenchmarkConfig())
    times = pulse_times(training_stim_times, train.shape[0], train.shape[-1])
    mask = np.zeros((train.shape[0], train.shape[-1]), dtype=bool)
    for trial, row in enumerate(times):
        for start, end in windows(row, train.shape[-1], pre, post):
            mask[trial, start:end] = True
    return fit_lrr(concatenate_trials(train), mask.ravel())


def lrr(x, stim_times=None, fs=None, config=None, W=None):
    """Replay trials through the one-sample API using explicitly pretrained W.

    stim_times, fs and config are accepted for benchmark compatibility only.
    No training or trigger-dependent gating happens here.
    """
    x = validate_signal(x)
    if W is None:
        raise ValueError("LRR requires pretrained W; call fit_lrr or fit_lrr_trials offline")
    model = OnlineLRR(W)
    if model.W.shape[0] != x.shape[1]:
        raise ValueError("Signal and W channel counts must match")
    cleaned = np.empty_like(x)
    for trial in range(x.shape[0]):
        model.reset()
        for t in range(x.shape[-1]):
            cleaned[trial, :, t] = model.process_sample(x[trial, :, t])
    return BenchmarkResult(cleaned, x-cleaned, {"W": model.W,
                           "processing": "fixed-weight sample-by-sample"})


def asar_model(mean, std, config, weights=None):
    """Shared construction for online cancellation and frozen rest evaluation."""
    return OnlineASAR(mean, std, filter_length=config.asar_filter_length,
                      mu=config.asar_mu, threshold=config.asar_threshold,
                      epsilon=config.asar_epsilon,
                      reference_channels=config.asar_reference_channels, weights=weights)


def asar(x, stim_times=None, fs=None, config=None, calibration_data=None):
    """Replay ASAR sample by sample, independently resetting each trial.

    Default calibration replays the first N test samples, matching MATLAB;
    that initialization is offline. For causal inference, provide a separate
    preceding calibration recording, shaped (1 or n_trials, channels, time).
    Ground truth and stimulation markers are never used.
    """
    config = config or BenchmarkConfig(method="asar")
    x = validate_signal(x)
    calibration = x if calibration_data is None else validate_signal(calibration_data)
    if calibration.shape[0] not in (1, x.shape[0]) or calibration.shape[1] != x.shape[1]:
        raise ValueError("Calibration must have matching channels and one or n_trials trials")
    cleaned = np.empty_like(x)
    means, stds, weights = [], [], []
    for trial in range(x.shape[0]):
        mean, std = calibrate_asar(calibration[0 if calibration.shape[0] == 1 else trial],
                                   config.asar_stats_samples)
        model = asar_model(mean, std, config)
        for t in range(x.shape[-1]):
            cleaned[trial, :, t] = model.process_sample(x[trial, :, t])
        means.append(mean)
        stds.append(std)
        weights.append(model.weights)
    return BenchmarkResult(cleaned, x-cleaned, {
        "mean": np.stack(means), "std": np.stack(stds), "weights": np.stack(weights),
        "reference_channels": model.reference_channels,
        "calibration_source": "test_prefix_replay" if calibration_data is None else "provided",
        "stats_samples": config.asar_stats_samples,
        "output_convention": "post_update", "rest_evaluation": "frozen_weights_zero_history"})


def pulse(x, stim_times=None, fs=None, config=None, **kwargs):
    """PULSE U-Net inference with an explicitly trained model and stimulation trace."""
    from .pulse_nn import pulse as apply_pulse
    return apply_pulse(x, stim_times, fs, config, **kwargs)


def dictionary_learning(x, stim_times=None, fs=None, config=None, **kwargs):
    """Offline clustered artifact templates; see dictionary_learning module."""
    from .dictionary_learning import dictionary_learning as apply_dictionary
    return apply_dictionary(x, stim_times, fs, config, **kwargs)


METHODS = {"dictionary_learning": dictionary_learning, "asar": asar, "lrr": lrr, "linear_interpolation": linear_interpolation, "template_subtraction": average_template_subtraction,
           "svd_template_subtraction": svd_template_subtraction,
           "window_svd": window_svd, "window_ica": window_ica, "pulse": pulse, "low_rank_tv": low_rank_tv}


def run_benchmark(dataset, config=None, stim_times=None, **kwargs):
    """Run directly on a data_loading.Dataset; input and sampling grid are preserved."""
    config = config or BenchmarkConfig()
    if config.method not in METHODS:
        raise ValueError(f"Unknown method {config.method!r}; choose from {sorted(METHODS)}")
    return METHODS[config.method](dataset.X, stim_times=stim_times, fs=dataset.fs, config=config, **kwargs)
