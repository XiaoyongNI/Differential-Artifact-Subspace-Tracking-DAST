"""Artifact removal benchmarks using arrays shaped (trials, channels, time)."""
from __future__ import annotations

from dataclasses import dataclass, field
import warnings
import numpy as np


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


def template_subtraction(x, stim_times, fs, config=None, training_data=None, training_stim_times=None):
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
            components.append({"trial": trial, "window": (start, end), "removed_components": indices.tolist()})
    return BenchmarkResult(cleaned, x-cleaned, {"components": components, "skipped_degenerate_windows": skipped})


def window_svd(x, stim_times, fs, config=None):
    """Fit centered spatial SVD per merged artifact window; remove leading ranks."""
    return _decomposition(x, stim_times, fs, config or BenchmarkConfig(method="window_svd"), False)


def window_ica(x, stim_times, fs, config=None):
    """Fit FastICA per merged artifact window and subtract selected contributions."""
    return _decomposition(x, stim_times, fs, config or BenchmarkConfig(method="window_ica"), True)


def pulse(x, stim_times=None, fs=None, config=None):
    """Upstream PULSE repository's marker-free LowRankTV (not a named PULSE class)."""
    from ._pulse_low_rank_tv import LowRankTV
    config = config or BenchmarkConfig(method="pulse")
    x = validate_signal(x)
    if config.rho <= 0 or config.lambda_tv < 0 or config.max_iters < 1 or config.tol <= 0:
        raise ValueError("rho, max_iters, tol must be positive; lambda_tv nonnegative")
    model = LowRankTV(lambda_tv=config.lambda_tv, rho=config.rho, max_iters=config.max_iters,
                      tol=config.tol, verbose=False)
    cleaned = model.fit(x).transform(x)
    return BenchmarkResult(cleaned, x-cleaned, {"implementation": "upstream LowRankTV",
                           "commit": "295d0c9fe1195efacb6454fe303109bb83f8c4a4",
                           "artifact_definition": "input minus returned neural estimate"})


METHODS = {"linear_interpolation": linear_interpolation, "template_subtraction": template_subtraction,
           "window_svd": window_svd, "window_ica": window_ica, "pulse": pulse, "low_rank_tv": pulse}


def run_benchmark(dataset, config=None, stim_times=None, **kwargs):
    """Run directly on a data_loading.Dataset; input and sampling grid are preserved."""
    config = config or BenchmarkConfig()
    if config.method not in METHODS:
        raise ValueError(f"Unknown method {config.method!r}; choose from {sorted(METHODS)}")
    return METHODS[config.method](dataset.X, stim_times=stim_times, fs=dataset.fs, config=config, **kwargs)
