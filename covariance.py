"""Neural second-moment estimation and timing for optional covariance removal.

Rx is an uncentered second moment in removal coordinates. Rn uses the same
convention (including any residual neural mean after the pipeline's centering),
so subtracting it removes expected neural power consistently.
"""
from __future__ import annotations

from dataclasses import dataclass
import warnings
import numpy as np

from algorithms import apply_lpf_3d, apply_bpf_3d
from benchmarks.methods import BenchmarkConfig, pulse_times, window_samples, windows


def add_covariance_arguments(parser, include_timing=False):
    parser.add_argument("--removal-method", choices=["projection", "covariance"], default="projection")
    parser.add_argument("--neural-cov-source", choices=["rest", "interpulse"], default="rest")
    parser.add_argument("--covariance-lambda", type=float, default=1.0)
    parser.add_argument("--covariance-eps", type=float, default=1e-6)
    parser.add_argument("--covariance-beta", type=float, default=0.99)
    parser.add_argument("--pulse-pre-ms", type=float, default=0.1,
                        help="Pulse window start before trigger, same convention as benchmark templates")
    parser.add_argument("--pulse-post-ms", type=float, default=1.1,
                        help="Pulse window end after trigger; include artifact duration")
    parser.add_argument("--interpulse-middle-fraction", type=float, default=0.5,
                        help="Central fraction of each gap between pulse windows used for Rn")
    parser.add_argument("--interpulse-min-samples", type=int, default=20,
                        help="Minimum pooled safe samples; otherwise use rest or bypass with a warning")
    if include_timing:
        parser.add_argument("--stim-times", type=str,
                            help="Numeric .npy of shared or per-trial pulse indices relative to dataset.X")
        parser.add_argument("--first-pulse-sample", type=int,
                            help="Verified first pulse index for a periodic schedule; phase is never inferred")


def covariance_config_kwargs(args):
    return {name: getattr(args, name) for name in (
        "removal_method", "neural_cov_source", "covariance_lambda", "covariance_eps",
        "covariance_beta", "pulse_pre_ms", "pulse_post_ms", "interpulse_middle_fraction",
        "interpulse_min_samples")}


def validate_covariance_config(config):
    if config.removal_method not in ("projection", "covariance"):
        raise ValueError("removal_method must be projection or covariance")
    if config.neural_cov_source not in ("rest", "interpulse"):
        raise ValueError("neural_cov_source must be rest or interpulse")
    if not np.isfinite([config.covariance_lambda, config.covariance_eps, config.covariance_beta]).all():
        raise ValueError("Covariance parameters must be finite")
    if config.covariance_lambda < 0 or config.covariance_eps <= 0 or not 0 <= config.covariance_beta < 1:
        raise ValueError("Require lambda >= 0, eps > 0 and 0 <= covariance_beta < 1")
    if not np.isfinite(config.interpulse_middle_fraction) or not 0 < config.interpulse_middle_fraction <= 1:
        raise ValueError("interpulse_middle_fraction must be in (0, 1]")
    if not isinstance(config.interpulse_min_samples, (int, np.integer)) or config.interpulse_min_samples < 2:
        raise ValueError("interpulse_min_samples must be an integer >= 2")


def resolve_stim_times(dataset, config, stim_times=None):
    """Resolve full-dataset triggers, with explicit phases and no automatic detection."""
    if stim_times is not None and config.first_pulse_sample is not None:
        raise ValueError("Supply either stim_times or first_pulse_sample")
    if stim_times is None and config.first_pulse_sample is not None:
        first = config.first_pulse_sample
        if not isinstance(first, (int, np.integer)) or not 0 <= first < dataset.X.shape[-1]:
            raise ValueError("first_pulse_sample must be an integer inside dataset.X")
        if not np.isfinite([dataset.fs, dataset.stim_rate]).all() or not 0 < dataset.stim_rate < dataset.fs:
            raise ValueError("A periodic pulse schedule requires 0 < stim_rate < fs")
        count = int(np.ceil(dataset.X.shape[-1] * dataset.stim_rate / dataset.fs))
        stim_times = np.rint(first + np.arange(count) * dataset.fs / dataset.stim_rate).astype(int)
        stim_times = np.unique(stim_times[stim_times < dataset.X.shape[-1]])
    if stim_times is None:
        stim_times = (dataset.metadata or {}).get("stim_times")
    if stim_times is None:
        return None
    return pulse_times(stim_times, dataset.X.shape[0], dataset.X.shape[-1])


def interpulse_mask(times, n_samples, fs, pre_ms=0.1, post_ms=1.1, middle_fraction=0.5):
    """Middle of gaps between merged half-open pulse windows; no edge extrapolation.

    Excludes (1-middle_fraction)/2 of the free gap on each side. Merging pulse
    windows prevents a long/overlapping pulse being mistaken for a clean gap.
    """
    if not np.isfinite(middle_fraction) or not 0 < middle_fraction <= 1:
        raise ValueError("middle_fraction must be in (0, 1]")
    pre, post = window_samples(fs, BenchmarkConfig(pre_ms=pre_ms, post_ms=post_ms))
    intervals = windows(pulse_times(times, 1, n_samples)[0], n_samples, pre, post)
    mask = np.zeros(n_samples, dtype=bool)
    margin = (1 - middle_fraction) / 2
    for (_, end), (begin, _) in zip(intervals[:-1], intervals[1:]):
        gap = begin - end
        start = end + int(np.ceil(margin * gap))
        stop = begin - int(np.ceil(margin * gap))
        if stop > start:
            mask[start:stop] = True
    return mask


def _moments(samples):
    """Stable population moments; inputs are channels-by-samples, never derivatives."""
    if samples.ndim != 2 or samples.shape[1] < 2 or not np.isfinite(samples).all():
        raise ValueError("Need at least two finite neural samples per channel")
    mean = samples.mean(axis=1)
    centered = samples - mean[:, None]
    covariance = centered @ centered.T / samples.shape[1]
    if not np.isfinite(covariance).all():
        raise ValueError("Neural covariance overflowed; rescale the input channels")
    return mean, (covariance + covariance.T) / 2


def _rest_samples(dataset, config, n_selected):
    rest = dataset.baseline if dataset.baseline is not None else dataset.y_clean
    if rest is None:
        return None, "unavailable"
    rest = np.asarray(rest, dtype=float)
    if rest.ndim != 3 or rest.shape[1] != dataset.X.shape[1] or any(s == 0 for s in rest.shape) or not np.isfinite(rest).all():
        raise ValueError("Clean/rest data must be finite (trials, matching_channels, samples)")
    # Same selected trials where baselines are paired; independent rest may have
    # any trial count and is pooled without requiring time alignment with X.
    if rest.shape[0] == dataset.X.shape[0]:
        rest = rest[-n_selected:]
    if config.preprocessing == "lpf":
        rest = apply_lpf_3d(rest, config.lpf_cutoff_hz, dataset.fs, order=config.lpf_order)
    elif config.preprocessing == "bpf":
        rest = apply_bpf_3d(rest, config.bpf_low_hz, config.bpf_high_hz, dataset.fs, order=config.bpf_order)
    source = "baseline" if dataset.baseline is not None else "y_clean"
    return rest.transpose(1, 0, 2).reshape(rest.shape[1], -1), source


@dataclass
class NeuralCovariance:
    mean: np.ndarray | None
    covariance: np.ndarray | None
    stats: object
    diagnostics: dict

    def at(self, trial, sample):
        """Express pooled raw neural moments in the exact removal coordinates.

        Includes the centering offset so Rn is comparable to Rx = E[x x.T].
        Streaming normalization uses the current mean/scale histories rather
        than independently normalizing the resting data.
        """
        if self.mean is None:
            return None
        mode = self.stats.mode
        if mode == "none":
            mean, scale = np.zeros_like(self.mean), np.ones_like(self.mean)
        elif mode == "global_per_channel":
            mean = self.stats.mean[0, :, 0]
            scale = self.stats.scale[0, :, 0]
        elif mode == "per_trial":
            mean = self.stats.mean[trial, 0, 0]
            scale = self.stats.scale[trial, 0, 0]
        elif mode == "streaming_causal":
            mean = self.stats.debug["mu_hist"][trial, :, sample]
            scale = self.stats.debug["scale_hist"][trial, :, sample]
        else:
            raise ValueError(f"Unsupported removal normalization: {mode}")
        offset = self.mean - mean
        matrix = (self.covariance + np.outer(offset, offset)) / np.outer(np.broadcast_to(scale, self.mean.shape),
                                                                 np.broadcast_to(scale, self.mean.shape))
        return (matrix + matrix.T) * 0.5


def prepare_neural_covariance(dataset, config, X_pre, stats, stim_times=None):
    """Pool rest or safe inter-pulse samples before applying removal scaling.

    Inter-pulse estimation is offline/pooled (it can use later gaps in selected
    trials). Insufficient gaps use an explicitly reported rest fallback; if rest
    is also unavailable, covariance removal is bypassed, never estimated from
    contaminated stimulation data.
    """
    validate_covariance_config(config)
    if X_pre.ndim != 3 or X_pre.shape[1] != dataset.X.shape[1] or not np.isfinite(X_pre).all():
        raise ValueError("Covariance removal requires finite preprocessed signal with matching channels")
    diagnostics = {"requested_source": config.neural_cov_source, "source": config.neural_cov_source,
                   "neural_samples": 0, "safe_samples": 0}
    samples = None
    if config.neural_cov_source == "interpulse":
        window_samples(dataset.fs, BenchmarkConfig(pre_ms=config.pulse_pre_ms, post_ms=config.pulse_post_ms))
        full_times = resolve_stim_times(dataset, config, stim_times)
        masks = np.zeros((X_pre.shape[0], X_pre.shape[-1]), dtype=bool)
        if full_times is not None:
            selected = full_times[-X_pre.shape[0]:]
            for tr, times in enumerate(selected):
                masks[tr] = interpulse_mask(times, X_pre.shape[-1], dataset.fs,
                                            config.pulse_pre_ms, config.pulse_post_ms,
                                            config.interpulse_middle_fraction)
        # Align to the same original-time samples used for non-differential
        # removal after derivative cropping, while keeping pulse indices raw.
        offset = config.derivative_order if config.use_derivative_for_tracking else 0
        masks[:, :offset] = False
        diagnostics["safe_mask"] = masks
        diagnostics["safe_samples"] = int(masks.sum())
        if masks.sum() >= config.interpulse_min_samples:
            samples = X_pre.transpose(1, 0, 2)[:, masks]
        else:
            warnings.warn("Insufficient safe inter-pulse samples; attempting clean/rest fallback", RuntimeWarning)
            diagnostics["source"] = "rest_fallback"
    if samples is None:
        samples, description = _rest_samples(dataset, config, X_pre.shape[0])
        diagnostics["rest_data"] = description
        if samples is None or samples.shape[1] < 2:
            if config.neural_cov_source == "rest":
                raise ValueError("Rest covariance requires dataset.baseline or dataset.y_clean with at least two samples")
            warnings.warn("No usable rest fallback: covariance removal bypassed", RuntimeWarning)
            diagnostics["source"] = "bypass"
            return NeuralCovariance(None, None, stats, diagnostics)
    diagnostics["neural_samples"] = samples.shape[1]
    mean, cov = _moments(samples)
    return NeuralCovariance(mean, cov, stats, diagnostics)
