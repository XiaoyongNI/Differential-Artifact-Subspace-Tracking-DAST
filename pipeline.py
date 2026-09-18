"""Composable TBME artifact-cancellation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from algorithms import (
    MCC_OPAST,
    PASTd,
    apply_bpf_3d,
    apply_lpf_3d,
    choose_k_from_delta_ema,
    denormalize_channels,
    derivative_time,
    harmonic_rls_3d,
    normalize_channels,
)
from data_loading import Dataset
from utils import harmonic_energy_ratio, ideal_u1s_from_stim_geometry, subspace_angle_deg


@dataclass
class PipelineConfig:
    n_test: Optional[int] = 1
    preprocessing: str = "lpf"
    lpf_cutoff_hz: float = 500.0
    lpf_order: int = 5
    bpf_low_hz: float = 500.0
    bpf_high_hz: float = 8000.0
    bpf_order: int = 5
    normalization: str = "none"
    use_derivative_for_tracking: bool = True
    derivative_order: int = 1
    tracker: str = "PASTd"
    rank: int = 1
    beta: float = 0.999
    reorth_interval: int = 10**9
    mcc_sigma: float = 10.0
    adaptive_k: bool = False
    adaptive_k_threshold: float = 0.1
    adaptive_k_check_period: int = 100
    warm_start_samples: int = 0
    carry_tracker_across_trials: bool = False
    harmonic_filter: bool = False
    harmonic_max_harmonics: int = 10
    harmonic_settling_time: float = 1e-2
    convergence_skip_samples: int = 0
    compute_ideal_u: bool = True
    save_npz: Optional[Path] = None
    extra: Dict[str, object] = field(default_factory=dict)


@dataclass
class PipelineResult:
    dataset: Dataset
    config: PipelineConfig
    stages: Dict[str, np.ndarray]
    diagnostics: Dict[str, np.ndarray | float | dict]


def _make_tracker(config: PipelineConfig, n_channels: int):
    if config.tracker == "PASTd":
        return PASTd(n_channels, config.rank, beta=config.beta, reorth_interval=config.reorth_interval)
    if config.tracker == "MCC_OPAST":
        return MCC_OPAST(n_channels, config.rank, beta=config.beta, sigma=config.mcc_sigma, reorth_interval=config.reorth_interval)
    raise ValueError("tracker must be 'PASTd' or 'MCC_OPAST'.")


def _select_test_trials(X: np.ndarray, y_clean: Optional[np.ndarray], n_test: Optional[int]):
    if n_test is None or n_test <= 0 or n_test >= X.shape[0]:
        return X.copy(), None if y_clean is None else y_clean.copy()
    return X[-n_test:].copy(), None if y_clean is None else y_clean[-n_test:].copy()


def run_pipeline(dataset: Dataset, config: PipelineConfig) -> PipelineResult:
    X_raw, y_clean = _select_test_trials(dataset.X, dataset.y_clean, config.n_test)
    fs = float(dataset.fs)
    stages: Dict[str, np.ndarray] = {"raw": X_raw}

    if config.preprocessing == "lpf":
        X_pre = apply_lpf_3d(X_raw, cutoff=config.lpf_cutoff_hz, fs=fs, order=config.lpf_order)
    elif config.preprocessing == "bpf":
        X_pre = apply_bpf_3d(X_raw, lowcut=config.bpf_low_hz, highcut=config.bpf_high_hz, fs=fs, order=config.bpf_order)
    elif config.preprocessing in (None, "none"):
        X_pre = X_raw.copy()
    else:
        raise ValueError("preprocessing must be 'none', 'lpf', or 'bpf'.")
    stages["preprocessed"] = X_pre

    X_norm, norm_stats = normalize_channels(X_pre, mode=config.normalization)
    stages["normalized"] = X_norm

    if config.use_derivative_for_tracking:
        X_track = derivative_time(X_norm, order=config.derivative_order)
        X_project = X_norm[:, :, config.derivative_order:]
    else:
        X_track = X_norm
        X_project = X_norm
    stages["tracking_input"] = X_track

    n_trials, n_channels, n_steps = X_track.shape
    X_past = np.zeros_like(X_project)
    chosen_k = np.full((n_trials, n_steps), config.rank, dtype=int)
    delta_u = np.zeros((n_trials, n_steps, config.rank), dtype=float)
    delta_u_ema = np.zeros_like(delta_u)
    u_trace = np.zeros((n_trials, n_steps, n_channels, config.rank), dtype=float)

    ideal_u = None
    if config.compute_ideal_u and dataset.stim_channels:
        ideal_u = ideal_u1s_from_stim_geometry(dataset.stim_channels, n_channels, rank=config.rank)

    angle_trace = np.full((n_trials, n_steps, config.rank), np.nan, dtype=float)

    tracker = None
    for tr in range(n_trials):
        if tracker is None or not config.carry_tracker_across_trials:
            tracker = _make_tracker(config, n_channels)
            if config.warm_start_samples > 0:
                n_warm = min(config.warm_start_samples, X_track.shape[-1])
                tracker.warm_start(X_track[tr, :, :n_warm])
            active_k = 1 if config.adaptive_k else config.rank
        for t in range(n_steps):
            y = tracker.update(X_track[tr, :, t], project_x=X_project[tr, :, t], track_delta=True) if isinstance(tracker, PASTd) else tracker.update(X_track[tr, :, t], project_x=X_project[tr, :, t])
            if isinstance(tracker, PASTd):
                delta_u[tr, t] = tracker.delta_u
                delta_u_ema[tr, t] = tracker.delta_u_ema
                if config.adaptive_k and t > 0 and t % config.adaptive_k_check_period == 0:
                    active_k = choose_k_from_delta_ema(tracker.delta_u_ema, config.beta, config.adaptive_k_threshold)
            U = tracker.get_components()
            chosen_k[tr, t] = active_k
            u_trace[tr, t] = U
            X_past[tr, :, t] = X_project[tr, :, t] - U[:, :active_k] @ y[:active_k]
            if ideal_u is not None:
                n_angle = min(active_k, ideal_u.shape[1])
                angle_trace[tr, t, :n_angle] = subspace_angle_deg(U[:, :n_angle], ideal_u[:, :n_angle])

    stages["after_subspace"] = X_past
    if config.harmonic_filter:
        X_out = harmonic_rls_3d(
            X_past,
            fs=fs,
            stim_rate=float(dataset.stim_rate),
            max_harmonics=config.harmonic_max_harmonics,
            settling_time=config.harmonic_settling_time,
        )
        stages["after_harmonic"] = X_out
    else:
        X_out = X_past
    # Intermediate stages retain processing units; final output restores input units.
    time_offset = config.derivative_order if config.use_derivative_for_tracking else 0
    X_out = denormalize_channels(X_out, norm_stats, time_offset=time_offset)
    stages["cleaned"] = X_out

    diagnostics: Dict[str, np.ndarray | float | dict] = {
        "chosen_k": chosen_k,
        "delta_u": delta_u,
        "delta_u_ema": delta_u_ema,
        "u_trace": u_trace,
        "angle_trace_deg": angle_trace,
        "normalization": norm_stats.__dict__,
        "harmonic_ratio_before": harmonic_energy_ratio(X_track, fs, dataset.stim_rate, config.harmonic_max_harmonics),
        "harmonic_ratio_after": harmonic_energy_ratio(X_out, fs, dataset.stim_rate, config.harmonic_max_harmonics),
    }
    if ideal_u is not None:
        diagnostics["ideal_u"] = ideal_u
    if y_clean is not None:
        diagnostics["y_clean"] = y_clean[:, :, config.derivative_order:] if config.use_derivative_for_tracking else y_clean

    result = PipelineResult(dataset=dataset, config=config, stages=stages, diagnostics=diagnostics)
    if config.save_npz is not None:
        save_result_npz(result, config.save_npz)
    return result


def save_result_npz(result: PipelineResult, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {f"stage_{k}": v for k, v in result.stages.items()}
    payload.update({f"diag_{k}": v for k, v in result.diagnostics.items() if isinstance(v, np.ndarray)})
    payload["fs"] = np.array(result.dataset.fs)
    payload["stim_rate"] = np.array(result.dataset.stim_rate)
    payload["dataset_name"] = np.array(result.dataset.name)
    np.savez(path, **payload)

