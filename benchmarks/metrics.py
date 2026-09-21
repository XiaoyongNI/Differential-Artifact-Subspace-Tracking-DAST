"""Rest-period similarity and counterfactual neural-preservation metrics."""
from __future__ import annotations

import numpy as np
from scipy.signal import welch

from algorithms import (apply_lpf_3d, apply_bpf_3d,
                        covariance_aware_subspace_suppression)
from .methods import linear_interpolation, pulse, low_rank_tv


def _pearson(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.size < 3 or a.shape != b.shape or not (np.isfinite(a).all() and np.isfinite(b).all()):
        return np.nan
    a, b = a-a.mean(), b-b.mean()
    denominator = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.clip(a@b/denominator, -1, 1)) if denominator > 0 else np.nan


def _finite_mean(values):
    values = np.asarray(values)
    return float(np.mean(values[np.isfinite(values)])) if np.isfinite(values).any() else np.nan


def rest_correlations(output, rest, fs):
    """Mean per-trial/channel Pearson r, plus individual values.

    Time: pair interval starts and truncate to the shorter signal, without lag
    optimization. Different periods are similarity comparisons, not accuracy.
    Frequency: linear Welch PSD with identical bins, a 20 ms Hann segment (or
    shorter if necessary), 50% overlap, constant detrending and DC excluded.
    Rest trials are paired if counts match, or one shared rest trial is reused.
    Otherwise time r is unavailable; frequency r uses a pooled mean rest PSD.
    """
    output, rest = np.asarray(output, dtype=float), np.asarray(rest, dtype=float)
    if output.ndim != 3 or rest.ndim != 3 or output.shape[1] != rest.shape[1]:
        raise ValueError('Output/rest must be trials-by-matching-channels-by-time')
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError('fs must be positive and finite')
    time_r = np.full(output.shape[:2], np.nan)
    freq_r = np.full_like(time_r, np.nan)
    n = min(output.shape[-1], rest.shape[-1])
    if n < 3 or rest.shape[0] == 0:
        return {'rest_correlation_time': np.nan, 'rest_correlation_freq': np.nan}, time_r, freq_r
    nperseg = min(n, max(4, int(round(.02*fs))))
    _, rest_psd = welch(rest, fs=fs, nperseg=nperseg, axis=-1)
    _, out_psd = welch(output, fs=fs, nperseg=nperseg, axis=-1)
    paired = rest.shape[0] == output.shape[0] or rest.shape[0] == 1
    for tr in range(output.shape[0]):
        rest_tr = tr if rest.shape[0] == output.shape[0] else 0
        for ch in range(output.shape[1]):
            if paired:
                time_r[tr, ch] = _pearson(output[tr, ch, :n], rest[rest_tr, ch, :n])
                spectrum = rest_psd[rest_tr, ch, 1:]
            else:
                spectrum = rest_psd[:, ch, 1:].mean(axis=0)
            freq_r[tr, ch] = _pearson(out_psd[tr, ch, 1:], spectrum)
    return {'rest_correlation_time': _finite_mean(time_r),
            'rest_correlation_freq': _finite_mean(freq_r)}, time_r, freq_r


def preprocess_rest(rest, method, config, fs):
    """Use the same input filtering as a PAST run, without differentiation."""
    rest = np.asarray(rest, dtype=float)
    if rest.ndim != 3 or not np.isfinite(rest).all():
        raise ValueError('Rest must be a finite trials-by-channels-by-time array')
    if method.startswith('past'):
        if config.preprocessing == 'lpf':
            return apply_lpf_3d(rest, config.lpf_cutoff_hz, fs, order=config.lpf_order)
        if config.preprocessing == 'bpf':
            return apply_bpf_3d(rest, config.bpf_low_hz, config.bpf_high_hz, fs, order=config.bpf_order)
    return rest.copy()


def _normalization_at_end(result, trial):
    stats = result.diagnostics['normalization']
    if stats['mode'] == 'none':
        return 0., 1.
    if stats['mode'] == 'global_per_channel':
        return stats['mean'][0, :, 0, None], stats['scale'][0, :, 0, None]
    if stats['mode'] == 'per_trial':
        return stats['mean'][trial, 0, 0], stats['scale'][trial, 0, 0]
    if stats['mode'] == 'streaming_causal':
        return stats['debug']['mu_hist'][trial, :, -1, None], stats['debug']['scale_hist'][trial, :, -1, None]
    raise ValueError('Unsupported normalization for frozen rest evaluation')


def clean_rest_output(rest, method, result, config, fs, stim_times=None):
    """Apply stimulation-learned removal to rest without relearning PAST U/Rx.

    PAST freezes each trial's final U, k, channel scales and (if used) Rn/Rx.
    Only the linear action on neural signal is evaluated: the stimulation mean
    is not subtracted/added to a separate resting recording. Window methods
    preserve each clean window's own channel means.
    Window SVD/ICA reuse fitted components at the same relative sample windows;
    template subtraction replays the estimated template correction at those
    positions. Dictionary learning likewise replays the selected and scaled
    stimulation correction, without learning or matching again on rest.
    Interpolation uses the same hypothetical pulse positions. These
    are counterfactual removal tests: the rest recording has no actual pulses.
    LowRankTV has no fitted model state, so its solver runs directly on rest.
    A shared rest trial may be reused; unmatched independent trial counts cannot
    be paired with learned trial-specific states and return no energy metric.
    """
    rest = np.asarray(rest, dtype=float)
    n_trials = (result.stages['raw'].shape[0] if method.startswith('past') else result.cleaned.shape[0])
    if rest.shape[0] not in (1, n_trials):
        return None, None
    reference = np.broadcast_to(rest, (n_trials, *rest.shape[1:])).copy()
    cleaned = reference.copy()
    if method.startswith('past'):
        for tr in range(n_trials):
            U = result.diagnostics['u_trace'][tr, -1]
            k = int(result.diagnostics['chosen_k'][tr, -1])
            _, scale = _normalization_at_end(result, tr)
            # Test the linear removal on the clean neural component itself.
            # Replaying the stimulation mean as an affine offset would inject
            # a stimulation-dependent DC correction into a resting recording.
            normalized = reference[tr]/scale
            if config.removal_method == 'covariance':
                if result.diagnostics['covariance']['source'] == 'bypass':
                    output = normalized
                else:
                    output = covariance_aware_subspace_suppression(
                        normalized, U[:, :k], result.diagnostics['neural_covariance_final'][tr],
                        result.diagnostics['mixed_covariance_final'][tr],
                        config.covariance_lambda, config.covariance_eps)
            else:
                active_U = U[:, :k]
                if k == 1:
                    _, projection_cleaned = clean_energy_loss(normalized[None], active_U[:, 0][None])
                    output = projection_cleaned[0]
                else:
                    output = normalized-active_U @ (active_U.T @ normalized)
            cleaned[tr] = output*scale
    elif method == 'asar':
        from .methods import asar_model
        for tr in range(n_trials):
            model = asar_model(result.diagnostics['mean'][tr], result.diagnostics['std'][tr],
                               config, weights=result.diagnostics['weights'][tr])
            for t in range(reference.shape[-1]):
                cleaned[tr, :, t] = model.process_sample(reference[tr, :, t], adapt=False)
    elif method == 'lrr':
        from .methods import lrr
        cleaned = lrr(reference, W=result.diagnostics['W']).cleaned
    elif method == 'svd_template_subtraction':
        _, after_svd = clean_rest_output(reference, 'window_svd', result.diagnostics['svd'], config, fs)
        _, cleaned = clean_rest_output(after_svd, 'template_subtraction',
                                       result.diagnostics['template_subtraction'], config, fs)
    elif method == 'linear_interpolation':
        times = [np.asarray(row)[np.asarray(row) < reference.shape[-1]] for row in stim_times]
        cleaned = linear_interpolation(reference, times, fs, config).cleaned
    elif method in ('template_subtraction', 'dictionary_learning'):
        n = min(reference.shape[-1], result.artifact.shape[-1])
        cleaned[:, :, :n] -= result.artifact[:, :, :n]
    elif method in ('window_svd', 'window_ica'):
        for component in result.diagnostics['components']:
            tr = component['trial']
            start, end = component['window']
            end = min(end, reference.shape[-1])
            if end <= start:
                continue
            epoch = reference[tr, :, start:end]
            if method == 'window_svd':
                U = component['removal_basis']
                removed = U @ (U.T @ (epoch-epoch.mean(axis=1, keepdims=True)))
            else:
                removed = component['removal_operator'] @ (epoch-epoch.mean(axis=1, keepdims=True))
            cleaned[tr, :, start:end] = epoch-removed
    elif method == 'pulse':
        # Replay the stimulation trace/mask on rest with frozen network and
        # training normalization; crop/zero-pad to the reference duration.
        if reference.shape[-1] < 16:
            return None, None
        trace = np.zeros((n_trials, 1, reference.shape[-1]), dtype=np.float32)
        mask = np.zeros_like(trace)
        n = min(reference.shape[-1], result.cleaned.shape[-1])
        trace[..., :n] = result.diagnostics['stim_trace'][..., :n]
        mask[..., :n] = result.diagnostics['artifact_mask'][..., :n]
        cleaned = pulse(reference, fs=fs, config=config, model=result.diagnostics['model'],
                        stim_trace=trace, artifact_mask=mask).cleaned
    elif method == 'low_rank_tv':
        cleaned = low_rank_tv(reference, config=config).cleaned
    else:
        raise ValueError(f'No clean/rest evaluation for {method}')
    return reference, cleaned


ENERGY_COLUMNS = ("trial", "original_energy", "removed_energy", "residual_energy",
                  "energy_loss_fraction", "energy_loss_db")


def _energy_row(trial, x, projected, x_cleaned):
    original_energy = float(np.sum(x**2))
    removed_energy = float(np.sum(projected**2))
    residual_energy = float(np.sum(x_cleaned**2))
    return {
        "trial": trial,
        "original_energy": original_energy,
        "removed_energy": removed_energy,
        "residual_energy": residual_energy,
        "energy_loss_fraction": removed_energy / original_energy if original_energy > 0 else np.nan,
        "energy_loss_db": 10.0 * np.log10(residual_energy / original_energy)
        if original_energy > 0 and residual_energy > 0 else np.nan,
    }


def clean_energy_loss(clean, u1s):
    """Same per-trial hard-projection calculation as the clean-loss script."""
    n_trials = min(clean.shape[0], u1s.shape[0])
    rows = []
    cleaned = np.zeros_like(clean[:n_trials], dtype=float)
    for trial_idx in range(n_trials):
        x = clean[trial_idx].astype(float, copy=False)
        u1 = u1s[trial_idx]
        projected = np.outer(u1, u1 @ x)
        x_cleaned = x - projected
        cleaned[trial_idx] = x_cleaned
        rows.append(_energy_row(trial_idx, x, projected, x_cleaned))
    return rows, cleaned


def clean_energy_loss_from_output(clean, cleaned):
    """The same energy fields for covariance/other removal outputs on clean data.

    The removed component is clean - cleaned; summary averages trial fractions,
    rather than weighting trials by their original energy.
    """
    clean, cleaned = np.asarray(clean, dtype=float), np.asarray(cleaned, dtype=float)
    if clean.ndim != 3 or clean.shape != cleaned.shape:
        raise ValueError('Clean/rest reference and cleaned arrays must align as trials/channels/time')
    if not (np.isfinite(clean).all() and np.isfinite(cleaned).all()):
        raise ValueError('Clean/rest energy evaluation requires finite signals')
    rows = [_energy_row(tr, clean[tr], clean[tr]-cleaned[tr], cleaned[tr])
            for tr in range(clean.shape[0])]
    return rows, cleaned


def summarize_clean_energy(rows):
    return {"energy_loss_fraction": _finite_mean([row["energy_loss_fraction"] for row in rows]),
            "energy_loss_db": _finite_mean([row["energy_loss_db"] for row in rows])}
