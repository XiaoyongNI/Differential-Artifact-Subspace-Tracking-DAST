"""Detect ERAASR pulse onsets, save benchmark-ready indices, and plot one trial.

Run from the project root:
    python label_eraasr_triggers.py --trial 0
    python -m benchmarks.run --dataset ERAASR --methods past --compare-removal \
        --stim-times plots/eraasr_triggers/triggers.npy

Labels are signal-derived onset estimates, not acquisition hardware timestamps.
Indices refer to load_eraasr().X (original full-record sample offset: 1600).
No labels are inferred for unobserved/missing pulses or the cropped left edge.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import warnings

import numpy as np
from scipy.signal import find_peaks

from data_loading import ROOT, load_eraasr


ERAASR_CROP_START = 1600


def detect_pulse_triggers(
    signals, fs, stim_rate, channel=None, height_sigma=6., onset_sigma=3.,
    min_spacing_fraction=.7, onset_lookback_fraction=.35, interval_tolerance=.15,
):
    """Detect pulse peaks and then label their leading edges independently per trial.

    A channel median of abs(diff(x)) rejects isolated single-channel transients.
    Optionally use one chosen channel instead. The derivative at sample t means
    x[t]-x[t-1], so index t is aligned with the original waveform. Robust score
    median/MAD set a high peak threshold and lower leading-edge threshold.
    Peak spacing and the bounded leading-edge search use the known pulse period.
    This is an offline heuristic: irregular intervals are flagged, never filled
    with a fabricated periodic schedule. The waveform and returned scores are
    otherwise unfiltered and unnormalized.
    """
    x = np.asarray(signals, dtype=float)
    if x.ndim != 3 or x.shape[-1] < 3 or min(x.shape[:2]) < 1 or not np.isfinite(x).all():
        raise ValueError("signals must be finite (trials, channels, time), with at least 3 samples")
    if not np.isfinite([fs, stim_rate]).all() or not 0 < stim_rate < fs:
        raise ValueError("Require 0 < stim_rate < fs")
    if channel is not None and (not isinstance(channel, (int, np.integer)) or not 0 <= channel < x.shape[1]):
        raise ValueError("channel must be a valid zero-based channel index")
    values = [height_sigma, onset_sigma, min_spacing_fraction, onset_lookback_fraction, interval_tolerance]
    if not np.isfinite(values).all() or not 0 < onset_sigma <= height_sigma:
        raise ValueError("Require finite parameters and 0 < onset_sigma <= height_sigma")
    if not 0 < min_spacing_fraction <= 1 or not 0 < onset_lookback_fraction < min_spacing_fraction or not 0 < interval_tolerance < 1:
        raise ValueError("Invalid spacing, lookback, or interval tolerance fractions")
    period = fs / stim_rate
    distance = max(1, int(np.ceil(min_spacing_fraction * period)))
    lookback = max(1, int(round(onset_lookback_fraction * period)))
    triggers, diagnostics, scores = [], [], []
    for trial, signal in enumerate(x):
        derivative = np.abs(np.diff(signal, axis=-1, prepend=signal[:, :1]))
        score = np.median(derivative, axis=0) if channel is None else derivative[channel]
        median = float(np.median(score))
        mad = float(1.4826 * np.median(np.abs(score - median)))
        # A nonzero numerical floor also supports noiseless synthetic signals.
        sigma = max(mad, np.finfo(float).eps * max(1., float(score.max())))
        high, low = median + height_sigma*sigma, median + onset_sigma*sigma
        peaks, _ = find_peaks(score, height=high, prominence=onset_sigma*sigma, distance=distance)
        onsets = []
        for peak in peaks:
            start = max(1, int(peak)-lookback)
            crossing = np.flatnonzero(score[start:peak+1] > low)
            # A high peak necessarily exceeds the lower threshold. Search the
            # entire bounded region: the strongest edge may be a later phase
            # of a biphasic pulse, with intervening low-gradient samples.
            onsets.append(start + int(crossing[0]))
        onsets = np.unique(np.asarray(onsets, dtype=int))
        intervals = np.diff(onsets)
        regular = float(np.mean(np.abs(intervals-period) <= interval_tolerance*period)) if intervals.size else None
        if len(onsets) < 2:
            warnings.warn(f"Trial {trial}: fewer than two pulse onsets detected", RuntimeWarning)
        elif regular < .8:
            warnings.warn(f"Trial {trial}: only {regular:.0%} of intervals agree with stimulation rate; inspect labels", RuntimeWarning)
        diagnostics.append({"trial": trial, "n_triggers": len(onsets), "score_median": median,
                            "score_mad_sigma": mad, "peak_threshold": high, "onset_threshold": low,
                            "regular_interval_fraction": regular,
                            "median_interval_samples": float(np.median(intervals)) if intervals.size else None,
                            "peak_samples": peaks.tolist()})
        triggers.append(onsets)
        scores.append(score)
    return triggers, np.stack(scores), diagnostics


def plot_trial(signals, triggers, score, diagnostic, fs, stim_rate, trial, channel, output):
    """Plot full crop, a three-pulse zoom, and aligned detection score."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    x = signals[trial, channel]
    t = np.arange(x.size)*1000/fs
    labels = triggers[trial]
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), constrained_layout=True)
    axes[0].plot(t, x, lw=.8, color='#24547a', label=f'Channel {channel}')
    axes[0].scatter(t[labels], x[labels], color='#c63530', s=22, zorder=4, label='Estimated pulse onset')
    for label in labels:
        axes[0].axvline(t[label], color='#c63530', lw=.6, alpha=.4)
    axes[0].set_title(f'ERAASR trial {trial}: {len(labels)} detected pulse onsets; expected spacing {1000/stim_rate:.2f} ms')
    axes[0].set_ylabel('Signal (loaded units)')
    axes[0].set_xlabel('Time relative to loader crop (ms)')
    axes[0].legend(loc='upper right')

    period = fs/stim_rate
    left = max(0, int(labels[0]-period*.35)) if labels.size else 0
    right = min(x.size, left+int(3.5*period))
    axes[1].plot(t[left:right], x[left:right], lw=1, color='#24547a')
    axes[2].plot(t[left:right], score[left:right], lw=1, color='#24547a', label='Absolute derivative score')
    axes[2].axhline(diagnostic['peak_threshold'], color='#8758a5', linestyle='--', label='Peak threshold')
    axes[2].axhline(diagnostic['onset_threshold'], color='#c98c27', linestyle='--', label='Onset threshold')
    for label in labels[(labels >= left) & (labels < right)]:
        axes[1].axvline(t[label], color='#c63530', lw=1, alpha=.7)
        axes[1].scatter(t[label], x[label], color='#c63530', s=30, zorder=4)
        axes[1].annotate(str(label), (t[label], x[label]), xytext=(4, 12), textcoords='offset points', color='#c63530')
        axes[2].axvline(t[label], color='#c63530', lw=1, alpha=.7)
    axes[1].set_title('Leading-edge alignment: labels show zero-based cropped-signal sample indices')
    axes[1].set_ylabel('Signal (loaded units)')
    axes[2].set_ylabel('Derivative score')
    axes[2].set_xlabel('Time relative to loader crop (ms)')
    axes[2].legend(loc='upper right')
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(output.with_suffix('.png'), dpi=180)
    fig.savefig(output.with_suffix('.pdf'))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--trial', type=int, default=0, help='Zero-based trial to plot; all trials are labeled')
    parser.add_argument('--plot-channel', type=int, default=15)
    parser.add_argument('--detection-channel', type=int, default=None, help='Default: channel-median score')
    parser.add_argument('--height-sigma', type=float, default=6.)
    parser.add_argument('--onset-sigma', type=float, default=3.)
    parser.add_argument('--min-spacing-fraction', type=float, default=.7)
    parser.add_argument('--onset-lookback-fraction', type=float, default=.35)
    parser.add_argument('--interval-tolerance', type=float, default=.15)
    parser.add_argument('--out-dir', type=Path, default=Path('plots/eraasr_triggers'))
    args = parser.parse_args()
    dataset = load_eraasr(args.root)
    if not 0 <= args.trial < len(dataset.X) or not 0 <= args.plot_channel < dataset.X.shape[1]:
        parser.error('Trial or plot channel out of range')
    triggers, scores, diagnostics = detect_pulse_triggers(
        dataset.X, dataset.fs, dataset.stim_rate, args.detection_channel,
        args.height_sigma, args.onset_sigma, args.min_spacing_fraction,
        args.onset_lookback_fraction, args.interval_tolerance,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # JSON retains ragged/missing detections without object arrays or padding.
    metadata = {'dataset': dataset.name, 'fs': dataset.fs, 'stim_rate': dataset.stim_rate,
                'crop_start_original_sample': ERAASR_CROP_START, 'parameters': {
                    key: value for key, value in vars(args).items() if key not in ('root', 'out_dir')},
                'method': 'absolute_derivative_peak_then_leading_edge',
                'triggers': [row.tolist() for row in triggers], 'diagnostics': diagnostics}
    (args.out_dir/'triggers.json').write_text(json.dumps(metadata, indent=2)+'\n')
    counts = np.array([len(row) for row in triggers])
    if np.all(counts == counts[0]) and counts[0] > 0:
        np.save(args.out_dir/'triggers.npy', np.stack(triggers))
        print(f'Benchmark-ready triggers: {args.out_dir / "triggers.npy"} ({len(triggers)} trials x {counts[0]} pulses)')
    else:
        warnings.warn('Unequal/empty trigger counts: only JSON and CSV saved; use JSON triggers through Python API, not a shared pulse schedule', RuntimeWarning)
    with (args.out_dir/'triggers.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['trial', 'pulse', 'sample_crop', 'sample_original', 'time_crop_ms'])
        for trial, row in enumerate(triggers):
            for pulse, sample in enumerate(row):
                writer.writerow([trial, pulse, sample, sample+ERAASR_CROP_START, sample*1000/dataset.fs])
    output = args.out_dir/f'eraasr_trial_{args.trial}_triggers'
    plot_trial(dataset.X, triggers, scores[args.trial], diagnostics[args.trial], dataset.fs,
               dataset.stim_rate, args.trial, args.plot_channel, output)
    print(f'Trial {args.trial} triggers: {triggers[args.trial].tolist()}')
    print(f'Pulse counts: {counts.tolist()}')
    print(f'Plot: {output.with_suffix(".png")}')


if __name__ == '__main__':
    main()
