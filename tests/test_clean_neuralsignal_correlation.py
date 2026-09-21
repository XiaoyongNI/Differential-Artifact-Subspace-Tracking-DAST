"""Time-domain Pearson correlation of ERAASR clean signals between trials.

Run from the project root:
    python tests/test_clean_neuralsignal_correlation.py
    python tests/test_clean_neuralsignal_correlation.py --lpf-cutoff 500

Compare the same channel and prestimulation sample indices in different trials.
No lag search, stimulation samples, artifact removal, or cross-channel flattening
is used. Constant trial/channel waveforms have undefined (NaN) correlations.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

# Also support direct execution from tests/ without installing the project.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from algorithms import apply_lpf_3d
from data_loading import ROOT, load_eraasr


def trial_time_correlations(clean):
    """Return (channel x trial x trial) Pearson matrices and channel means."""
    clean = np.asarray(clean, dtype=float)
    if clean.ndim != 3 or clean.shape[0] < 2 or clean.shape[1] < 1 or clean.shape[2] < 3:
        raise ValueError('Need at least two trials, one channel and three clean time samples')
    if not np.isfinite(clean).all():
        raise ValueError('Clean neural signals must be finite')
    centered = clean-clean.mean(axis=-1, keepdims=True)
    norm = np.linalg.norm(centered, axis=-1, keepdims=True)
    normalized = np.divide(centered, norm, out=np.full_like(centered, np.nan), where=norm > 0)
    correlation = np.clip(np.einsum('ict,jct->cij', normalized, normalized, optimize=True), -1., 1.)
    valid = np.isfinite(correlation)
    mean = np.divide(np.where(valid, correlation, 0.).sum(axis=0), valid.sum(axis=0),
                     out=np.full(correlation.shape[1:], np.nan), where=valid.sum(axis=0) > 0)
    return correlation, mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--max-trials', type=int, default=None)
    parser.add_argument('--lpf-cutoff', type=float, default=None, help='Optional LPF in Hz; default uses raw rest data')
    parser.add_argument('--lpf-order', type=int, default=5)
    parser.add_argument('--out-dir', type=Path, default=Path('plots/clean_neuralsignal_correlation'))
    args = parser.parse_args()
    dataset = load_eraasr(args.root)
    clean = dataset.baseline if dataset.baseline is not None else dataset.y_clean
    if clean is None:
        parser.error('ERAASR loader did not provide a clean/rest period')
    if args.max_trials is not None:
        if args.max_trials < 2:
            parser.error('--max-trials must be at least two')
        clean = clean[:args.max_trials]
    if args.lpf_cutoff is not None:
        if not np.isfinite(args.lpf_cutoff) or not 0 < args.lpf_cutoff < dataset.fs/2 or args.lpf_order < 1:
            parser.error('Require 0 < lpf-cutoff < fs/2 and a positive filter order')
        clean = apply_lpf_3d(clean, args.lpf_cutoff, dataset.fs, order=args.lpf_order)
    correlation, mean = trial_time_correlations(clean)
    pairs_i, pairs_j = np.triu_indices(clean.shape[0], k=1)
    values = correlation[:, pairs_i, pairs_j]
    finite_values = values[np.isfinite(values)]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = 'ERAASR_clean_trial_time_correlation'
    source = (dataset.metadata or {}).get('baseline_source', 'ERAASR clean/rest signal')
    np.savez_compressed(args.out_dir/f'{prefix}.npz',
                        correlation_per_channel=correlation, mean_correlation=mean,
                        trial_indices=np.arange(clean.shape[0]), channel_indices=np.arange(clean.shape[1]),
                        pair_trial_i=pairs_i, pair_trial_j=pairs_j, pair_correlations=values,
                        fs=dataset.fs, clean_source=source, n_samples=clean.shape[-1],
                        lpf_cutoff=np.nan if args.lpf_cutoff is None else args.lpf_cutoff, lpf_order=args.lpf_order)
    with (args.out_dir/f'{prefix}_pairs.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['trial_i', 'trial_j', 'channel', 'time_correlation'])
        for i, j in zip(pairs_i, pairs_j):
            for ch in range(clean.shape[1]):
                writer.writerow([i, j, ch, correlation[ch, i, j]])
    with (args.out_dir/f'{prefix}_summary.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['trial_i', 'trial_j', 'mean_time_correlation', 'mean_abs_time_correlation', 'valid_channels'])
        for i, j in zip(pairs_i, pairs_j):
            channel_values = correlation[:, i, j]
            valid = channel_values[np.isfinite(channel_values)]
            writer.writerow([i, j, mean[i, j], float(np.abs(valid).mean()) if valid.size else np.nan, valid.size])

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    heatmap = ax.imshow(mean, vmin=-1, vmax=1, cmap='RdBu_r', origin='lower')
    ax.set(xlabel='Trial index (zero-based)', ylabel='Trial index (zero-based)',
           title='ERAASR clean/rest: mean time correlation across channels')
    fig.colorbar(heatmap, ax=ax, label='Pearson r')
    fig.savefig(args.out_dir/f'{prefix}.png', dpi=180)
    fig.savefig(args.out_dir/f'{prefix}.pdf')
    plt.close(fig)
    print(f'Clean source: {source}; shape={clean.shape}; fs={dataset.fs:g} Hz')
    print(f'Preprocessing: {"raw" if args.lpf_cutoff is None else f"LPF {args.lpf_cutoff:g} Hz, order {args.lpf_order}"}')
    print(f'Compared {len(pairs_i)} distinct trial pairs x {clean.shape[1]} channels; no lag search')
    if finite_values.size:
        print(f'Time Pearson r (off-diagonal only): mean={finite_values.mean():.6f}, '
              f'median={np.median(finite_values):.6f}, mean(abs)={np.abs(finite_values).mean():.6f}')
    else:
        print('Time Pearson r: undefined for all trial/channel pairs')
    print(f'Results: {args.out_dir}')


if __name__ == '__main__':
    main()
