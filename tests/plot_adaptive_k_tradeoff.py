"""Artifact suppression versus neural preservation for fixed and adaptive k.

python tests/plot_adaptive_k_tradeoff.py --dataset all

One PASTd run (rank k_max, first differences, no normalization) per recording
drives every condition. The removal (manuscript Algorithm 1, run_pipeline)
    Q = orth(U[:, :k]);  out(t) = x(t) - Q Q^T x(t)
is linear in x(t), so it is applied separately to the known neural n(t) and
artifact a(t) components. PASTd deflation is sequential, so the first
k tracked vectors of a rank-k_max tracker equal those of a rank-k tracker, and
QR of U_new spans the same nested prefixes: fixed k = 0..k_max and the
adaptive rule all share one tracker trajectory per recording.

Adaptive k follows run_pipeline: start at k=1, re-evaluate every
--check-period samples with --rule: 'stability_gap' (default; largest jump in
log instability between consecutive components) or 'relative' (the original
(ema_j - ema_1)/(1-beta) < threshold test).

Metrics per recording, pooled over channels and evaluation samples:
    artifact suppression (dB) = 10 log10(sum a^2 / sum r_a^2)
    neural distortion (%)     = 100 sum (n_out - n)^2 / sum n^2
    output SNR (dB)           = 10 log10(sum n^2 / sum (out - n)^2)
Recordings: SWEC + synthetic trials (cached crop) and PBS configurations
0-12 (Config. 13 has no stimulation). Summaries are median [Q1, Q3].
"""
from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from algorithms import PASTd, choose_k_from_delta_ema, choose_k_from_stability_gap
from data_loading import ROOT, load_swec_pbs
from plotting import set_tbme_style
from plot_stable_low_dim_artifact_subspace import load_pbs_artifact

DATASET_LABELS = {'swec_synthetic': 'SWEC+Synthetic', 'swec_pbs': 'SWEC+PBS'}
FIXED_COLOR, ADAPTIVE_COLOR = '#0072B2', '#D55E00'


def tradeoff_energies(neural, artifact, k_max, beta, threshold, check_period, skip, rule='relative',
                      warm_start=0):
    """Accumulate energies for k = 0..k_max (columns 0..k_max) and adaptive k.

    neural, artifact: channels-by-time arrays of one recording. Sample 0 only
    seeds the first difference, as in run_pipeline. warm_start > 0 mirrors the
    pipeline's causal SVD re-initialization after that many samples. Returns a
    dict of energy arrays and the adaptive k trace.
    """
    n = np.asarray(neural, dtype=float)
    a = np.asarray(artifact, dtype=float)
    x = n + a
    channels, n_time = x.shape
    if k_max < 1 or k_max > channels:
        raise ValueError('Require 1 <= k_max <= channels')
    tracker = PASTd(channels, k_max, beta=beta)
    dx = np.diff(x, axis=-1)
    columns = k_max + 2  # fixed k = 0..k_max, then adaptive
    energy = {name: np.zeros(columns) for name in ('artifact_residual', 'neural_error', 'output_error')}
    energy['artifact_input'] = 0.0
    energy['neural_input'] = 0.0
    chosen_k = np.empty(dx.shape[1], dtype=int)
    active_k = 1
    gap_ema = np.zeros(k_max)
    ema_start = 0
    zeros = np.zeros((channels, 1))
    for t in range(dx.shape[1]):
        if 0 < warm_start < dx.shape[1] and t == warm_start:
            tracker.warm_start(dx[:, :t])
            tracker.delta_u_ema[:] = 0.0
            gap_ema = np.zeros(k_max)
            ema_start = t
        tracker.update(dx[:, t])
        if t > ema_start:
            gap_ema = beta * gap_ema + (1 - beta) * tracker.delta_u
        if t > 0 and t % check_period == 0:
            if rule == 'stability_gap':
                if t > ema_start + 1:
                    active_k = choose_k_from_stability_gap(gap_ema)
            else:
                active_k = choose_k_from_delta_ema(tracker.delta_u_ema, beta, threshold)
        chosen_k[t] = active_k
        if t < skip:
            continue
        n_t, a_t = n[:, t + 1], a[:, t + 1]
        Q = np.linalg.qr(tracker.U)[0]
        # Cumulative removed components for k = 0..k_max.
        removed_n = np.hstack([zeros, np.cumsum(Q * (Q.T @ n_t), axis=1)])
        removed_a = np.hstack([zeros, np.cumsum(Q * (Q.T @ a_t), axis=1)])
        removed_n = np.hstack([removed_n, removed_n[:, active_k:active_k + 1]])
        removed_a = np.hstack([removed_a, removed_a[:, active_k:active_k + 1]])
        residual_a = a_t[:, None] - removed_a
        energy['artifact_residual'] += np.sum(residual_a ** 2, axis=0)
        energy['neural_error'] += np.sum(removed_n ** 2, axis=0)
        energy['output_error'] += np.sum((residual_a - removed_n) ** 2, axis=0)
        energy['artifact_input'] += float(a_t @ a_t)
        energy['neural_input'] += float(n_t @ n_t)
    energy['chosen_k'] = chosen_k
    return energy


def metrics_from_energies(energy):
    with np.errstate(divide='ignore'):
        return dict(
            artifact_suppression_db=10 * np.log10(energy['artifact_input'] / energy['artifact_residual']),
            neural_distortion_pct=100 * energy['neural_error'] / energy['neural_input'],
            output_snr_db=10 * np.log10(energy['neural_input'] / energy['output_error']),
        )


def load_synthetic(index, args):
    meta = json.loads((args.cache_dir / 'source.json').read_text())
    clean = np.load(args.cache_dir / 'clean.npy', mmap_mode='r')
    artifacts = np.load(args.cache_dir / 'artifacts.npy', mmap_mode='r')
    neural = np.array(clean[index], dtype=float)
    artifact = sum(np.array(artifacts[meta['stim_channels'].index(c), index], dtype=float)
                   for c in args.stim_channels)
    if args.synthetic_rms_ratio is not None:
        artifact *= args.synthetic_rms_ratio * np.sqrt(np.mean(neural ** 2) / np.mean(artifact ** 2))
    return neural, artifact, float(meta['fs'])


def load_pbs(args):
    """Same construction as tests/run_pbs_dast_table.py: one clean reference
    (separated SWEC channels), isolated PBS artifacts, common available interval."""
    base = load_swec_pbs(root=args.root, configuration_number=0,
                         artifact_scale=args.pbs_artifact_scale, separated_recording_channels=True)
    clean = base.y_clean[0]
    artifacts = {c: args.pbs_artifact_scale * load_pbs_artifact(args.root, c).X[0]
                 for c in args.pbs_configurations}
    length = min(clean.shape[-1], *(a.shape[-1] for a in artifacts.values()))
    return {c: (clean[:, :length], a[:, :length], float(base.fs)) for c, a in artifacts.items()}


def run_one(job):
    dataset, index, payload, args = job
    neural, artifact, fs = payload if payload is not None else load_synthetic(index, args)
    k_max = min(args.k_max, neural.shape[0])
    energy = tradeoff_energies(neural, artifact, k_max, args.beta, args.threshold,
                               args.check_period, int(round(args.skip_ms * 1e-3 * fs)), args.rule)
    metrics = metrics_from_energies(energy)
    chosen = energy['chosen_k']
    return dict(dataset=dataset, recording=index, fs=fs, k_max=k_max,
                final_adaptive_k=int(chosen[-1]),
                mean_adaptive_k=float(chosen.mean()),
                **{key: value.tolist() for key, value in metrics.items()})


def summarize(rows, k_max):
    """Median and quartiles across recordings for each condition column."""
    summary = {}
    for key in ('artifact_suppression_db', 'neural_distortion_pct', 'output_snr_db'):
        values = np.array([r[key][:k_max + 2] for r in rows], dtype=float)
        summary[key] = np.nanpercentile(values, [25, 50, 75], axis=0)
    summary['final_adaptive_k'] = np.percentile([r['final_adaptive_k'] for r in rows], [25, 50, 75])
    return summary


def plot(summaries, k_max, path_stem):
    set_tbme_style()
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'stix', 'font.size': 9,
                         'axes.labelsize': 9, 'legend.fontsize': 8, 'grid.alpha': .18,
                         'grid.linewidth': .5, 'pdf.fonttype': 42})
    names = list(summaries)
    fig, axes = plt.subplots(2, len(names), figsize=(3.5 * len(names), 4.6), squeeze=False)
    ks = np.arange(k_max + 1)
    for col, name in enumerate(names):
        s = summaries[name]
        dist, supp, snr = s['neural_distortion_pct'], s['artifact_suppression_db'], s['output_snr_db']
        ax = axes[0, col]
        ax.errorbar(dist[1, :-1], supp[1, :-1],
                    xerr=np.abs(dist[[0, 2], :-1] - dist[1, :-1]),
                    yerr=np.abs(supp[[0, 2], :-1] - supp[1, :-1]),
                    color=FIXED_COLOR, marker='o', ms=4, lw=1.2, elinewidth=.6, capsize=1.5,
                    label='Fixed $k$')
        for k in ks:
            ax.annotate(f'{k}', (dist[1, k], supp[1, k]), textcoords='offset points',
                        xytext=(4, 4) if k == 0 else (4, -9), fontsize=7, color='0.25')
        ax.errorbar(dist[1, -1], supp[1, -1],
                    xerr=np.abs(dist[[0, 2], -1:] - dist[1, -1]),
                    yerr=np.abs(supp[[0, 2], -1:] - supp[1, -1]),
                    color=ADAPTIVE_COLOR, marker='*', ms=10, lw=0, elinewidth=.8, capsize=2,
                    mec='white', mew=.6, zorder=5, label='Adaptive $k$')
        ax.set(xlabel='Neural distortion (%)', ylabel='Artifact suppression (dB)',
               title=DATASET_LABELS.get(name, name))
        ax = axes[1, col]
        ax.errorbar(ks, snr[1, :-1], yerr=np.abs(snr[[0, 2], :-1] - snr[1, :-1]),
                    color=FIXED_COLOR, marker='o', ms=4, lw=1.2, elinewidth=.6, capsize=1.5)
        ax.axhline(snr[1, -1], color=ADAPTIVE_COLOR, ls='--', lw=1.2)
        ax.axhspan(snr[0, -1], snr[2, -1], color=ADAPTIVE_COLOR, alpha=.12, lw=0)
        ax.set(xlabel='Removed dimension $k$', ylabel='Output SNR (dB)', xticks=ks)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc='upper center', frameon=False, bbox_to_anchor=(.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, .95))
    for ext in ('pdf', 'png'):
        fig.savefig(f'{path_stem}.{ext}', dpi=300)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', choices=['all', 'swec_synthetic', 'swec_pbs'], default='all')
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--cache-dir', type=Path, default=ROOT / 'results/pulse_swec_cache')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'figures/select_k')
    p.add_argument('--stim-channels', type=int, nargs='+', default=[0, 1, 2, 8])
    p.add_argument('--synthetic-rms-ratio', type=float,
                   help='Rescale each synthetic artifact to this RMS ratio over neural (default: cached scale)')
    p.add_argument('--trials', type=int, nargs='+', help='Synthetic trial indices (default: all cached)')
    p.add_argument('--pbs-configurations', type=int, nargs='+', default=list(range(13)))
    p.add_argument('--pbs-artifact-scale', type=float, default=1e4)
    p.add_argument('--k-max', type=int, default=6)
    p.add_argument('--beta', type=float, default=.999)
    p.add_argument('--threshold', type=float, default=.1)
    p.add_argument('--check-period', type=int, default=100)
    p.add_argument('--rule', choices=['stability_gap', 'relative'], default='stability_gap')
    p.add_argument('--skip-ms', type=float, default=0.,
                   help='Exclude initial tracker convergence from metrics (default includes startup)')
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    if not 0 < args.beta < 1 or args.threshold <= 0 or args.check_period < 1 or args.k_max < 1:
        p.error('Require 0 < beta < 1, positive threshold, check period and k_max')
    return args


def main():
    args = parse_args()
    datasets = ['swec_synthetic', 'swec_pbs'] if args.dataset == 'all' else [args.dataset]
    jobs = []
    for name in datasets:
        if name == 'swec_synthetic':
            n_cached = np.load(args.cache_dir / 'clean.npy', mmap_mode='r').shape[0]
            indices = args.trials if args.trials is not None else range(n_cached)
            jobs.extend((name, i, None, args) for i in indices)
        else:
            jobs.extend((name, c, payload, args) for c, payload in load_pbs(args).items())
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(run_one, jobs))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ratio = '' if args.synthetic_rms_ratio is None else f'_rms{args.synthetic_rms_ratio:g}'
    rule = 'gap' if args.rule == 'stability_gap' else f'relative_thr{args.threshold:g}'
    stem = args.output_dir / f'adaptive_k_tradeoff_{rule}_beta{args.beta:g}{ratio}'
    conditions = [f'k={k}' for k in range(args.k_max + 1)] + ['adaptive']
    with open(f'{stem}_per_recording.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'recording', 'condition', 'artifact_suppression_db',
                         'neural_distortion_pct', 'output_snr_db', 'final_adaptive_k'])
        for r in rows:
            for j, condition in enumerate(conditions[:r['k_max'] + 1] + ['adaptive']):
                writer.writerow([r['dataset'], r['recording'], condition, r['artifact_suppression_db'][j],
                                 r['neural_distortion_pct'][j], r['output_snr_db'][j], r['final_adaptive_k']])

    summaries = {name: summarize([r for r in rows if r['dataset'] == name], args.k_max)
                 for name in datasets}
    with open(f'{stem}_summary.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'condition', 'metric', 'q1', 'median', 'q3'])
        for name, s in summaries.items():
            for metric in ('artifact_suppression_db', 'neural_distortion_pct', 'output_snr_db'):
                for j, condition in enumerate(conditions):
                    writer.writerow([name, condition, metric, *s[metric][:, j]])
    plot(summaries, args.k_max, stem)
    protocol = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    protocol.update(tracking='PASTd on first differences; no filtering or normalization; '
                             'canonical initialization; reset per recording',
                    adaptive_rule=f'start k=1; {args.rule} rule every check_period samples',
                    final_adaptive_k={name: s['final_adaptive_k'].tolist() for name, s in summaries.items()})
    Path(f'{stem}_protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    for name, s in summaries.items():
        print(f'\n{DATASET_LABELS[name]} (median across recordings)')
        print(f'{"condition":>10} {"supp dB":>9} {"neural %":>9} {"SNR dB":>8}')
        for j, condition in enumerate(conditions):
            print(f'{condition:>10} {s["artifact_suppression_db"][1, j]:9.2f} '
                  f'{s["neural_distortion_pct"][1, j]:9.3f} {s["output_snr_db"][1, j]:8.2f}')
        print('final adaptive k [Q1, median, Q3]:', s['final_adaptive_k'].tolist())
    print('Saved:', f'{stem}.pdf')


if __name__ == '__main__':
    main()
