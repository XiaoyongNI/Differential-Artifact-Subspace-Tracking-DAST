"""Adaptive k tracks the artifact dimension: trade-off and selection accuracy.

python tests/plot_adaptive_k_dimension.py

SWEC + synthetic: stimulating sites on 1..3 different electrode strips
(channels 0, 8, 16) gives an artifact of known spatial dimension 1..3
(k_max = 4 leaves one neural-dominated direction for the stability gap). Artifacts are rescaled to fixed RMS ratios over the neural signal.
SWEC + PBS: configurations 0-12, artifact dimension 1 (Fig. 2).

Each recording uses the stability-gap rule (choose_k_from_stability_gap) and
fixed k = 0..k_max from one PASTd trajectory (tests/plot_adaptive_k_tradeoff.py).
Lock time is the first sample after the adaptive k last changed, taken from the
algorithm's own output (no ground truth). With --metric-window converged
(default), metrics for fixed and adaptive k alike use only samples from the
lock time on, i.e. steady-state operation after the subspace has converged.
Dashed line: total artifact energy 10 log10(sum_t ||a^t||^2) over the same
evaluated samples (signal units squared), the suppression reached if the
residual artifact energy were one unit squared.
Outputs: trade-off figure for --plot-ratio, per-recording CSV, accuracy table.
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
import matplotlib.ticker
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from data_loading import ROOT
from plotting import set_tbme_style
from plot_adaptive_k_tradeoff import load_pbs, load_synthetic, metrics_from_energies, tradeoff_energies

STRIP_CHANNELS = [0, 8, 16]
FIXED_COLOR, ADAPTIVE_COLOR = '#0072B2', '#D55E00'


def job_args(args, stim_channels=None, ratio=None):
    return argparse.Namespace(cache_dir=args.cache_dir, stim_channels=stim_channels,
                              synthetic_rms_ratio=ratio, k_max=args.k_max, beta=args.beta,
                              check_period=args.check_period, metric_window=args.metric_window,
                              warm_start=args.warm_start)


def run_recording(job):
    dataset, index, payload, args = job
    neural, artifact, fs = payload if payload is not None else load_synthetic(index, args)
    settings = (args.k_max, args.beta, .1, args.check_period)
    energy = tradeoff_energies(neural, artifact, *settings, skip=0, rule='stability_gap',
                               warm_start=args.warm_start)
    chosen = energy['chosen_k']
    changes = np.flatnonzero(np.diff(chosen))
    lock = int(changes[-1] + 1) if len(changes) else 0
    if args.metric_window == 'converged' and lock > 0:
        energy = tradeoff_energies(neural, artifact, *settings, skip=lock, rule='stability_gap',
                                   warm_start=args.warm_start)
    metrics = metrics_from_energies(energy)
    start = 1 + (lock if args.metric_window == 'converged' else 0)  # sample 0 seeds the difference
    return dict(recording=index, final_adaptive_k=int(chosen[-1]), lock_ms=1e3 * lock / fs,
                artifact_energy_db=float(10 * np.log10(np.sum(artifact[:, start:] ** 2))),
                evaluated_ms=1e3 * (len(chosen) - (lock if args.metric_window == 'converged' else 0)) / fs,
                **{key: value.tolist() for key, value in metrics.items()})


def quartiles(values):
    return np.nanpercentile(np.asarray(values, dtype=float), [25, 50, 75], axis=0)


def plot(panels, k_max, path_stem):
    set_tbme_style()
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'stix', 'font.size': 16,
                         'axes.labelsize': 16, 'axes.titlesize': 16, 'legend.fontsize': 16,
                         'xtick.labelsize': 16, 'ytick.labelsize': 16,
                         'grid.alpha': .18, 'grid.linewidth': .5, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 6.2), layout='constrained')
    fig.get_layout_engine().set(w_pad=.02, h_pad=.03, wspace=.04, hspace=.06)
    for ax, (title, rows, true_k) in zip(axes.flat, panels):
        dist = quartiles([r['neural_distortion_pct'] for r in rows])
        supp = quartiles([r['artifact_suppression_db'] for r in rows])
        ks = np.arange(k_max + 1)
        energy = float(np.median([r['artifact_energy_db'] for r in rows]))
        ax.axhline(energy, ls='--', color='0.45', lw=1.3, label='Total artifact energy')
        top = max(energy, float(np.nanmax(supp[2, :-1]))) * 1.08
        ax.set_ylim(-.04 * top, top)
        ax.errorbar(dist[1, :-1], supp[1, :-1], xerr=np.abs(dist[[0, 2], :-1] - dist[1, :-1]),
                    yerr=np.abs(supp[[0, 2], :-1] - supp[1, :-1]), color=FIXED_COLOR, marker='o',
                    ms=5, lw=1.4, elinewidth=.7, capsize=1.5, label='Fixed $k$')
        for k in ks[1:]:  # k = 0 is always the origin
            ax.annotate(f'{k}', (dist[1, k], supp[1, k]), textcoords='offset points',
                        xytext=(4, 4) if k == 0 else (4, -14), fontsize=12, color='0.25')
        ax.plot(dist[1, true_k], supp[1, true_k], 'o', ms=13, mfc='none', mec='0.1', mew=1.1,
                label='True dimension')
        ax.errorbar(dist[1, -1], supp[1, -1], xerr=np.abs(dist[[0, 2], -1:] - dist[1, -1]),
                    yerr=np.abs(supp[[0, 2], -1:] - supp[1, -1]), color=ADAPTIVE_COLOR, marker='*',
                    ms=13, lw=0, elinewidth=.9, capsize=2, mec='white', mew=.6, zorder=5,
                    label='Adaptive $k$')
        final = [r['final_adaptive_k'] for r in rows]
        accuracy = np.mean(np.array(final) == true_k)
        ax.set_title(title)
        ax.text(.97, .1, f'$k={int(np.median(final))}$ ({accuracy:.0%})',
                transform=ax.transAxes, ha='right', va='center', fontsize=13, color='0.15')
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4, steps=[1, 2.5, 5, 10]))
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc='outside upper center', frameon=False,
               columnspacing=1., handletextpad=.4)
    fig.supxlabel('Neural distortion (%)', fontsize=16)
    fig.supylabel('Artifact suppression (dB)', fontsize=16)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{path_stem}.{ext}', dpi=300)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--cache-dir', type=Path, default=ROOT / 'results/pulse_swec_cache')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'figures/select_k')
    p.add_argument('--ratios', type=float, nargs='+', default=[2., 5., 10.])
    p.add_argument('--plot-ratio', type=float, default=5.)
    p.add_argument('--k-max', type=int, default=4)
    p.add_argument('--warm-start', type=int, default=128,
                   help='Causal SVD re-initialization after this many samples (0: canonical start only)')
    p.add_argument('--beta', type=float, default=.999)
    p.add_argument('--check-period', type=int, default=100)
    p.add_argument('--workers', type=int, default=20)
    p.add_argument('--metric-window', choices=['converged', 'all'], default='converged',
                   help="'converged': evaluate from each recording's lock time; 'all': whole recording")
    args = p.parse_args()
    if args.plot_ratio not in args.ratios:
        p.error('--plot-ratio must be one of --ratios')
    if args.k_max <= len(STRIP_CHANNELS):
        p.error('--k-max must exceed the largest artifact dimension so a neural component is tracked')
    return args


def main():
    args = parse_args()
    n_trials = np.load(args.cache_dir / 'clean.npy', mmap_mode='r').shape[0]
    jobs, keys = [], []
    for ratio in args.ratios:
        for dim in range(1, len(STRIP_CHANNELS) + 1):
            options = job_args(args, STRIP_CHANNELS[:dim], ratio)
            for trial in range(n_trials):
                jobs.append(('swec_synthetic', trial, None, options))
                keys.append(('synthetic', ratio, dim))
    pbs_args = argparse.Namespace(root=args.root, pbs_artifact_scale=1e4, pbs_configurations=list(range(13)))
    for c, payload in load_pbs(pbs_args).items():
        jobs.append(('swec_pbs', c, payload, job_args(args)))
        keys.append(('pbs', None, 1))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(run_recording, jobs))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / (f'adaptive_k_dimension_gap_kmax{args.k_max}_warm{args.warm_start}'
                              f'_beta{args.beta:g}_{args.metric_window}')
    conditions = [f'k={k}' for k in range(args.k_max + 1)] + ['adaptive']
    with open(f'{stem}_per_recording.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'rms_ratio', 'true_dimension', 'recording', 'condition',
                         'artifact_suppression_db', 'neural_distortion_pct', 'output_snr_db', 'final_adaptive_k',
                         'lock_ms', 'evaluated_ms', 'artifact_energy_db'])
        for (name, ratio, dim), r in zip(keys, rows):
            for j, condition in enumerate(conditions):
                writer.writerow([name, ratio, dim, r['recording'], condition, r['artifact_suppression_db'][j],
                                 r['neural_distortion_pct'][j], r['output_snr_db'][j], r['final_adaptive_k'],
                                 r['lock_ms'], r['evaluated_ms'], r['artifact_energy_db']])

    groups = {}
    for key, r in zip(keys, rows):
        groups.setdefault(key, []).append(r)
    with open(f'{stem}_accuracy.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'rms_ratio', 'true_dimension', 'recordings', 'selection_accuracy',
                         'selected_k_counts', 'median_lock_ms', 'median_evaluated_ms'])
        print(f'{"condition":>22} {"accuracy":>9} {"lock ms":>8} {"eval ms":>8}  selected k')
        for (name, ratio, dim), group in groups.items():
            final = [r['final_adaptive_k'] for r in group]
            counts = {int(k): int(n) for k, n in zip(*np.unique(final, return_counts=True))}
            accuracy = float(np.mean(np.array(final) == dim))
            lock = float(np.median([r['lock_ms'] for r in group]))
            evaluated = float(np.median([r['evaluated_ms'] for r in group]))
            writer.writerow([name, ratio, dim, len(group), accuracy, json.dumps(counts), lock, evaluated])
            label = 'PBS' if name == 'pbs' else f'synthetic d={dim} rms={ratio:g}'
            print(f'{label:>22} {accuracy:9.2f} {lock:8.0f} {evaluated:8.0f}  {counts}')

    panels = [(f'Synthetic-{dim}stim', groups[('synthetic', args.plot_ratio, dim)], dim)
              for dim in range(1, len(STRIP_CHANNELS) + 1)]
    panels.append(('PBS', groups[('pbs', None, 1)], 1))
    plot(panels, args.k_max, stem)
    Path(f'{stem}_protocol.json').write_text(json.dumps(dict(
        strip_channels=STRIP_CHANNELS, ratios=args.ratios, plot_ratio=args.plot_ratio, k_max=args.k_max,
        beta=args.beta, check_period=args.check_period, rule='stability_gap', warm_start=args.warm_start,
        metric_window=args.metric_window,
        tracking='PASTd on first differences; no filtering or normalization; reset per recording',
        pbs='configurations 0-12, separated SWEC channels, artifact scale 1e4, common interval',
        notes=['Rule designed after inspecting RMS ratios 2 and 10 and PBS statistics; '
               'RMS ratio 5 was not inspected before evaluation.']), indent=2) + '\n')
    print('Saved:', f'{stem}.pdf')


if __name__ == '__main__':
    main()
