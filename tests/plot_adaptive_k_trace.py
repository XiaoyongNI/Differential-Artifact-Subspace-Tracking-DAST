"""Instability traces of the tracked components and the stability-gap choice of k.

python tests/plot_adaptive_k_trace.py

SWEC + synthetic trial with 1, 2 and 3 stimulation channels on distinct strips
(channels 0, 8, 16: artifact dimension 1, 2, 3), one column each, artifact
rescaled to --rms-ratio over the neural RMS. PASTd (rank --k-max, beta) tracks first differences without
normalization, exactly as run_pipeline with adaptive_k_rule='stability_gap'.

Top: instability e_j(t), the moving average of ||u_j(t) - u_j(t-1)|| that
excludes the first update. It is shown bias-corrected (divided by
1 - beta^t) for readability; the correction is common to all components, so
the selection, which uses log ratios, is unchanged.
Bottom: k chosen by choose_k_from_stability_gap every --check-period samples.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
import matplotlib.ticker
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from algorithms import PASTd, choose_k_from_stability_gap
from data_loading import ROOT
from plotting import set_tbme_style
from plot_adaptive_k_tradeoff import load_synthetic

COLORS = ('#0072B2', '#D55E00', '#009E73', '#CC79A7')


def trace(neural, artifact, k_max, beta, check_period, warm_start=0):
    dx = np.diff(neural + artifact, axis=-1)
    tracker = PASTd(dx.shape[0], k_max, beta=beta)
    ema = np.zeros(k_max)
    instability = np.full((dx.shape[1], k_max), np.nan)
    chosen = np.ones(dx.shape[1], dtype=int)
    k = 1
    ema_start = 0
    for t in range(dx.shape[1]):
        if 0 < warm_start < dx.shape[1] and t == warm_start:
            tracker.warm_start(dx[:, :t])  # causal: samples 0..t-1 only
            ema = np.zeros(k_max)
            ema_start = t
        tracker.update(dx[:, t])
        if t > ema_start:  # skip the first update after (re)initialization
            ema = beta * ema + (1 - beta) * tracker.delta_u
            instability[t] = ema / (1 - beta ** (t - ema_start))
        if t > ema_start + 1 and t % check_period == 0:
            k = choose_k_from_stability_gap(ema)
        chosen[t] = k
    changes = np.flatnonzero(np.diff(chosen))
    lock = int(changes[-1] + 1) if len(changes) else 0
    return instability, chosen, lock


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cache-dir', type=Path, default=ROOT / 'results/pulse_swec_cache')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'figures/select_k')
    p.add_argument('--trial', type=int, default=0)
    p.add_argument('--strip-channels', type=int, nargs='+', default=[0, 8, 16],
                   help='Column n stimulates the first n of these channels')
    p.add_argument('--rms-ratio', type=float, default=5.)
    p.add_argument('--k-max', type=int, default=4)
    p.add_argument('--beta', type=float, default=.999)
    p.add_argument('--check-period', type=int, default=100)
    p.add_argument('--warm-start', type=int, default=128,
                   help='Causal SVD re-initialization after this many samples (0: canonical start only)')
    args = p.parse_args()
    if not 0 < args.beta < 1 or args.k_max <= len(args.strip_channels):
        p.error('Require 0 < beta < 1 and k_max larger than the largest artifact dimension')

    meta = json.loads((args.cache_dir / 'source.json').read_text())
    results = []
    for n in range(1, len(args.strip_channels) + 1):
        options = argparse.Namespace(cache_dir=args.cache_dir, stim_channels=args.strip_channels[:n],
                                     synthetic_rms_ratio=args.rms_ratio)
        neural, artifact, fs = load_synthetic(args.trial, options)
        results.append((n, *trace(neural, artifact, args.k_max, args.beta, args.check_period, args.warm_start)))
    time = np.arange(1, len(results[0][2]) + 1) / fs

    set_tbme_style()
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'stix', 'font.size': 16,
                         'axes.labelsize': 16, 'axes.titlesize': 16, 'legend.fontsize': 16,
                         'xtick.labelsize': 15, 'ytick.labelsize': 15,
                         'grid.alpha': .18, 'grid.linewidth': .5, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, len(results), figsize=(7.4, 4.4), sharex=True,
                             layout='constrained', gridspec_kw=dict(height_ratios=(1.7, 1)))
    fig.get_layout_engine().set(w_pad=.02, h_pad=.02, wspace=.03, hspace=.03)
    for col, (n, instability, chosen, lock) in enumerate(results):
        ax, ax_k = axes[0, col], axes[1, col]
        for j in range(args.k_max):
            ax.plot(time, instability[:, j], color=COLORS[j], lw=1.5,
                    label=rf'$\hat{{\mathbf{{u}}}}_{j + 1}$')
        ax.set_yscale('log')
        # Skip 10 ms after the last (re)initialization: EMA still dominated by single updates.
        settled = instability[args.warm_start + int(.01 * fs):]
        ax.set_ylim(np.nanmin(settled) / 1.3, np.nanmax(settled) * 1.3)
        if col:
            ax_k.tick_params(labelleft=False)
        ax.set_title(f'Synthetic-{n}stim', fontsize=15)
        ax_k.step(time, chosen, where='post', color='0.15', lw=1.6)
        ax_k.axhline(n, color='0.5', ls='--', lw=1.2)
        ax_k.set(xlim=(0, time[-1]), ylim=(.5, args.k_max + .5), yticks=range(1, args.k_max + 1))
        ax_k.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(.5))
        ax_k.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
        if lock:
            for a in (ax, ax_k):
                a.axvline(lock / fs, color='0.6', ls=':', lw=1.2)
    axes[0, 0].set_ylabel(r'Instability $\bar{e}_i$')
    axes[1, 0].set_ylabel('Selected $k$')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color='0.5', ls='--', lw=1.2))
    labels.append('True $k$')
    fig.legend(handles, labels, ncol=len(labels), loc='outside upper center', frameon=False,
               columnspacing=.9, handlelength=1.3, handletextpad=.4)
    fig.supxlabel('Time (s)', fontsize=16)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / (f'adaptive_k_trace_gap_1-{len(results)}stim_kmax{args.k_max}_warm{args.warm_start}'
                              f'_rms{args.rms_ratio:g}_beta{args.beta:g}_trial{args.trial}')
    for ext in ('pdf', 'png'):
        fig.savefig(f'{stem}.{ext}', dpi=300)
    plt.close(fig)
    report = dict(source_cache=meta['source'], trial=args.trial, strip_channels=args.strip_channels,
                  rms_ratio=args.rms_ratio, k_max=args.k_max, beta=args.beta, check_period=args.check_period,
                  rule='choose_k_from_stability_gap; EMA excludes the first update', warm_start=args.warm_start,
                  columns=[dict(true_k=n, final_k=int(chosen[-1]), lock_s=lock / fs,
                                final_instability=instability[-1].tolist())
                           for n, instability, chosen, lock in results])
    Path(f'{stem}.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['columns'], indent=1))
    print('Saved:', f'{stem}.pdf')


if __name__ == '__main__':
    main()
