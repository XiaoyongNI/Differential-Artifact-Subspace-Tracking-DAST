"""Eight-channel overlays for PBS configurations 0 and 8.

Uses saved DAST + harmonic outputs from add_pbs_harmonic_filter.py.
Run: python tests/plot_pbs_dast_waveforms.py
"""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
from data_loading import ROOT
from plotting import set_tbme_style


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=ROOT/'results/pbs_dast_harmonic_table')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'figures/pbs_dast_harmonic')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_tbme_style()
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'stix',
                         'font.size': 24, 'axes.labelsize': 24,
                         'xtick.labelsize': 22, 'ytick.labelsize': 22,
                         'axes.grid': True, 'grid.alpha': .18,
                         'grid.linewidth': .5, 'pdf.fonttype': 42})
    styles = [('reference', 'Reference', '0.18', '--'),
              ('contaminated', 'Noisy signal', '#D55E00', '--'),
              ('cleaned', 'DAST', '#0072B2', '-')]
    handles = [Line2D([0], [0], color=color, linestyle=ls, linewidth=1.2,
                      label=label) for _, label, color, ls in styles]
    for config in (0, 8):
        with np.load(args.source_dir/f'configuration_{config:02d}.npz') as data:
            signals = {key: data[key][0] for key, *_ in styles}
            time_ms = data['original_sample_indices']/float(data['fs'])*1000
        if any(x.shape != (8, len(time_ms)) or not np.isfinite(x).all()
               for x in signals.values()):
            raise ValueError('Expected three finite, aligned eight-channel signals')
        fig, axes = plt.subplots(8, 1, figsize=(7.2, 11), sharex=True)
        for ch, ax in enumerate(axes):
            # Draw noisy signal first so the reference and reconstruction remain visible.
            for key, label, color, ls in (styles[1], styles[0], styles[2]):
                ax.plot(time_ms, signals[key][ch], color=color,
                        linestyle=ls, linewidth=1.2, label=label)
            ax.set_ylabel(r'$\mu$V')
            ax.text(.01, .88, f'Ch {ch+1}',
        transform=ax.transAxes, va='top', fontsize=26,
        bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=1))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.margins(x=0, y=.18)
            ax.spines[['top', 'right']].set_visible(False)
        axes[-1].set_xlabel('Time (ms)')
        # fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .966),
        #            ncol=3, frameon=False, handlelength=2.5)
        fig.subplots_adjust(left=.22, right=.965, bottom=.1, top=.975, hspace=.23)
        for ext in ('pdf', 'png', 'svg'):
            path = args.output_dir/f'pbs_config_{config:02d}_eight_channels.{ext}'
            fig.savefig(path, dpi=300, facecolor='white')
            print(path)
        plt.close(fig)


if __name__ == '__main__':
    main()
