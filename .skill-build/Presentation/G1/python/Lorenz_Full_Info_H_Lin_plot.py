"""Reproduce the G1 EKF/KalmanNet figure without a shared MATLAB workspace."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
try:
    from .Lorenz_Full_Info_H_Lin_data import load_data
    from .myLabel import myLabel
    from . import style as s
except ImportError:
    from Lorenz_Full_Info_H_Lin_data import load_data
    from myLabel import myLabel
    import style as s


def plot_results(data=None, ax=None):
    """Return (figure, axes). Plot only the first two ratios, as in MATLAB."""
    data = load_data() if data is None else data
    if ax is None:
        _, ax = plt.subplots(figsize=(24, 14))
    x = -np.asarray(data['r2_arr_dB'])
    for q_idx in range(2):
        ratio_db = 20 * np.log10(data['q_arr'][q_idx])
        label = s.q_str + f' {ratio_db:g} ' + s.db_str + s.del_str
        for key, prefix, name in [('MSE', 'KF', s.EKF_str),
                                   ('MSE_KNET', 'KNET', s.KNET_str)]:
            ax.plot(x, data[key][:, q_idx],
                    color=getattr(s, prefix + '_COLOR')[q_idx],
                    linestyle=getattr(s, prefix + '_LINE')[q_idx],
                    linewidth=getattr(s, prefix + '_LINE_WIDTH'),
                    marker=getattr(s, prefix + '_MARKER')[q_idx],
                    markersize=getattr(s, prefix + '_MARKER_SIZE'),
                    markerfacecolor='none', markeredgecolor='k', label=label + name)
    ax.plot(x, data['r2_arr_dB'], color='r', linestyle='-.',
            linewidth=s.KNET_LINE_WIDTH, label=s.BL_str)
    ax.set_xlim(0, 40)
    ax.set_ylim(-51, 0)
    ax.tick_params(axis='both', labelsize=30)
    ax.grid(True)
    ax.legend(fontsize=35, loc='upper right', framealpha=1)
    myLabel(ax)
    ax.figure.tight_layout()
    return ax.figure, ax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save', type=Path, help='Output PNG, PDF, SVG, or EPS path')
    parser.add_argument('--no-show', action='store_true', help='Render without a GUI')
    args = parser.parse_args()
    if args.no_show:
        plt.switch_backend('Agg')
    fig, _ = plot_results()
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(args.save), dpi=160, bbox_inches='tight')
        print(f'Saved {args.save.resolve()}')
    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == '__main__':
    main()

