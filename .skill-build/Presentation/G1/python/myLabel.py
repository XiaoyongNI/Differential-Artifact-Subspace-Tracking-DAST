"""Axis labels from myLabel.m; the original title is disabled."""
import matplotlib.pyplot as plt


def myLabel(ax=None, *, fontsize=36):
    ax = plt.gca() if ax is None else ax
    return (ax.set_xlabel(r'$\frac{1}{r^2}$ [dB]', fontsize=fontsize),
            ax.set_ylabel('MSE [dB]', fontsize=fontsize))


my_label = myLabel
