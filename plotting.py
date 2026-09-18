"""TBME-style plotting helpers for comparing pipeline stages."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from algorithms import build_harmonic_freqs
from utils import ensure_dir, one_sided_spectrum


TBME_COLORS = {
    "raw": "#1f77b4",
    "preprocessed": "#2ca02c",
    "normalized": "#9467bd",
    "tracking_input": "#8c564b",
    "after_subspace": "#ff7f0e",
    "after_harmonic": "#d62728",
    "cleaned": "#d62728",
}


def set_tbme_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "lines.linewidth": 0.9,
        }
    )


def plot_time_frequency_comparison(
    stages: Mapping[str, np.ndarray],
    fs: float,
    stim_rate: Optional[float] = None,
    trial: int = 0,
    channel: int = 0,
    max_time_s: float = 0.2,
    max_freq_hz: Optional[float] = None,
    stage_order: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    save_path: Optional[str | Path] = None,
):
    set_tbme_style()
    keys = list(stage_order) if stage_order is not None else list(stages.keys())
    keys = [k for k in keys if k in stages]
    n_cols = len(keys)
    fig, axes = plt.subplots(2, n_cols, figsize=(4.2 * n_cols, 5.4), squeeze=False)
    if title:
        fig.suptitle(title)
    for j, key in enumerate(keys):
        x = np.asarray(stages[key])[trial, channel]
        n_show = min(x.size, int(max_time_s * fs))
        t_ms = np.arange(n_show) / fs * 1e3
        color = TBME_COLORS.get(key, None)
        axes[0, j].plot(t_ms, x[:n_show], color=color)
        axes[0, j].set_title(key.replace("_", " "))
        axes[0, j].set_xlabel("Time (ms)")
        axes[0, j].set_ylabel("Amplitude")

        freqs, mag = one_sided_spectrum(x, fs)
        f_lim = fs / 2 if max_freq_hz is None else min(max_freq_hz, fs / 2)
        mask = freqs <= f_lim
        axes[1, j].plot(freqs[mask], mag[mask], color=color)
        if stim_rate is not None:
            for h in build_harmonic_freqs(stim_rate, fs, max_harmonics=20):
                if h <= f_lim:
                    axes[1, j].axvline(h, color="0.55", linestyle="--", linewidth=0.5, alpha=0.55)
        axes[1, j].set_xlabel("Frequency (Hz)")
        axes[1, j].set_ylabel("|FFT|")
        axes[1, j].set_xlim(0, f_lim)
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        ensure_dir(save_path.parent)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_u_convergence(
    u_trace: np.ndarray,
    fs: float,
    ideal_u: Optional[np.ndarray] = None,
    trial: int = 0,
    component: int = 0,
    save_path: Optional[str | Path] = None,
):
    set_tbme_style()
    U = np.asarray(u_trace)[trial, :, :, component]
    t = np.arange(U.shape[0]) / fs
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.plot(t, U)
    if ideal_u is not None:
        ax.axhline(0.0, color="0.2", linewidth=0.5)
        ref = ideal_u[:, component]
        ax.plot([t[0], t[-1]], [ref[0], ref[0]], color="black", linestyle="--", linewidth=0.8, label="ideal ch0")
        ax.legend(loc="best")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"u{component + 1} weights")
    ax.set_title(f"Subspace convergence, component {component + 1}")
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        ensure_dir(save_path.parent)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_k_trace(chosen_k: np.ndarray, fs: float, trial: int = 0, save_path: Optional[str | Path] = None):
    set_tbme_style()
    k = np.asarray(chosen_k)[trial]
    t = np.arange(k.size) / fs
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    ax.step(t, k, where="post")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("chosen k")
    ax.set_title("Adaptive rank selection")
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        ensure_dir(save_path.parent)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def quick_report_plots(result, out_dir: str | Path, trial: int = 0, channel: int = 0) -> None:
    out_dir = ensure_dir(out_dir)
    fs = result.dataset.fs
    stim_rate = result.dataset.stim_rate
    f_lim = min(fs / 2, stim_rate * 10 + 100)
    plot_time_frequency_comparison(
        result.stages,
        fs=fs,
        stim_rate=stim_rate,
        trial=trial,
        channel=channel,
        max_freq_hz=f_lim,
        stage_order=("raw", "preprocessed", "tracking_input", "after_subspace", "cleaned"),
        title=f"{result.dataset.name}: trial {trial}, channel {channel}",
        save_path=out_dir / f"{result.dataset.name}_trial{trial}_ch{channel}_time_freq.png",
    )
    plot_k_trace(result.diagnostics["chosen_k"], fs=fs, trial=trial, save_path=out_dir / f"{result.dataset.name}_chosen_k.png")
    if "u_trace" in result.diagnostics:
        plot_u_convergence(
            result.diagnostics["u_trace"],
            fs=fs,
            ideal_u=result.diagnostics.get("ideal_u"),
            trial=trial,
            save_path=out_dir / f"{result.dataset.name}_u1_convergence.png",
        )

