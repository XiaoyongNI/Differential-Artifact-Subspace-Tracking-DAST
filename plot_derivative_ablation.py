"""Plot dataset-selectable cancellation with and without time-difference tracking.

The projection/removal step is always applied to the original voltage signal.
Only the tracker update input changes:

* without differential: PASTd updates with x(t)
* with differential: PASTd updates with first-, second-, or third-order
  time differences, then removes from x(t)

The tracker state carries across selected trials. By default, the first trial
is used for learning and the second trial is plotted. Single-trial datasets
(such as PBS) use their only trial for both online learning and evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from algorithms import apply_lpf_3d

from data_loading import ROOT, load_dataset
from pipeline import PipelineConfig, run_pipeline
from plotting import set_tbme_style
from utils import ensure_dir


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def nrmse(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.ptp(b)) + 1e-12
    return rmse(a, b) / denom


def snr_improvement_db(before: np.ndarray, after: np.ndarray, ref: np.ndarray) -> float:
    e0 = np.mean((before - ref) ** 2) + 1e-12
    e1 = np.mean((after - ref) ** 2) + 1e-12
    return float(10.0 * np.log10(e0 / e1))


def baseline_correlation(estimate, baseline, fs, domain="spectral"):
    """Pearson r against a separate baseline, not reconstruction accuracy.

    Spectral mode uses linear Welch PSDs on identical frequency bins. Time mode
    pairs samples from each interval's start, truncating to the shorter length.
    """
    a, b = (np.asarray(x, dtype=float).ravel() for x in (estimate, baseline))
    if min(a.size, b.size) < 3 or not (np.isfinite(a).all() and np.isfinite(b).all()):
        return float("nan")
    if domain == "spectral":
        nperseg = min(a.size, b.size, max(4, int(round(0.02 * fs))))
        _, a = welch(a, fs=fs, nperseg=nperseg)
        _, b = welch(b, fs=fs, nperseg=nperseg)
        a, b = a[1:], b[1:]  # Exclude DC; retain all positive frequencies.
    elif domain == "time":
        n = min(a.size, b.size)
        a, b = a[:n], b[:n]
    else:
        raise ValueError("Correlation domain must be spectral or time.")
    a, b = a - a.mean(), b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.clip(a @ b / denom, -1, 1)) if denom > 0 else float("nan")


DATASET_CHOICES = {
    "swec": "swec_synthetic",
    "eraasr": "ERAASR",
    "dictionary": "dictionary_2stim",
    "dictionary_4stim": "dictionary_4stim",
    "dictionary_5stim": "dictionary_5stim",
    "dictionary_14stim": "dictionary_14stim",
    "pbs": "swec_pbs",
}


def load_ablation_dataset(args: argparse.Namespace):
    """Resolve CLI dataset selection independently of processing and plotting."""
    name = DATASET_CHOICES[args.dataset]
    options = {}
    if name == "swec_synthetic":
        options.update(stim_channels=args.stim_channels, crop_to_stim=True)
    elif name == "swec_pbs":
        options.update(configuration_number=args.pbs_configuration,
                       separated_recording_channels=args.pbs_separated_channels)
    if name in {"swec_synthetic", "swec_pbs"} and args.artifact_scale is not None:
        options["artifact_scale"] = args.artifact_scale
    return load_dataset(name, root=args.root, **options)


def run_ablation(dataset, args: argparse.Namespace):
    """Compare undifferentiated tracking with orders 1, 2, and 3."""
    if dataset.X.shape[2] <= 3:
        raise ValueError("Third-order differentiation requires at least four time samples.")
    n_selected = min(args.n_test, dataset.X.shape[0])
    if args.trial is None:
        args.trial = min(1, n_selected - 1)
    if args.skip_ms is None:
        # Keep the SWEC/PBS settling interval; shorter recordings use prior-trial learning.
        args.skip_ms = 500.0 if dataset.X.shape[2] / dataset.fs > 0.5 else 0.0
    if not 0 <= args.trial < n_selected:
        raise ValueError(f"--trial must be between 0 and {n_selected - 1}")
    if not 0 <= args.channel < dataset.X.shape[1]:
        raise ValueError(f"--channel must be between 0 and {dataset.X.shape[1] - 1}")
    if args.skip_ms / 1000 >= dataset.X.shape[2] / dataset.fs:
        raise ValueError("--skip-ms exceeds the trial duration; reduce it for this dataset")

    common = dict(
        n_test=args.n_test,
        preprocessing=args.preprocessing,
        lpf_cutoff_hz=args.lpf_cutoff,
        normalization="none",
        tracker="PASTd",
        rank=args.rank,
        beta=args.beta,
        adaptive_k=args.adaptive_k,
        adaptive_k_threshold=args.adaptive_k_threshold,
        harmonic_filter=args.harmonic_filter,
        harmonic_max_harmonics=args.harmonics,
        harmonic_settling_time=args.harmonic_settling_time,
        compute_ideal_u=True,
        carry_tracker_across_trials=True,
    )
    res_no_diff = run_pipeline(dataset, PipelineConfig(use_derivative_for_tracking=False, **common))
    res_diff = {
        order: run_pipeline(dataset, PipelineConfig(
            use_derivative_for_tracking=True, derivative_order=order, **common
        ))
        for order in (1, 2, 3)
    }
    return res_no_diff, res_diff


def plot_ablation(dataset, res_no_diff, res_diff, args: argparse.Namespace) -> Path:
    set_tbme_style()
    out_dir = ensure_dir(args.out_dir)

    trial = args.trial
    ch = args.channel
    fs = float(dataset.fs)

    offset = max(res_diff)
    raw = res_no_diff.stages["raw"][trial, ch, offset:]
    # ERAASR's prestimulation baseline is not simultaneous ground truth.
    has_reference = (dataset.y_clean is not None and dataset.y_clean.shape == dataset.X.shape
                     and dataset.name != "ERAASR")
    ref = dataset.y_clean[-args.n_test:][trial, ch, offset:] if has_reference else None
    baseline = None
    if not has_reference and dataset.baseline is not None:
        baseline = dataset.baseline[-args.n_test:][trial:trial + 1, ch:ch + 1].copy()
        if args.preprocessing == "lpf":
            baseline = apply_lpf_3d(baseline, cutoff=args.lpf_cutoff, fs=fs,
                                    order=res_no_diff.config.lpf_order)
        baseline = baseline.ravel()
    no_diff = res_no_diff.stages["cleaned"][trial, ch, offset:]
    with_diff = {
        order: result.stages["cleaned"][trial, ch, offset - order:]
        for order, result in res_diff.items()
    }

    n = min(raw.size, no_diff.size, *(signal.size for signal in with_diff.values()))
    if ref is not None:
        n = min(n, ref.size)
    if args.skip_ms is not None and (not np.isfinite(args.skip_ms) or args.skip_ms < 0):
        raise ValueError("--skip-ms must be finite and nonnegative")
    # Align all conditions at the first sample available for the highest order.
    # Crop only the display, after the tracker has processed the full trials.
    start = max(0, int(np.ceil(args.skip_ms * fs / 1000.0)) - offset)
    stop = n
    if args.window_s is not None:
        if not np.isfinite(args.window_s) or args.window_s <= 0:
            raise ValueError("--window-s must be finite and positive")
        stop = min(n, start + int(args.window_s * fs))
    if stop <= start:
        raise ValueError("No samples to plot: reduce --skip-ms or increase --window-s")
    raw, no_diff = (signal[start:stop] for signal in (raw, no_diff))
    with_diff = {order: signal[start:stop] for order, signal in with_diff.items()}
    if ref is not None:
        ref = ref[start:stop]
    t = (np.arange(start, stop) + offset) / fs

    signals = [no_diff, *with_diff.values()] + ([ref] if ref is not None else [])
    ylim = (min(float(np.min(s)) for s in signals),
            max(float(np.max(s)) for s in signals))
    pad = 0.08 * (ylim[1] - ylim[0] + 1e-12)
    ylim = (ylim[0] - pad, ylim[1] + pad)

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    conditions = [
        ("Without time differentiation", no_diff, "#D55E00", "--"),
        ("1st-order differentiation", with_diff[1], "#0072B2", "-"),
        # ("2nd-order differentiation", with_diff[2], "#009E73", "-."),
        # ("3rd-order differentiation", with_diff[3], "#CC79A7", ":"),
    ]
    if ref is not None:
        ax.plot(t, ref, color="0.18", linestyle="--", linewidth=1.1, label="Reference")
    for label, estimate, color, linestyle in conditions:
        ax.plot(t, estimate, color=color, linestyle=linestyle, linewidth=1.0, label=label)
    ax.set_ylim(*ylim)
    ax.set_xlabel(f"Time (s)")
    ax.set_ylabel("Amplitude (uV)")
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center",
               ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.subplots_adjust(top=0.78, bottom=0.18)

    path = out_dir / (
        f"{dataset.name}_derivative_ablation_trial{trial}_ch{ch}_"
        f"rank{args.rank}_beta{args.beta:g}.pdf"
    )
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    metrics = [
        f"{dataset.name}, trial index {trial} (zero-based), channel {ch}",
        f"Metrics over the displayed interval: {len(t)} samples at {fs:g} Hz",
        f"Displayed trial time: {t[0]:.6f} to {t[-1]:.6f} s (skip {args.skip_ms:g} ms)",
    ]
    domain = getattr(args, "baseline_correlation", "spectral")
    if has_reference:
        metrics.extend(["Reference: ground-truth clean signal",
                        "SNR gain is relative to the signal before cancellation."])
    elif baseline is not None:
        metadata = dataset.metadata or {}
        metrics.extend([
            f"Baseline: {metadata.get('baseline_source', 'separate non-stimulation period')}; same selected trial and channel",
            f"Baseline interval (source time, s): {metadata.get('baseline_interval_s', 'unspecified')}; {baseline.size} samples",
            f"Baseline preprocessing: {args.preprocessing}",
            "Correlation describes similarity to a separate period, not aligned reconstruction accuracy.",
        ])
        if domain == "spectral":
            segment = min(len(t), baseline.size, max(4, int(round(.02 * fs))))
            metrics.append(f"Metric: Pearson r of linear Welch PSD; Hann window, nperseg={segment}, 50% overlap, constant detrending; DC excluded, positive frequencies through Nyquist.")
        else:
            metrics.append(f"Metric: time-domain Pearson r; interval starts paired, first {min(len(t), baseline.size)} samples; no lag search.")
    else:
        metrics.append("Baseline unavailable: no identified non-stimulation period.")
    metrics.append("")
    for label, estimate, _, _ in conditions:
        if ref is None:
            value = baseline_correlation(estimate, baseline, fs, domain) if baseline is not None else float("nan")
            score = f"{value:.6f}" if np.isfinite(value) else "N/A (missing, nonfinite, too short, or constant baseline/estimate)"
            metrics.append(f"{label}: {domain} baseline correlation (Pearson r)={score}")
            continue
        metrics.append(
            f"{label}: RMSE={rmse(estimate, ref):.6f} uV, "
            f"NRMSE={nrmse(estimate, ref):.6f}, "
            f"SNR gain={snr_improvement_db(raw, estimate, ref):.6f} dB"
        )
    path.with_suffix(".txt").write_text("\n".join(metrics) + "\n")
    return path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare no differentiation and 1st-, 2nd-, and 3rd-order tracking on a selected dataset.")
    p.add_argument("--dataset", choices=DATASET_CHOICES, default="swec", type=str.lower)
    p.add_argument("--pbs-configuration", default=0, type=int)
    p.add_argument("--pbs-separated-channels", action="store_true", default=True)
    p.add_argument("--root", default=ROOT, type=Path,
                   help="Project directory containing the data/ folder.")
    p.add_argument("--stim-channels", nargs="*", type=int, default=[0])
    p.add_argument("--artifact-scale", default=None, type=float)
    p.add_argument("--n-test", default=10, type=int,
                   help="Number of final trials to process, including learning trials (all available if fewer).")
    p.add_argument("--trial", default=5, type=int,
                   help="Zero-based trial index; default is second trial, or first for single-trial data.")
    p.add_argument("--channel", default=0, type=int)
    p.add_argument("--rank", default=1, type=int)
    p.add_argument("--beta", default=0.999, type=float)
    p.add_argument("--adaptive-k", action="store_true")
    p.add_argument("--adaptive-k-threshold", default=0.1, type=float)
    p.add_argument("--preprocessing", default="none", choices=["none", "lpf"])
    p.add_argument("--lpf-cutoff", default=500.0, type=float)
    p.add_argument("--harmonic-filter", action="store_true")
    p.add_argument("--harmonics", default=10, type=int)
    p.add_argument("--harmonic-settling-time", default=1e-2, type=float)
    p.add_argument("--skip-ms", default=0, type=float,
                   help="Skip milliseconds for plotting only; default 500, or 0 for trials no longer than 500 ms.")
    p.add_argument("--window-s", default=0.5, type=float,
                   help="Duration to display after --skip-ms, in seconds.")
    p.add_argument("--baseline-correlation", choices=["spectral", "time"], default="spectral",
                   help="Correlation with a separate non-stimulation baseline when aligned ground truth is unavailable.")
    p.add_argument("--out-dir", default=Path("plots/derivative"), type=Path)
    args = p.parse_args()
    if args.n_test < 1 or (args.trial is not None and not 0 <= args.trial < args.n_test):
        p.error("--n-test must be positive and --trial must satisfy 0 <= trial < n-test")
    if args.skip_ms is not None and (not np.isfinite(args.skip_ms) or args.skip_ms < 0):
        p.error("--skip-ms must be finite and nonnegative")
    if not np.isfinite(args.window_s) or args.window_s <= 0:
        p.error("--window-s must be finite and positive")
    return args


def main() -> None:
    args = parse_args()
    dataset = load_ablation_dataset(args)
    res_no_diff, res_diff = run_ablation(dataset, args)
    path = plot_ablation(dataset, res_no_diff, res_diff, args)
    print(f"Wrote {path}")
    print(f"Wrote {path.with_suffix('.png')}")
    print(f"Wrote {path.with_suffix('.txt')}")


if __name__ == "__main__":
    main()
