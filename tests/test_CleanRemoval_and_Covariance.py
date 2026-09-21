"""
Test separability between learned artifact direction and clean neural signal.

For each dataset, this script:
1. Loads a clean neural period and an artifact/stimulation period.
2. Learns the first PASTd component u1 from the artifact period.
3. Computes channel covariance/correlation on the clean neural signal.
4. Projects the clean, original waveform onto u1 and reports how much original
   neural-signal energy would be removed by cancellation.

PASTd can learn from the low-pass-filtered first temporal derivative or the
component removed from the mixed signal by harmonic RLS. Clean-period energy
loss is always computed on the original non-differential signal.

Compare both methods on ERAASR with:
    python test_CleanRemoval_and_Covariance.py --dataset eraasr --preprocessing both
"""

import argparse
import csv
import os
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from algorithms import PASTd, apply_lpf_3d, derivative_time, harmonic_rls_3d
from data_loading import ROOT, load_dataset as load_shared_dataset


OUT_DIR = "plots/separability_artifactandneural"


@dataclass
class DatasetBundle:
    name: str
    fs: float
    clean: np.ndarray
    artifact_period: np.ndarray
    remove_k: int
    description: str
    stim_rate: float


DATASET_CONFIGS = {
    "swec_ethz": ("swec_synthetic", {}, "SWEC-ETHZ clean seizure_data, artifact period with stim channels [0, 1, 2, 8]"),
    "dictionary2stim": ("dictionary_2stim", {"clean_stop": 20000}, "dictionary2stim clean 0:20000, artifact 50000:70000"),
    "dictionary14stim": ("dictionary_14stim", {"clean_stop": 5000}, "dictionary14stim clean 0:5000, artifact 12220:17000, stim channels removed"),
    "stanford": ("ERAASR", {}, "stanford clean 0:1200, artifact 1600:3200"),
}


def load_dataset(name, root=ROOT):
    name = "stanford" if name == "eraasr" else name
    try:
        shared_name, kwargs, description = DATASET_CONFIGS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset '{name}'. Expected one of {sorted(DATASET_CONFIGS)}") from exc
    dataset = load_shared_dataset(shared_name, root=Path(root), **kwargs)
    if dataset.y_clean is None:
        raise ValueError(f"Dataset '{name}' does not provide a clean reference")
    clean, artifact_period = dataset.y_clean, dataset.X
    if name == "dictionary14stim":
        clean = np.delete(clean, dataset.stim_channels, axis=1)
        artifact_period = np.delete(artifact_period, dataset.stim_channels, axis=1)
    return DatasetBundle(name, dataset.fs, clean, artifact_period, 1, description, dataset.stim_rate)


def maybe_limit_trials(bundle, max_trials):
    if max_trials is None:
        return bundle
    n = min(max_trials, bundle.clean.shape[0], bundle.artifact_period.shape[0])
    if n < 2:
        raise ValueError("--max-trials must retain at least two trials")
    return replace(bundle, clean=bundle.clean[:n], artifact_period=bundle.artifact_period[:n])


def preprocess_for_past(artifact_period, fs, lpf_cutoff=500.0, lpf_order=5, normalize=True):
    x_lpf = apply_lpf_3d(artifact_period, cutoff=lpf_cutoff, fs=fs, order=lpf_order)
    x_delta = derivative_time(x_lpf, order=1)

    if normalize:
        mean = x_delta.mean(axis=(0, 2), keepdims=True)
        std = x_delta.std(axis=(0, 2), keepdims=True).clip(min=1e-12)
        x_delta = (x_delta - mean) / std

    return x_delta


def harmonic_preprocess_for_past(
    artifact_period, fs, stim_rate, max_harmonics=10, settling_time=1e-2, normalize=True,
):
    """Track the component removed by harmonic RLS from the raw mixed signal.

    RLS resets for each trial. Its output is mean-centered internally, so the
    difference also contains the input trial mean. Optional normalization uses
    training trials only, matching preprocess_for_past. No derivative or LPF is
    applied on this path.
    """
    mixed = np.asarray(artifact_period, dtype=float)
    filtered = harmonic_rls_3d(
        mixed, fs=fs, stim_rate=stim_rate,
        max_harmonics=max_harmonics, settling_time=settling_time,
    )
    artifacts = mixed - filtered
    if normalize:
        mean = artifacts.mean(axis=(0, 2), keepdims=True)
        std = artifacts.std(axis=(0, 2), keepdims=True).clip(min=1e-12)
        artifacts = (artifacts - mean) / std
    return artifacts


def learn_u1_per_trial(x_delta, beta=0.99, reorth_interval=1e10):
    n_trials, n_channels, n_timepoints = x_delta.shape
    u1s = np.zeros((n_trials, n_channels), dtype=float)

    for trial_idx in range(n_trials):
        tracker = PASTd(n_channels, 1, beta=beta, reorth_interval=reorth_interval)
        for t in range(n_timepoints):
            delta_x_t = x_delta[trial_idx, :, t]
            tracker.update(delta_x_t, track_delta=False)
        u1 = tracker.get_components()[:, 0].copy()
        u1s[trial_idx] = u1 / max(np.linalg.norm(u1), 1e-12)

    return u1s


def split_train_test_trials(n_trials, train_fraction=0.8):
    if n_trials < 2:
        raise ValueError("Need at least two trials for an 80/20 train/test split")

    n_train = int(np.floor(train_fraction * n_trials))
    n_train = min(max(n_train, 1), n_trials - 1)
    train_indices = np.arange(n_train)
    test_indices = np.arange(n_train, n_trials)
    return train_indices, test_indices


def learn_u1_from_training_trials(x_delta_train, beta=0.99, reorth_interval=1e10):
    n_trials, n_channels, n_timepoints = x_delta_train.shape
    tracker = PASTd(n_channels, 1, beta=beta, reorth_interval=reorth_interval)

    for trial_idx in range(n_trials):
        for t in range(n_timepoints):
            delta_x_t = x_delta_train[trial_idx, :, t]
            tracker.update(delta_x_t, track_delta=False)

    u1 = tracker.get_components()[:, 0].copy()
    return u1 / max(np.linalg.norm(u1), 1e-12)


def covariance_and_correlation(clean):
    x = np.transpose(clean, (1, 0, 2)).reshape(clean.shape[1], -1)

    covariance = np.cov(x, bias=False)
    std = np.sqrt(np.diag(covariance))
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.clip(covariance / np.outer(std, std), -1.0, 1.0)
    correlation_percent = 100 * np.abs(correlation)

    # Unique channel pairs only: i < j
    upper_offdiag = np.triu(
        np.ones_like(correlation, dtype=bool),
        k=1
    )

    return {
        "covariance": covariance,
        "correlation": correlation,
        "correlation_percent": correlation_percent,

        "mean_abs_corr_offdiag_percent":
            float(np.nanmean(correlation_percent[upper_offdiag])),

        "median_abs_corr_offdiag_percent":
            float(np.nanmedian(correlation_percent[upper_offdiag])),

        "max_abs_corr_offdiag_percent":
            float(np.nanmax(correlation_percent[upper_offdiag])),
    }


def clean_energy_loss(clean, u1s):
    n_trials = min(clean.shape[0], u1s.shape[0])
    rows = []
    cleaned = np.zeros_like(clean[:n_trials], dtype=float)

    for trial_idx in range(n_trials):
        x = clean[trial_idx].astype(float, copy=False)
        u1 = u1s[trial_idx]
        projected = np.outer(u1, u1 @ x)
        x_cleaned = x - projected
        cleaned[trial_idx] = x_cleaned

        original_energy = float(np.sum(x**2))
        removed_energy = float(np.sum(projected**2))
        residual_energy = float(np.sum(x_cleaned**2))
        rows.append(
            {
                "trial": trial_idx,
                "original_energy": original_energy,
                "removed_energy": removed_energy,
                "residual_energy": residual_energy,
                "energy_loss_fraction": removed_energy / original_energy if original_energy > 0 else np.nan,
                "energy_loss_db": 10.0 * np.log10(
                    residual_energy / original_energy
                )
                if original_energy > 0 and residual_energy > 0
                else np.nan,
            }
        )

    return rows, cleaned


def clean_energy_loss_fixed_u1(clean, u1, trial_indices=None):
    if trial_indices is None:
        trial_indices = np.arange(clean.shape[0])

    rows = []
    cleaned = np.zeros_like(clean, dtype=float)

    for local_idx, trial_idx in enumerate(trial_indices):
        x = clean[local_idx].astype(float, copy=False)
        projected = np.outer(u1, u1 @ x)
        x_cleaned = x - projected
        cleaned[local_idx] = x_cleaned

        original_energy = float(np.sum(x**2))
        removed_energy = float(np.sum(projected**2))
        residual_energy = float(np.sum(x_cleaned**2))
        rows.append(
            {
                "trial": int(trial_idx),
                "original_energy": original_energy,
                "removed_energy": removed_energy,
                "residual_energy": residual_energy,
                "energy_loss_fraction": removed_energy / original_energy if original_energy > 0 else np.nan,
                "energy_loss_db": 10.0 * np.log10(
                    residual_energy / original_energy
                )
                if original_energy > 0 and residual_energy > 0
                else np.nan,
            }
        )

    return rows, cleaned


def summarize_energy_rows(rows):
    loss = np.array([row["energy_loss_fraction"] for row in rows], dtype=float)
    loss_db = np.array([row["energy_loss_db"] for row in rows], dtype=float)
    return {
        "mean_loss_fraction": float(np.nanmean(loss)),
        "median_loss_fraction": float(np.nanmedian(loss)),
        "max_loss_fraction": float(np.nanmax(loss)),
        "mean_loss_percent": float(100.0 * np.nanmean(loss)),
        "mean_residual_over_original_db": float(np.nanmean(loss_db)),
    }


def run_one_dataset(dataset_name, args, preprocessing="derivative", bundle=None):
    if bundle is None:
        bundle = maybe_limit_trials(load_dataset(dataset_name, root=args.root), args.max_trials)
    train_indices, test_indices = split_train_test_trials(bundle.artifact_period.shape[0])
    # print train and test indices
    print(f"Dataset: {bundle.name}; preprocessing: {preprocessing}")
    print(f"  train indices: {train_indices}")
    print(f"  test indices: {test_indices}")

    training = bundle.artifact_period[train_indices]
    if preprocessing == "derivative":
        print(f"  Preprocessing: derivative with LPF cutoff {args.lpf_cutoff} Hz, order {args.lpf_order}, normalize={not args.no_normalize}")
        tracking_input = preprocess_for_past(
            training, bundle.fs, lpf_cutoff=args.lpf_cutoff,
            lpf_order=args.lpf_order, normalize=not args.no_normalize,
        )
        # if lpf cutoff is applied, apply it also to clean signal
        if args.lpf_cutoff is not None:
            bundle = replace(
                bundle,
                clean=apply_lpf_3d(bundle.clean, cutoff=args.lpf_cutoff, fs=bundle.fs, order=args.lpf_order),
            )

    elif preprocessing == "harmonic":
        print(f"  Preprocessing: harmonic RLS with max_harmonics {args.harmonic_max_harmonics}, settling_time {args.harmonic_settling_time}, normalize={not args.no_normalize}")
        tracking_input = harmonic_preprocess_for_past(
            training, bundle.fs, bundle.stim_rate,
            max_harmonics=args.harmonic_max_harmonics,
            settling_time=args.harmonic_settling_time,
            normalize=not args.no_normalize,
        )

        if args.lpf_cutoff is not None:
            bundle = replace(
                bundle,
                clean=apply_lpf_3d(bundle.clean, cutoff=args.lpf_cutoff, fs=bundle.fs, order=args.lpf_order),
            )
            
    else:
        raise ValueError(f"Unknown preprocessing: {preprocessing}")
    u1 = learn_u1_from_training_trials(
        tracking_input,
        beta=args.beta,
        reorth_interval=args.reorth_interval,
    )

    cov_stats = covariance_and_correlation(bundle.clean)
    energy_rows, cleaned = clean_energy_loss_fixed_u1(
        bundle.clean[test_indices],
        u1,
        trial_indices=test_indices,
    )
    energy_summary = summarize_energy_rows(energy_rows)

    os.makedirs(args.out_dir, exist_ok=True)
    suffix = "" if preprocessing == "derivative" else "_harmonic"
    out_path = os.path.join(args.out_dir, f"{bundle.name}{suffix}_separability_results.npz")
    np.savez(
        out_path,
        preprocessing=preprocessing,
        fs=bundle.fs,
        stim_rate=bundle.stim_rate,
        normalized=not args.no_normalize,
        beta=args.beta,
        reorth_interval=args.reorth_interval,
        lpf_cutoff=args.lpf_cutoff,
        lpf_order=args.lpf_order,
        harmonic_max_harmonics=args.harmonic_max_harmonics,
        harmonic_settling_time=args.harmonic_settling_time,
        u1=u1,
        u1s=u1[None, :],
        train_indices=train_indices,
        test_indices=test_indices,
        covariance=cov_stats["covariance"],
        correlation=cov_stats["correlation"],
        correlation_percent=cov_stats["correlation_percent"],
        clean_original=bundle.clean[test_indices],
        clean_after_u1_cancellation=cleaned,
        energy_rows=np.array(
            [
                [
                    row["trial"],
                    row["original_energy"],
                    row["removed_energy"],
                    row["residual_energy"],
                    row["energy_loss_fraction"],
                    row["energy_loss_db"],
                ]
                for row in energy_rows
            ],
            dtype=float,
        ),
        energy_row_columns=np.array(
            [
                "trial",
                "original_energy",
                "removed_energy",
                "residual_energy",
                "energy_loss_fraction",
                "energy_loss_db",
            ]
        ),
        description=bundle.description,
    )

    print(f"\nDataset: {bundle.name}; preprocessing: {preprocessing}")
    print(f"  {bundle.description}")
    print(f"  clean shape: {bundle.clean.shape}, artifact-period shape: {bundle.artifact_period.shape}")
    print(
        f"  train/test trials: {len(train_indices)}/{len(test_indices)} "
        f"(fixed u1 learned from training trials only)"
    )
    print(
        "  clean channel correlation abs(offdiag, upper triangle): "
        f"mean={cov_stats['mean_abs_corr_offdiag_percent']:.3f}%, "
        f"median={cov_stats['median_abs_corr_offdiag_percent']:.3f}%, "
        f"max={cov_stats['max_abs_corr_offdiag_percent']:.3f}%"
    )
    print(
        "  test clean energy removed by fixed learned u1 cancellation: "
        f"mean={energy_summary['mean_loss_percent']:.4f}%, "
        f"median={100.0 * energy_summary['median_loss_fraction']:.4f}%, "
        f"max={100.0 * energy_summary['max_loss_fraction']:.4f}%"
    )
    print(
        "  residual/original energy after cancellation: "
        f"{energy_summary['mean_residual_over_original_db']:.4f} dB"
    )
    print(f"  saved: {out_path}")

    return {
        "dataset": bundle.name,
        "preprocessing": preprocessing,
        **cov_stats,
        **energy_summary,
        "out_path": out_path,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["all", "swec_ethz", "dictionary2stim", "dictionary14stim", "stanford", "eraasr"],
        default="all",
    )
    parser.add_argument("--preprocessing", choices=["derivative", "harmonic", "both"], default="derivative")
    parser.add_argument("--harmonic-max-harmonics", type=int, default=10)
    parser.add_argument("--harmonic-settling-time", type=float, default=1e-2)
    parser.add_argument("--root", type=Path, default=ROOT, help="Project root containing data/")
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--beta", type=float, default=0.99)
    parser.add_argument("--reorth-interval", type=float, default=1e10)
    parser.add_argument("--lpf-cutoff", type=float, default=500.0)
    parser.add_argument("--lpf-order", type=int, default=5)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--out-dir", default=OUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = (
        ["swec_ethz", "dictionary2stim", "dictionary14stim", "stanford"]
        if args.dataset == "all"
        else [args.dataset]
    )
    methods = ["derivative", "harmonic"] if args.preprocessing == "both" else [args.preprocessing]
    summaries = []
    for name in datasets:
        bundle = maybe_limit_trials(load_dataset(name, root=args.root), args.max_trials)
        summaries.extend(run_one_dataset(name, args, method, bundle) for method in methods)
    print("\nSummary")
    for row in summaries:
        print(
            f"  {row['dataset']:<16} "
            f"{row['preprocessing']:<12} loss={row['mean_loss_percent']:.4f}%  "
            f"mean_corr={row['mean_abs_corr_offdiag_percent']:.3f}%"
        )

    if args.preprocessing == "both":
        comparison_path = Path(args.out_dir) / "preprocessing_comparison.csv"
        fields = ["dataset", "preprocessing", "mean_loss_percent", "median_loss_fraction",
                  "max_loss_fraction", "mean_residual_over_original_db", "out_path"]
        with comparison_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summaries)
        for derivative, harmonic in zip(summaries[::2], summaries[1::2]):
            difference = harmonic["mean_loss_percent"] - derivative["mean_loss_percent"]
            print(f"  {derivative['dataset']}: harmonic minus derivative loss = {difference:+.4f} percentage points")
        print(f"Comparison saved: {comparison_path}")


if __name__ == "__main__":
    main()
