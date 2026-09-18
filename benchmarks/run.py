"""Run from the project root: python -m benchmarks.run --dataset synthetic."""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
from time import perf_counter
import numpy as np

from data_loading import ROOT, Dataset, available_datasets, load_dataset
from pipeline import PipelineConfig, run_pipeline
from .methods import BenchmarkConfig, METHODS, run_benchmark, pulse_times


def synthetic_dataset():
    """Reproducible mixed signals with aligned clean reference and known triggers."""
    rng = np.random.default_rng(42)
    fs = 2000.
    time = np.arange(600) / fs
    clean = np.stack([np.sin(2*np.pi*(13+ch*4)*time + rng.uniform(0, 6, (3, 1)))
                      for ch in range(4)], axis=1) + rng.normal(0, .03, (3, 4, 600))
    triggers = np.arange(40, 560, 50)
    artifact = np.zeros_like(clean)
    waveform = 8 * np.exp(-np.arange(6)/2) * np.cos(np.arange(6)*np.pi)
    for t in triggers:
        artifact[:, :, t:t+6] += np.array([1, .8, .5, .2])[None, :, None] * waveform
    return Dataset("synthetic", clean+artifact, fs, 40., y_clean=clean), triggers


def error_metrics(output, reference):
    error = np.mean((output-reference)**2)
    power = np.mean(reference**2)
    return {"rmse": float(np.sqrt(error)), "relative_rmse": float(np.sqrt(error/power)) if power else np.nan}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="synthetic", choices=["synthetic", *sorted(available_datasets())])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--methods", nargs="+", default=["linear_interpolation", "template_subtraction", "window_svd", "window_ica", "pulse", "past"], choices=[*METHODS, "past"])
    parser.add_argument("--n-test", type=int, default=1, help="Last n trials, same selection for every method; 0 means all")
    parser.add_argument("--stim-times", type=Path, help=".npy of shared 1D or rectangular per-trial integer indices, relative to dataset.X")
    parser.add_argument("--first-pulse-sample", type=int, help="Explicit first pulse for a periodic schedule, relative to dataset.X")
    parser.add_argument("--pre-ms", type=float, default=.1)
    parser.add_argument("--post-ms", type=float, default=1.1)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--template-history", type=int, default=3, help="Previous pulses to average; 0 learns a fixed template from preceding training trials")
    parser.add_argument("--ica-components", type=int)
    parser.add_argument("--ica-selection", choices=["kurtosis", "energy"], default="kurtosis")
    parser.add_argument("--max-iters", type=int, default=200)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--lambda-tv", type=float, default=1.)
    parser.add_argument("--rho", type=float, default=1.)
    parser.add_argument("--past-preprocessing", choices=["none", "lpf", "bpf"], default="none")
    parser.add_argument("--past-no-derivative", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("plot/benchmarks"))
    args = parser.parse_args()
    if args.n_test < 0:
        parser.error("--n-test must be nonnegative")
    if args.stim_times is not None and args.first_pulse_sample is not None:
        parser.error("Use either --stim-times or --first-pulse-sample")
    if args.dataset == "synthetic":
        dataset, markers = synthetic_dataset()
    else:
        dataset, markers = load_dataset(args.dataset, root=args.root), None
    if args.stim_times is not None:
        markers = np.load(args.stim_times, allow_pickle=False)
    elif args.first_pulse_sample is not None:
        if not 0 <= args.first_pulse_sample < dataset.X.shape[-1] or not 0 < dataset.stim_rate < dataset.fs:
            parser.error("First pulse must be inside X and stimulation rate between 0 and fs")
        # Round absolute times rather than accumulating a rounded period.
        count = int(np.ceil(dataset.X.shape[-1]*dataset.stim_rate/dataset.fs))
        markers = np.rint(args.first_pulse_sample + np.arange(count)*dataset.fs/dataset.stim_rate).astype(int)
        markers = np.unique(markers[markers < dataset.X.shape[-1]])
    if markers is None and any(m not in ("pulse", "low_rank_tv", "past") for m in args.methods):
        parser.error("Windowed methods require --stim-times or --first-pulse-sample; no pulse phase is assumed")
    all_times = None if markers is None else pulse_times(markers, dataset.X.shape[0], dataset.X.shape[-1])
    start = max(0, dataset.X.shape[0]-args.n_test) if args.n_test else 0
    test_times = None if all_times is None else all_times[start:]
    # ERAASR y_clean is a different prestimulation interval, not sample-aligned
    # ground truth for its stimulation interval. Never compare those waveforms.
    aligned = dataset.y_clean is not None and dataset.y_clean.shape == dataset.X.shape
    test = replace(dataset, X=dataset.X[start:], y_clean=dataset.y_clean[start:] if aligned else None)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = BenchmarkConfig(pre_ms=args.pre_ms, post_ms=args.post_ms, rank=args.rank,
                             template_history=args.template_history or None, ica_components=args.ica_components,
                             ica_selection=args.ica_selection, max_iters=args.max_iters, tol=args.tol,
                             lambda_tv=args.lambda_tv, rho=args.rho)
    rows = []
    if not aligned:
        print("No aligned clean reference: reconstruction metrics will be omitted.")
    for method in args.methods:
        begin = perf_counter()
        offset = 0
        if method == "past":
            past_config = PipelineConfig(n_test=0, rank=args.rank, preprocessing=args.past_preprocessing,
                                         use_derivative_for_tracking=not args.past_no_derivative, compute_ideal_u=False)
            result = run_pipeline(test, past_config)
            output = result.stages["cleaned"]
            offset = 0 if args.past_no_derivative else past_config.derivative_order
            method_config = asdict(past_config)
        else:
            kwargs = {}
            if method == "template_subtraction" and config.template_history is None:
                if not start:
                    parser.error("Fixed template averaging requires preceding training trials: use --n-test less than total trials")
                kwargs = {"training_data": dataset.X[:start], "training_stim_times": all_times[:start]}
            method_config = asdict(replace(config, method=method))
            result = run_benchmark(test, replace(config, method=method), test_times, **kwargs)
            output = result.cleaned
        seconds = perf_counter()-begin
        raw = test.X[:, :, offset:]
        row = {"method": method, "seconds": seconds, "time_offset": offset,
               "removed_energy_fraction": float(np.sum((raw-output)**2)/max(np.sum(raw**2), 1e-12)),
               "rmse": np.nan, "relative_rmse": np.nan}
        if aligned:
            row.update(error_metrics(output, test.y_clean[:, :, offset:]))
        rows.append(row)
        payload = dict(cleaned=output, artifact=raw-output, raw=raw, fs=test.fs, stim_rate=test.stim_rate,
                       trial_indices=np.arange(start, dataset.X.shape[0]), time_offset=offset,
                       config_json=json.dumps(method_config), metrics_json=json.dumps(row))
        if aligned:
            payload["y_clean"] = test.y_clean[:, :, offset:]
        if test_times is not None:
            for i, times in enumerate(test_times):
                payload[f"stim_times_trial_{i}"] = times
        np.savez_compressed(args.out_dir / f"{dataset.name}_{method}.npz", **payload)
        print(f"{method:<22} {seconds:.3f}s  RMSE={row['rmse']:.6g}")
    with (args.out_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results: {args.out_dir}")


if __name__ == "__main__":
    main()
