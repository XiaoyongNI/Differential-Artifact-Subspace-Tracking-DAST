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
from covariance import add_covariance_arguments, covariance_config_kwargs
from .methods import BenchmarkConfig, METHODS, run_benchmark, pulse_times, fit_lrr_trials
from .metrics import (rest_correlations, preprocess_rest, clean_rest_output,
                      clean_energy_loss_from_output, summarize_clean_energy, ENERGY_COLUMNS)


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
    past_methods = ["past", "past_covariance_rest", "past_covariance_interpulse"]
    parser.add_argument("--methods", nargs="+", default=["linear_interpolation", "template_subtraction", "window_svd", "window_ica", "pulse", "past"], choices=[*METHODS, *past_methods])
    parser.add_argument("--compare-removal", action="store_true",
                        help="Include projection, covariance+rest and covariance+interpulse PAST runs")
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
    parser.add_argument("--past-normalization", choices=["none", "global_per_channel", "per_trial", "streaming_causal"], default="none")
    parser.add_argument("--out-dir", type=Path, default=Path("plot/benchmarks"))
    add_covariance_arguments(parser)
    args = parser.parse_args()
    methods = args.methods
    if args.compare_removal:
        methods = [m for m in methods if m not in past_methods] + past_methods
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
    if markers is None and any(m not in ("pulse", "low_rank_tv", *past_methods) for m in methods):
        parser.error("Windowed methods and offline LRR training require --stim-times or --first-pulse-sample; no pulse phase is assumed")
    all_times = None if markers is None else pulse_times(markers, dataset.X.shape[0], dataset.X.shape[-1])
    start = max(0, dataset.X.shape[0]-args.n_test) if args.n_test else 0
    test_times = None if all_times is None else all_times[start:]
    # ERAASR y_clean is a different prestimulation interval, not sample-aligned
    # ground truth for its stimulation interval. Never compare those waveforms.
    aligned = dataset.y_clean is not None and dataset.y_clean.shape == dataset.X.shape
    test = replace(dataset, X=dataset.X[start:], y_clean=dataset.y_clean[start:] if aligned else None)
    # Preserve the loader's clean/rest interval for covariance even when it is
    # not an aligned reference for existing reconstruction metrics.
    rest = dataset.baseline if dataset.baseline is not None else dataset.y_clean
    if rest is not None:
        test.baseline = rest[start:] if rest.shape[0] == dataset.X.shape[0] else rest
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = BenchmarkConfig(pre_ms=args.pre_ms, post_ms=args.post_ms, rank=args.rank,
                             template_history=args.template_history or None, ica_components=args.ica_components,
                             ica_selection=args.ica_selection, max_iters=args.max_iters, tol=args.tol,
                             lambda_tv=args.lambda_tv, rho=args.rho)
    rows = []
    if not aligned:
        print("No aligned clean reference: reconstruction metrics will be omitted.")
    if test.baseline is None:
        print("No clean/rest period: rest correlations and clean-energy metrics will be omitted.")
    for method in methods:
        training_seconds = 0.
        if method == "lrr":
            if not start:
                parser.error("LRR requires preceding training trials: use --n-test less than total trials")
            training_begin = perf_counter()
            lrr_weights = fit_lrr_trials(dataset.X[:start], all_times[:start], dataset.fs, config)
            training_seconds = perf_counter() - training_begin
        begin = perf_counter()
        offset = 0
        if method in past_methods:
            covariance_options = covariance_config_kwargs(args)
            if args.compare_removal or method != "past":
                covariance_options["removal_method"] = "projection" if method == "past" else "covariance"
                covariance_options["neural_cov_source"] = "interpulse" if method.endswith("interpulse") else "rest"
            past_config = PipelineConfig(n_test=0, rank=args.rank, preprocessing=args.past_preprocessing,
                                         normalization=args.past_normalization,
                                         use_derivative_for_tracking=not args.past_no_derivative, compute_ideal_u=False,
                                         **covariance_options)
            result = run_pipeline(test, past_config, stim_times=test_times)
            output = result.stages["cleaned"]
            offset = 0 if args.past_no_derivative else past_config.derivative_order
            method_config = asdict(past_config)
        else:
            kwargs = {}
            if method == "lrr":
                kwargs["W"] = lrr_weights
            if method == "template_subtraction" and config.template_history is None:
                if not start:
                    parser.error("Fixed template averaging requires preceding training trials: use --n-test less than total trials")
                kwargs = {"training_data": dataset.X[:start], "training_stim_times": all_times[:start]}
            method_config = asdict(replace(config, method=method))
            result = run_benchmark(test, replace(config, method=method), test_times, **kwargs)
            output = result.cleaned
        seconds = perf_counter()-begin
        raw = test.X[:, :, offset:]
        row = {"method": method, "seconds": seconds, "training_seconds": training_seconds,
               "seconds_per_sample": seconds / (output.shape[0] * output.shape[-1]), "time_offset": offset,
               "rmse": np.nan, "relative_rmse": np.nan,
               "rest_correlation_time": np.nan, "rest_correlation_freq": np.nan,
               "energy_loss_fraction": np.nan, "energy_loss_db": np.nan}
        if aligned:
            row.update(error_metrics(output, test.y_clean[:, :, offset:]))
        rest_payload = {}
        if test.baseline is not None:
            eval_config = past_config if method in past_methods else replace(config, method=method)
            rest_pre = preprocess_rest(test.baseline, method, eval_config, test.fs)
            # Compare interval starts after the same sample offset. This also
            # keeps synthetic simultaneous y_clean aligned with the output.
            rest_reference = rest_pre[:, :, offset:]
            correlations, time_r, freq_r = rest_correlations(output, rest_reference, test.fs)
            row.update(correlations)
            clean_reference, cleaned_rest = clean_rest_output(rest_pre, method, result, eval_config, test.fs, test_times)
            if clean_reference is not None:
                clean_reference, cleaned_rest = clean_reference[:, :, offset:], cleaned_rest[:, :, offset:]
                energy_rows, cleaned_rest = clean_energy_loss_from_output(clean_reference, cleaned_rest)
                row.update(summarize_clean_energy(energy_rows))
                rest_payload.update(rest_original=clean_reference, rest_after_removal=cleaned_rest,
                                    rest_energy_rows=np.array([[r[col] for col in ENERGY_COLUMNS] for r in energy_rows]),
                                    rest_energy_row_columns=np.array(ENERGY_COLUMNS))
            rest_payload.update(rest_correlation_reference=rest_reference,
                                rest_correlation_time_per_trial_channel=time_r,
                                rest_correlation_freq_per_trial_channel=freq_r,
                                energy_loss_definition="removed_energy / original_energy, averaged across trials")
        row["neural_cov_source"] = (result.diagnostics["covariance"]["source"]
                                    if method in past_methods and past_config.removal_method == "covariance" else "")
        rows.append(row)
        payload = dict(cleaned=output, artifact=raw-output, raw=raw, fs=test.fs, stim_rate=test.stim_rate,
                       trial_indices=np.arange(start, dataset.X.shape[0]), time_offset=offset,
                       config_json=json.dumps(method_config), metrics_json=json.dumps(row))
        payload.update(rest_payload)
        if method == "lrr":
            payload.update(lrr_weights=lrr_weights, training_trial_indices=np.arange(start))
        if aligned:
            payload["y_clean"] = test.y_clean[:, :, offset:]
        if test_times is not None:
            for i, times in enumerate(test_times):
                payload[f"stim_times_trial_{i}"] = times
        if method in past_methods and past_config.removal_method == "covariance":
            payload["covariance_diagnostics_json"] = json.dumps(result.diagnostics["covariance"])
            for key in ("neural_covariance_initial", "neural_covariance_final", "mixed_covariance_final", "artifact_power", "neural_safe_mask"):
                if key in result.diagnostics:
                    payload[key] = result.diagnostics[key]
            print(f"  Neural covariance: {result.diagnostics['covariance']}")
        np.savez_compressed(args.out_dir / f"{dataset.name}_{method}.npz", **payload)
        print(f"{method:<22} {seconds:.3f}s  RMSE={row['rmse']:.6g}  "
              f"rest r(time/freq)={row['rest_correlation_time']:.4f}/{row['rest_correlation_freq']:.4f}  "
              f"clean loss={row['energy_loss_fraction']:.4f}")
    with (args.out_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results: {args.out_dir}")


if __name__ == "__main__":
    main()
