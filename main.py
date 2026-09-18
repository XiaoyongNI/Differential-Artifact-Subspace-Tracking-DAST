"""Run the clean TBME artifact-cancellation pipeline.

Examples
--------
python main.py --dataset swec_synthetic --stim-channels 0 1 2 8 --rank 4 --adaptive-k --harmonic-filter
python main.py --dataset ERAASR --no-derivative --normalization global_per_channel
"""

from __future__ import annotations

import argparse
from pathlib import Path

from data_loading import ROOT, available_datasets, load_dataset
from pipeline import PipelineConfig, run_pipeline
from plotting import quick_report_plots


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean TBME artifact-cancellation runner.")
    p.add_argument("--dataset", default="swec_synthetic", choices=sorted(available_datasets()))
    p.add_argument("--root", default=ROOT, type=Path,
                   help="Project directory containing the data/ folder.")
    p.add_argument("--n-test", default=1, type=int)
    p.add_argument("--stim-channels", nargs="*", type=int, default=None)
    p.add_argument("--artifact-scale", default=1.0, type=float)
    p.add_argument("--preprocessing", default="lpf", choices=["none", "lpf", "bpf"])
    p.add_argument("--lpf-cutoff", default=500.0, type=float)
    p.add_argument("--bpf-low", default=500.0, type=float)
    p.add_argument("--bpf-high", default=8000.0, type=float)
    p.add_argument("--normalization", default="none", choices=["none", "global_per_channel", "per_trial", "streaming_causal"])
    p.add_argument("--no-derivative", action="store_true")
    p.add_argument("--tracker", default="PASTd", choices=["PASTd", "MCC_OPAST"])
    p.add_argument("--rank", default=1, type=int)
    p.add_argument("--beta", default=0.999, type=float)
    p.add_argument("--adaptive-k", action="store_true")
    p.add_argument("--adaptive-k-threshold", default=0.1, type=float)
    p.add_argument("--harmonic-filter", action="store_true")
    p.add_argument("--harmonics", default=10, type=int)
    p.add_argument("--harmonic-settling-time", default=1e-2, type=float)
    p.add_argument("--plot-dir", default=Path("TBME_clean_code/plots"), type=Path)
    p.add_argument("--save-npz", default=None, type=Path)
    p.add_argument("--trial", default=0, type=int)
    p.add_argument("--channel", default=0, type=int)
    return p.parse_args()


def dataset_kwargs(args: argparse.Namespace) -> dict:
    if args.dataset == "swec_synthetic":
        kwargs = {"artifact_scale": args.artifact_scale}
        if args.stim_channels is not None:
            kwargs["stim_channels"] = args.stim_channels
        return kwargs
    if args.dataset == "swec_pbs":
        return {"artifact_scale": args.artifact_scale}
    return {}


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset, root=args.root, **dataset_kwargs(args))
    config = PipelineConfig(
        n_test=args.n_test,
        preprocessing=args.preprocessing,
        lpf_cutoff_hz=args.lpf_cutoff,
        bpf_low_hz=args.bpf_low,
        bpf_high_hz=args.bpf_high,
        normalization=args.normalization,
        use_derivative_for_tracking=not args.no_derivative,
        tracker=args.tracker,
        rank=args.rank,
        beta=args.beta,
        adaptive_k=args.adaptive_k,
        adaptive_k_threshold=args.adaptive_k_threshold,
        harmonic_filter=args.harmonic_filter,
        harmonic_max_harmonics=args.harmonics,
        harmonic_settling_time=args.harmonic_settling_time,
        save_npz=args.save_npz,
    )
    result = run_pipeline(dataset, config)
    quick_report_plots(result, args.plot_dir, trial=args.trial, channel=args.channel)
    print(f"Dataset: {dataset.name}")
    print(f"Shape raw -> cleaned: {result.stages['raw'].shape} -> {result.stages['cleaned'].shape}")
    print(f"Harmonic energy ratio before: {result.diagnostics['harmonic_ratio_before']:.6f}")
    print(f"Harmonic energy ratio after:  {result.diagnostics['harmonic_ratio_after']:.6f}")
    print(f"Plots written to: {args.plot_dir}")
    if args.save_npz:
        print(f"Result npz written to: {args.save_npz}")


if __name__ == "__main__":
    main()

