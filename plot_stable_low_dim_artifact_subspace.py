"""Spatial artifact dimensionality and stability using the existing TBME loaders.

Run from any directory: python plot_stable_low_dim_artifact_subspace.py --help
Real-data estimates are pulse averages, not isolated artifact ground truth.
Pulse landmarks are detected (not supplied timestamps); see events.csv. Fixed
recording-wide normalization avoids changing coordinates between blocks.
Synthetic uses the repository's precomputed model; the identity of manuscript
Ref. [10] cannot be verified from the loader and must be checked by the author.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

from algorithms import normalize_channels
from data_loading import ROOT, Dataset, _load_xlsx_voltage_columns, load_dataset
from plotting import TBME_COLORS, set_tbme_style

LABELS = ["Synthetic", "PBS", "ERAASR", "Dictionary learning"]



def load_pbs_artifact(root, configuration_number):
    """Reuse PBS reader and load_swec_pbs crop/resampling without neural mixing.

    Matches the 0.1--1.0 s crop and 16 kHz to 10.24 kHz linear resampling.
    Avoids loading the synthetic archive only to subtract its clean signal.
    """
    path = root / f"data/PBS_data/Raw-Data/Data{configuration_number}.xlsx"
    pbs = _load_xlsx_voltage_columns(path)[1600:16000]
    source_t = np.arange(len(pbs)) / 16000.
    target_t = np.arange(int(len(pbs) * 10240 / 16000)) / 10240.
    artifact = interp1d(source_t, pbs, axis=0, kind="linear")(target_t).T[None]
    return Dataset(name=f"swec_pbs_config{configuration_number}", X=artifact,
                   fs=10240., stim_rate=250., metadata={"pbs_path": str(path)})

def compute_spatial_svd(X):
    """Return channel-domain left singular vectors and singular values of C x T."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or min(X.shape) < 1 or not np.isfinite(X).all():
        raise ValueError("SVD needs a finite, nonempty channels x samples matrix")
    # Scaling protects squared-energy calculations without changing orientation.
    scale = np.max(np.abs(X))
    if scale <= np.finfo(float).tiny:
        raise ValueError("Zero-energy artifact matrix")
    U, s, _ = np.linalg.svd(X / scale, full_matrices=False)
    return U, s


def cumulative_explained_energy(s):
    energy = np.square(s)
    if not np.isfinite(energy).all() or energy.sum() <= 0:
        raise ValueError("Invalid singular-value energy")
    return np.minimum(1., np.cumsum(energy) / energy.sum())


def dominant_vector_stability(reference, bases):
    """Absolute cosine removes the arbitrary SVD sign."""
    global_score = np.array([abs(reference[:, 0] @ u[:, 0]) for u in bases])
    consecutive = np.array([abs(a[:, 0] @ b[:, 0])
                            for a, b in zip(bases[:-1], bases[1:])])
    return np.clip(global_score, 0, 1), np.clip(consecutive, 0, 1)


def subspace_stability(reference, bases, k):
    """Projection overlap is invariant to signs and rotations within the basis."""
    def overlap(a, b):
        if min(a.shape[1], b.shape[1]) < k:
            return np.nan  # Never fill a deficient subspace with arbitrary vectors.
        return float(np.clip(np.linalg.norm(a[:, :k].T @ b[:, :k], "fro")**2 / k, 0, 1))
    return (np.array([overlap(reference, u) for u in bases]),
            np.array([overlap(a, b) for a, b in zip(bases[:-1], bases[1:])]))


def extract_stimulation_locked_artifact(X, fs, stim_rate, trials_per_block=5,
                                       pulse_fraction=0.8, markers=None):
    """Average complete pulse epochs within disjoint consecutive trial blocks.

    Optional markers: one array of zero-based cropped-signal samples per trial.
    Otherwise detect absolute derivative peaks on the strongest-recurring-derivative
    channel, following the repository's signal-derived landmark approach
    (dictionary-learning/+analyFunc/get_artifact_indices.m). This is a simpler
    Python detector, not an exact port. Known rate constrains spacing; irregular
    spacing is flagged. All channels use the SAME landmark, preserving phase.
    No fabricated regular pulse train is substituted for missed detections.
    """
    X = np.asarray(X)
    if X.ndim != 3 or not np.isfinite(X).all():
        raise ValueError("Extraction requires finite trials x channels x samples")
    if trials_per_block < 2 or len(X) < 2 * trials_per_block:
        raise ValueError("Need at least two disjoint blocks with >=2 trials each")
    if not 0 < pulse_fraction < 1 or fs <= 0 or stim_rate <= 0:
        raise ValueError("Invalid pulse window or sampling/stimulation rate")
    period = fs / stim_rate
    width = max(2, int(round(period * pulse_fraction)))
    pre = width // 4
    # Robust recurring strength avoids selecting an isolated saturation transient.
    strength = np.median(np.quantile(np.abs(np.diff(X, axis=2)), .98, axis=2), axis=0)
    channel = int(np.argmax(strength))
    if markers is None:
        # A sparse transient can still inflate a quantile when pulses are brief.
        # Require repeated, rate-consistent landmarks on the first trial.
        for candidate in np.argsort(strength)[::-1]:
            score = np.abs(np.diff(X[0, candidate], prepend=X[0, candidate, 0]))
            peaks, _ = find_peaks(score, distance=max(1, int(.7 * period)),
                                  height=score.mean() + 1.5 * score.std())
            if (len(peaks) >= max(2, int(.5 * X.shape[2] / period))
                    and np.mean(np.abs(np.diff(peaks) / period - 1) < .2) >= .8):
                channel = int(candidate)
                break
        else:
            raise ValueError("No channel provides recurring rate-consistent pulse landmarks")

    if markers is not None and len(markers) != len(X):
        raise ValueError("Markers must contain one list per trial")
    epochs, events = [], []
    for trial, x in enumerate(X):
        if markers is None:
            score = np.abs(np.diff(x[channel], prepend=x[channel, 0]))
            peaks, _ = find_peaks(score, distance=max(1, int(0.7 * period)),
                                  height=score.mean() + 1.5 * score.std())
        else:
            p = np.asarray(markers[trial])
            if not np.isfinite(p).all() or np.any(p != np.floor(p)):
                raise ValueError("Markers must be finite integer sample indices")
            peaks = np.unique(p.astype(int))
        peaks = peaks[(peaks >= pre) & (peaks + width - pre <= x.shape[1])]
        if len(peaks) < 2:
            raise ValueError(f"Trial {trial}: fewer than two complete pulse epochs")
        regular = np.mean(np.abs(np.diff(peaks) / period - 1) < 0.2)
        if regular < 0.8:
            warnings.warn(f"Trial {trial}: only {regular:.0%} of intervals match known pulse rate")
        epochs.append(np.stack([x[:, p-pre:p-pre+width] for p in peaks]).mean(axis=0))
        events.extend(dict(trial=trial, sample=int(p), detection_channel=channel,
                           method="provided" if markers is not None else "derivative_peak",
                           regular_interval_fraction=float(regular)) for p in peaks)
    # Use every trial; merge a trailing singleton into the preceding group.
    groups = [epochs[i:i+trials_per_block]
              for i in range(0, len(epochs), trials_per_block)]
    if len(groups[-1]) == 1:
        groups[-2].extend(groups.pop())
    blocks = [np.mean(group, axis=0) for group in groups]
    return blocks, events


def _summary(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict.fromkeys(["mean", "median", "sd", "p05", "p95"], np.nan)
    return dict(mean=float(values.mean()), median=float(np.median(values)),
                sd=float(values.std()), p05=float(np.percentile(values, 5)),
                p95=float(np.percentile(values, 95)))


def summarize_subspace_statistics(blocks, normalization="global_per_channel", rank=None):
    if len(blocks) < 2:
        raise ValueError("At least two independent windows required for stability")
    full = np.concatenate(blocks, axis=1)
    if not np.isfinite(full).all():
        raise ValueError("NaN/inf in artifact recording")
    sd = full.std(axis=1)
    keep = sd > max(1e-12, float(sd.max()) * 1e-10)
    if not keep.any():
        raise ValueError("All channels have zero/near-zero variance")
    full, norm = normalize_channels(full[keep][None], mode=normalization)
    full = full[0]
    transformed = [b[keep].copy() for b in blocks]
    if normalization != "none":
        transformed = [(b - norm.mean[0]) / norm.scale[0] for b in transformed]
    U, s = compute_spatial_svd(full)
    curve = cumulative_explained_energy(s)
    ranks = {f"rank{q}": int(np.searchsorted(curve, q / 100) + 1) for q in (90, 95, 99)}
    k = ranks["rank95"] if rank is None else rank
    numerical_rank = int(np.sum(s > s[0] * max(full.shape) * np.finfo(float).eps))
    if not 1 <= k <= numerical_rank:
        raise ValueError(f"Requested k={k} exceeds identifiable rank {numerical_rank}")
    bases = []
    for b in transformed:
        u, sb = compute_spatial_svd(b)
        r = int(np.sum(sb > sb[0] * max(b.shape) * np.finfo(float).eps))
        bases.append(u[:, :r])
    dominant, dominant_next = dominant_vector_stability(U, bases)
    overlap, overlap_next = subspace_stability(U, bases, k)
    if not np.isfinite(overlap).all():
        warnings.warn("Some windows cannot identify selected k; overlap saved as NaN")
    stats = dict(channels=int(keep.sum()), dropped_channels=int((~keep).sum()),
                 windows=len(blocks), k=k, **ranks,
                 **{f"E{i}": float(curve[i-1]) if i <= len(curve) else np.nan for i in (1, 2, 3)})
    for name, values in (("dominant", dominant), ("overlap", overlap),
                         ("dominant_consecutive", dominant_next), ("overlap_consecutive", overlap_next)):
        stats.update({f"{name}_{key}": val for key, val in _summary(values).items()})
    return dict(stats=stats, curve=curve, dominant=dominant, overlap=overlap,
                dominant_consecutive=dominant_next, overlap_consecutive=overlap_next,
                retained_channels=np.flatnonzero(keep).tolist())


def _band(values, aggregation):
    if aggregation == "median_iqr":
        return np.nanmedian(values, axis=0), *np.nanpercentile(values, [25, 75], axis=0)
    mean, sd = np.nanmean(values, axis=0), np.nanstd(values, axis=0)
    return mean, mean - sd, mean + sd


def plot_artifact_subspace_analysis(results, output, aggregation="mean_sd", max_plot_rank=12):
    set_tbme_style()
    plt.rcParams.update({"axes.grid": False, "font.size": 8, "axes.labelsize": 8})
    panels = [("spatial_energy", "Spatial rank, k", "Cumulative energy"),
              ("dominant_direction", "Recording progression (%)", "Absolute cosine"),
              ("subspace_overlap", "Recording progression (%)", "Projection overlap")]
    figures = [plt.subplots(figsize=(3.5, 3.0)) for _ in panels]
    axes = [ax for _, ax in figures]
    zoom = axes[0].inset_axes([.51, .15, .45, .42])
    colors = [TBME_COLORS[k] for k in ("raw", "after_subspace", "preprocessed", "normalized")]
    for label, color in zip(LABELS, colors):
        rows = [r for r in results if r["dataset"] == label]
        if not rows:
            continue
        trials = sum(r["trials"] for r in rows)
        legend_label = f"{label} ({trials} {'trial' if trials == 1 else 'trials'})"
        # Do not extend curves beyond a recording's available spatial rank.
        maxrank = max(len(r["curve"]) for r in rows)
        curves = np.full((len(rows), maxrank), np.nan)
        for i, r in enumerate(rows):
            curves[i, :len(r["curve"])] = r["curve"]
        center, lo, hi = _band(curves, aggregation)
        x = np.arange(1, maxrank + 1)
        for ax in (axes[0], zoom):
            ax.plot(x, center, color=color, label=legend_label)
            ax.fill_between(x, np.clip(lo, 0, 1), np.clip(hi, 0, 1), color=color, alpha=.16)
        # Acquisition order is not elapsed wall time; keep recordings separate.
        for i, r in enumerate(rows):
            x = np.linspace(0, 100, len(r["dominant"]))
            for ax, metric in zip(axes[1:], ("dominant", "overlap")):
                ax.plot(x, r[metric], color=color, alpha=.7, marker=".", markersize=2,
                        label=legend_label if i == 0 else "_nolegend_")
    axes[0].set_xlim(1, max(4, max_plot_rank))
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    axes[0].axhline(.95, color=".65", ls=":", lw=.6)
    zoom.set(xlim=(1, 4), ylim=(.8, 1), xticks=[1, 2, 3, 4], yticks=[.8, .9, 1.])
    zoom.tick_params(labelsize=7, pad=2)
    zoom.axhline(.95, color=".65", ls=":", lw=.6)
    axes[0].indicate_inset_zoom(zoom, edgecolor=".5", alpha=.5)
    for (fig, ax), (suffix, xlabel, ylabel) in zip(figures, panels):
        ax.set(xlabel=xlabel, ylabel=ylabel, ylim=(0, 1.025))
        if ax is not axes[0]:
            ax.set_xlim(0, 100)
            ax.set_ylim(0.98, 1.005)
            ax.set_yticks([0.98, 0.99, 1.00])
        fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=2,
                   frameon=False, fontsize=7, columnspacing=1.)
        fig.subplots_adjust(left=.18, right=.97, top=.96, bottom=.32)
        for ext in ("pdf", "png"):
            fig.savefig(output.parent / f"{output.name}_{suffix}.{ext}", dpi=600, facecolor="white")
        plt.close(fig)


def _write_csv(path, rows):
    if rows:
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="Project directory containing data/ (default: this script's directory)")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "plots/stable_low_dim_artifact_subspace")
    parser.add_argument("--datasets", nargs="+", choices=["synthetic", "pbs", "eraasr", "dictionary"], default=["synthetic", "pbs", "eraasr", "dictionary"])
    parser.add_argument("--stim-channels", type=int, nargs="+", default=[0, 1, 2, 8])
    parser.add_argument("--pbs-configurations", type=int, nargs="+", default=list(range(14)))
    parser.add_argument("--dictionary-configurations", nargs="+", choices=["2", "4", "5", "14"], default=["2"])
    parser.add_argument("--normalization", choices=["global_per_channel", "none"], default="global_per_channel")
    parser.add_argument("--skip-unnormalized", action="store_true")
    parser.add_argument("--aggregation", choices=["mean_sd", "median_iqr"], default="mean_sd")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--max-plot-rank", type=int, default=12)
    parser.add_argument("--window-s", type=float, default=.1)
    parser.add_argument("--trials-per-block", type=int, default=5)
    parser.add_argument("--pulse-fraction", type=float, default=.8)
    parser.add_argument("--markers-json", type=Path, help="Mapping loader dataset name to per-trial zero-based pulse sample lists, relative to loader crop")
    args = parser.parse_args()
    if not np.isfinite(args.window_s) or args.window_s <= 0:
        parser.error("--window-s must be finite and positive")
    jobs = []
    if "synthetic" in args.datasets:
        jobs.append(("Synthetic", "swec_synthetic", dict(stim_channels=args.stim_channels)))
    if "pbs" in args.datasets:
        jobs.extend(("PBS", "swec_pbs", dict(configuration_number=c)) for c in args.pbs_configurations)
    if "eraasr" in args.datasets:
        jobs.append(("ERAASR", "ERAASR", {}))
    if "dictionary" in args.datasets:
        jobs.extend(("Dictionary learning", f"dictionary_{c}stim", {}) for c in args.dictionary_configurations)
    # Validate the smaller recordings before loading the large synthetic archive.
    jobs.sort(key=lambda job: job[0] == "Synthetic")
    markers = json.loads(args.markers_json.read_text()) if args.markers_json else {}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results, event_rows, provenance = [], [], []
    modes = [args.normalization]
    if args.normalization != "none" and not args.skip_unnormalized:
        modes.append("none")
    for label, name, options in jobs:
        print(f"Loading {name} {options}", flush=True)
        ds = (load_pbs_artifact(args.root, **options) if label == "PBS"
              else load_dataset(name, root=args.root, **options))
        print(f"  shape={ds.X.shape}, fs={ds.fs:g}", flush=True)
        if label in ("Synthetic", "PBS"):
            if label == "Synthetic" and (ds.y_clean is None or ds.X.shape != ds.y_clean.shape):
                raise ValueError("Pure artifact extraction needs simultaneous clean ground truth")
            artifact = ds.X if label == "PBS" else ds.X - ds.y_clean
            width = max(2, round(args.window_s * ds.fs))
            blocks = [trial[:, i:i+width] for trial in artifact
                      for i in range(0, trial.shape[1] - width + 1, width)]
            method = ("Physical PBS, existing crop/resampling" if label == "PBS" else
                      "X - simultaneous y_clean") + "; nonoverlapping windows, tail discarded"
        else:
            blocks, events = extract_stimulation_locked_artifact(
                ds.X, ds.fs, ds.stim_rate, args.trials_per_block,
                args.pulse_fraction, markers.get(ds.name))
            event_rows.extend(dict(recording=ds.name, **e) for e in events)
            method = "Pulse averages within disjoint trial blocks; possible evoked neural residual"
        provenance.append(dict(recording=ds.name, shape=list(ds.X.shape), fs=ds.fs,
                               stim_rate=ds.stim_rate, extraction=method, options=options))
        for mode in modes:
            result = summarize_subspace_statistics(blocks, mode, args.rank)
            result.update(dataset=label, recording=ds.name, normalization=mode, trials=int(ds.X.shape[0]))
            results.append(result)
        del ds, blocks
    for mode in modes:
        selected = [r for r in results if r["normalization"] == mode]
        plot_artifact_subspace_analysis(selected, args.out_dir / f"artifact_subspace_{mode}", args.aggregation, args.max_plot_rank)
    _write_csv(args.out_dir / "recording_statistics.csv", [dict(dataset=r["dataset"], recording=r["recording"], normalization=r["normalization"], trials=r["trials"], **r["stats"]) for r in results])
    _write_csv(args.out_dir / "events.csv", event_rows)
    _write_csv(args.out_dir / "energy_curves.csv", [dict(recording=r["recording"], normalization=r["normalization"], rank=i+1, energy=float(e)) for r in results for i, e in enumerate(r["curve"])])
    _write_csv(args.out_dir / "window_metrics.csv", [dict(recording=r["recording"], normalization=r["normalization"], window=i+1, dominant=float(d), overlap=float(o), dominant_consecutive=float(r["dominant_consecutive"][i-1]) if i else np.nan, overlap_consecutive=float(r["overlap_consecutive"][i-1]) if i else np.nan) for r in results for i, (d, o) in enumerate(zip(r["dominant"], r["overlap"]))])
    summary = []
    for mode in modes:
        for label in LABELS:
            rows = [r["stats"] for r in results if r["dataset"] == label and r["normalization"] == mode]
            if not rows:
                continue
            row = dict(dataset=label, normalization=mode, recordings=len(rows),
                       trials=sum(r["trials"] for r in results if r["dataset"] == label and r["normalization"] == mode))
            for key in ("E1", "E2", "E3", "rank95", "dominant_mean", "overlap_mean"):
                values = [r[key] for r in rows]
                row[key] = float(np.nanmean(values) if args.aggregation == "mean_sd" else np.nanmedian(values))
            summary.append(row)
            print(row)
    _write_csv(args.out_dir / "dataset_summary.csv", summary)
    report = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, recordings=provenance,
                  retained_channels={r["recording"]: r["retained_channels"] for r in results},
                  notes=["No temporal filtering or derivative transform before spatial SVD.",
                         "Global reference uses concatenated independent block estimates (includes tested blocks); consecutive metrics also provided.",
                         "Global channel mean/std fitted once per recording to artifact blocks; none uses uncentered voltage energy. Pipeline default itself is none.",
                         "Trial indices are acquisition order, not known elapsed wall time. Single synthetic source is one recording, not independent replicates.",
                         "Dataset summaries weight recordings equally; rank95 summary can be fractional. Recording CSV has integer ranks.",
                         "Real-data landmarks inferred unless supplied; averaging cannot remove phase-locked neural responses.",
                         "Synthetic precomputed artifact model reused; manuscript Ref. [10] identity requires author verification."])
    (args.out_dir / "analysis_metadata.json").write_text(json.dumps(report, indent=2))
    print(f"Saved figures and statistics to {args.out_dir}")


if __name__ == "__main__":
    main()
