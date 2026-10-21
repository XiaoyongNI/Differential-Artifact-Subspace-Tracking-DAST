"""Paired artifact/neural spatial overlap, with and without time differences.

python tests/plot_differentiation_subspace_separation.py --dataset all

SWEC uses patient HDF5 loaders and the artifact generator already used by the
seizure-preservation experiment. Defaults select up to six two-second clips per
patient, then nonoverlapping 100-ms windows. Native EEG is resampled to the
configured rate (1024 Hz by default); interpolation adds no neural bandwidth.
PBS uses the existing isolated-artifact loader, resampled to that same rate,
and the first eight neural channels, as in load_swec_pbs.

Both normalization modes center each channel within each analyzed segment.
``none`` means centered, unscaled voltage; ``global_per_channel`` additionally
uses a fixed neural-reference channel SD fitted across the parent clip. The
SAME scale is applied to artifact and neural signals before either condition.
The overlap is rank-normalized in both modes; its unnormalized numerator is
also saved. Channel scaling changes spatial geometry and is a sensitivity
analysis, not an amplitude-only adjustment.

Real recordings are exploratory: existing pulse averages estimate artifact
(including possible evoked neural activity), paired with prestimulation data
from the same trial blocks. Short pulse epochs replace 100-ms artifact windows;
neural rest is never subtracted from a different stimulation interval. Each
continuous rest trial is differentiated before concatenation, avoiding false
derivatives across trial boundaries. These results have separate figures.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample_poly
from scipy.stats import wilcoxon
from threadpoolctl import threadpool_limits

from algorithms import derivative_time, normalize_channels
from data_loading import ROOT, load_dataset
from plotting import TBME_COLORS, set_tbme_style
from experiments.seizure_preservation.common import derived_seed, write_csv, write_json
from experiments.seizure_preservation.data import load_patient_data, split_patient_data, extract_windows
from experiments.seizure_preservation.cancellation import generate_synthetic_artifact
from experiments.seizure_preservation.reporting import _holm
from plot_stable_low_dim_artifact_subspace import (
    compute_spatial_svd, cumulative_explained_energy, subspace_stability,
    load_pbs_artifact, extract_stimulation_locked_artifact,
)

MODES = ('none', 'global_per_channel')
CONDITIONS = ('none', 'derivative')
PAIR_KEYS = ('dataset', 'subject', 'configuration', 'trial', 'window', 'normalization')
ROW_FIELDS = (*PAIR_KEYS, 'reference_type', 'unit', 'condition', 'artifact_rank',
              'neural_rank', 'subspace_overlap', 'unnormalized_projection_overlap',
              'artifact_samples', 'neural_samples', 'channels', 'start_sample', 'fs')


def projection_overlap(ua, un):
    """Use existing equal-rank helper, extending only the unequal-rank case.

    Truncating BOTH bases to min(rank) would discard selected directions and
    compute the wrong metric when the energy-selected ranks differ.
    """
    ka, kn = ua.shape[1], un.shape[1]
    if min(ka, kn) < 1:
        raise ValueError('Empty subspace')
    numerator = float(np.linalg.norm(ua.T @ un, 'fro') ** 2)
    raw = numerator / min(ka, kn)
    if not np.isfinite(raw) or not -1e-10 <= raw <= 1 + 1e-10:
        raise ValueError(f'Invalid overlap: {raw}')
    if ka == kn:
        value = float(subspace_stability(ua, [un], ka)[0][0])
    else:
        value = float(np.clip(raw, 0, 1))
    return value, numerator


def estimate(matrix, threshold):
    if matrix.shape[-1] < 3:
        raise ValueError('insufficient_samples')
    if not np.isfinite(matrix).all():
        raise ValueError('nonfinite_samples')
    centered = matrix - matrix.mean(axis=-1, keepdims=True)
    peak = float(np.max(np.abs(matrix)))
    if np.max(np.abs(centered)) <= max(np.finfo(float).tiny, peak * 1e-12):
        raise ValueError('zero_or_negligible_centered_energy')
    u, s = compute_spatial_svd(centered)
    rank = min(len(s), int(np.searchsorted(cumulative_explained_energy(s), threshold) + 1))
    numerical_rank = int(np.sum(s > s[0] * max(centered.shape) * np.finfo(float).eps))
    if rank > numerical_rank:
        raise ValueError('energy_rank_exceeds_numerical_rank')
    return u[:, :rank], s


def evaluate_pair(artifact_segments, neural_segments, scale, threshold):
    """All four SVDs succeed or the entire pair is excluded.

    Lists denote separate continuous epochs, never artificial adjacent samples.
    Scaling and centering commute with differencing; recenter each transformed
    segment to estimate fluctuation subspaces consistently in both conditions.
    """
    outputs, spectra = [], {}
    for condition in CONDITIONS:
        matrices = []
        for segments in (artifact_segments, neural_segments):
            transformed = []
            for x in segments:
                if x.ndim != 2 or x.shape[0] != len(scale) or x.shape[-1] < 4:
                    raise ValueError('insufficient_samples_or_channel_mismatch')
                if not np.isfinite(x).all():
                    raise ValueError('nonfinite_samples')
                x = np.asarray(x, dtype=float) / scale[:, None]
                x = derivative_time(x) if condition == 'derivative' else x
                transformed.append(x - x.mean(axis=-1, keepdims=True))
            matrices.append(np.concatenate(transformed, axis=1))
        ua, sa = estimate(matrices[0], threshold)
        un, sn = estimate(matrices[1], threshold)
        overlap, numerator = projection_overlap(ua, un)
        outputs.append(dict(condition=condition, artifact_rank=ua.shape[1], neural_rank=un.shape[1],
                            subspace_overlap=overlap, unnormalized_projection_overlap=numerator,
                            artifact_samples=matrices[0].shape[1], neural_samples=matrices[1].shape[1],
                            channels=len(scale)))
        spectra[condition] = (sa, sn)
    return outputs, spectra


def reference_scale(neural_segments):
    full = np.concatenate(neural_segments, axis=-1)
    if not np.isfinite(full).all():
        raise ValueError('nonfinite_normalization_reference')
    _, stats = normalize_channels(full[None], mode='global_per_channel')
    sd = full.std(axis=-1)
    # Keep the channel set identical in all comparisons; reject a parent clip
    # rather than amplify channels with unidentifiable neural variance.
    if np.any(sd <= max(1e-12, float(sd.max()) * 1e-10)):
        raise ValueError('zero_or_negligible_neural_channel_variance')
    return stats.scale[0, :, 0]


class Experiment:
    def __init__(self, args):
        self.args = args
        self.rows, self.exclusions, self.skipped, self.provenance, self.events = [], [], [], [], []
        self.diagnostic = {}

    def skip(self, dataset, subject, reason):
        row = dict(dataset=dataset, subject=str(subject), reason=str(reason))
        self.skipped.append(row)
        print('SKIP', row, flush=True)

    def exclude(self, meta, mode, reason):
        self.exclusions.append({**{k: meta.get(k, '') for k in PAIR_KEYS},
                                'normalization': mode, 'reason': str(reason)})

    def pair(self, artifact, neural, scales, meta):
        for mode in self.args.normalization:
            try:
                scale = scales[mode]
                if isinstance(scale, str):
                    raise ValueError(scale)
                outputs, spectra = evaluate_pair(artifact, neural, scale, self.args.energy_threshold)
            except (ValueError, np.linalg.LinAlgError) as exc:
                self.exclude(meta, mode, exc)
                continue
            self.rows.extend(dict(**meta, normalization=mode, **row) for row in outputs)
            if meta['dataset'].startswith('swec_') and mode not in self.diagnostic:
                self.diagnostic[mode] = (dict(meta), spectra)

    def scales(self, neural):
        scales = {'none': np.ones(neural[0].shape[0])}
        try:
            scales['global_per_channel'] = reference_scale(neural)
        except ValueError as exc:
            scales['global_per_channel'] = str(exc)
        return scales

    def windows(self, artifact, neural, meta):
        if artifact.shape != neural.shape:
            raise ValueError('Artifact and neural ground truth must have identical shapes')
        width = round(self.args.window_ms * meta['fs'] / 1000)
        scales = self.scales([neural])
        for i, start in enumerate(range(0, neural.shape[-1], width)):
            m = dict(meta, window=i, start_sample=start)
            if start + width > neural.shape[-1]:
                for mode in self.args.normalization:
                    self.exclude(m, mode, 'incomplete_tail')
                continue
            self.pair([artifact[:, start:start+width]], [neural[:, start:start+width]], scales, m)


def run_swec(exp):
    args = exp.args
    config = json.loads(args.patient_config.read_text())
    if args.data_root is not None:
        config['data_root'] = str(args.data_root)
    if args.artifact_source is not None:
        config['artifact_source'] = str(args.artifact_source)
    config['seed'] = args.seed
    if (not isinstance(config['target_fs'], int) or config['target_fs'] <= 0
            or config['artifact']['high_fs'] % config['target_fs']):
        raise ValueError('Configured target_fs must be a positive integer divisor of artifact high_fs')
    config['samples_per_patient'] = None
    # The existing class-stratified extractor performs reproducible selection
    # after recording-level grouping. Explicit finite per-class caps bound I/O.
    config['max_windows_per_class']['test'] = [(args.max_trials + 1)//2, args.max_trials//2]
    patients = config['patients'][:args.max_patients]
    exp.provenance.append(dict(source='patient_config', config=config,
                               sampling='Existing grouped test split; class-stratified seeded clip selection'))
    pbs = {}
    if args.dataset in ('all', 'swec_pbs'):
        for c in args.pbs_configurations:
            print(f'Loading PBS configuration {c}', flush=True)
            try:
                ds = load_pbs_artifact(args.root, c)
                gcd = np.gcd(int(ds.fs), config['target_fs'])
                pbs[c] = resample_poly(ds.X[0], config['target_fs']//gcd, int(ds.fs)//gcd, axis=-1)
                exp.provenance.append(dict(dataset='swec_pbs', configuration=c, **ds.metadata,
                                           source_fs=ds.fs, analysis_fs=config['target_fs']))
            except (OSError, ValueError) as exc:
                exp.skip('swec_pbs', f'configuration_{c}', exc)
    for pid in patients:
        print(f'Loading SWEC patient {pid}', flush=True)
        with tempfile.TemporaryDirectory(prefix='subspace_', dir=args.output_dir) as directory:
            try:
                patient = load_patient_data(pid, config)
                output = args.output_dir / 'sampling' / patient.patient
                groups = split_patient_data(patient, config, output)
                clean, labels = extract_windows(patient, groups['test'], 'test', config, directory)
                # Preserve exact native sample positions, labels and selection
                # provenance without retaining large intermediate signal arrays.
                for name in ('test_windows.csv', 'test_counts.json'):
                    (output/name).write_bytes((Path(directory)/name).read_bytes())
            except (OSError, ValueError) as exc:
                for dataset in ('swec_synthetic', 'swec_pbs'):
                    if args.dataset in ('all', dataset):
                        exp.skip(dataset, pid, exc)
                continue
            for trial, neural in enumerate(clean):
                base = dict(subject=patient.patient, trial=trial, fs=config['target_fs'], reference_type='ground_truth')
                if args.dataset in ('all', 'swec_synthetic'):
                    seed = derived_seed(args.seed, patient.patient, trial, 'subspace_artifact')
                    try:
                        # Fixed positive reference amplitude: overall artifact
                        # magnitude cannot change either spatial subspace.
                        _, artifact, params = generate_synthetic_artifact(neural, 1.0, config, seed)
                        exp.provenance.append(dict(dataset='swec_synthetic', subject=patient.patient,
                                                   trial=trial, native_fs=patient.fs,
                                                   amplitude_reference='fixed unit RMS, not training RMS', **params))
                        exp.windows(artifact, neural, dict(base, dataset='swec_synthetic',
                                    configuration='seeded_synthetic', unit=patient.patient))
                    except (OSError, ValueError) as exc:
                        exp.skip('swec_synthetic', f'{patient.patient}/trial{trial}', exc)
                for c, artifact in pbs.items():
                    if neural.shape[0] < artifact.shape[0]:
                        exp.skip('swec_pbs', patient.patient, 'Fewer neural channels than PBS channels')
                        break
                    length = min(neural.shape[1], artifact.shape[1])
                    exp.windows(artifact[:, :length], neural[:artifact.shape[0], :length],
                                dict(base, dataset='swec_pbs', configuration=str(c), unit=f'PBS_{c}'))
            del clean


def run_real(exp):
    args = exp.args
    names = []
    if args.dataset in ('all', 'eraasr'):
        names.append('ERAASR')
    if args.dataset in ('all', 'dictionary'):
        names.extend(f'dictionary_{c}stim' for c in args.dictionary_configurations)
    for name in names:
        print(f'Loading real recording {name}', flush=True)
        try:
            ds = load_dataset(name, root=args.root)
            if ds.baseline is None:
                raise ValueError('No explicit prestimulation neural reference')
            if ds.baseline.shape[:2] != ds.X.shape[:2]:
                raise ValueError('Rest trials/channels cannot be paired with stimulation trials')
            # This estimator needs >=2 disjoint groups of >=2 trials. The
            # SWEC clip cap is intentionally separate from this block count.
            blocks, events = extract_stimulation_locked_artifact(ds.X, ds.fs, ds.stim_rate,
                                                                 trials_per_block=args.trials_per_block)
        except (OSError, ValueError) as exc:
            exp.skip(name, name, exc)
            continue
        exp.events.extend(dict(dataset=name, **event) for event in events)
        groups = [list(range(i, min(i+args.trials_per_block, len(ds.X))))
                  for i in range(0, len(ds.X), args.trials_per_block)]
        if len(groups[-1]) == 1:
            groups[-2].extend(groups.pop())
        exp.provenance.append(dict(dataset=name, reference_type='pulse_average_estimate',
            neural_reference=ds.metadata, artifact_method='existing stimulation-locked block average',
            artifact_epoch_samples=blocks[0].shape[1], rest_samples_per_trial=ds.baseline.shape[-1],
            fs=ds.fs, groups=groups, limitation='Pulse-locked neural responses may remain; not artifact ground truth'))
        scales = exp.scales(list(ds.baseline))
        for i, (artifact, trials) in enumerate(zip(blocks, groups)):
            exp.pair([artifact], [ds.baseline[t] for t in trials], scales,
                dict(dataset=name, subject=name, configuration=name, trial=';'.join(map(str, trials)),
                     window=i, unit=name, reference_type='pulse_average_estimate', start_sample=0, fs=ds.fs))
        del ds, blocks


def paired_observations(rows):
    indexed = defaultdict(dict)
    for row in rows:
        key = tuple(row[k] for k in PAIR_KEYS)
        if row['condition'] in indexed[key]:
            raise AssertionError('Duplicate observation')
        value = row['subspace_overlap']
        assert np.isfinite(value) and -1e-10 <= value <= 1+1e-10
        indexed[key][row['condition']] = row
    assert all(set(pair) == set(CONDITIONS) for pair in indexed.values()), 'Unmatched conditions'
    return indexed


def summarize(rows):
    paired_observations(rows)
    # First average windows within subject/configuration, then average those
    # subject means within PBS configuration. Reused PBS recordings are never
    # counted once per patient as independent replicates.
    cells = defaultdict(list)
    for r in rows:
        cells[(r['dataset'], r['normalization'], r['unit'], r['subject'], r['condition'])].append(r['subspace_overlap'])
    units = defaultdict(list)
    for (dataset, mode, unit, subject, condition), values in cells.items():
        units[(dataset, mode, unit, condition)].append(float(np.mean(values)))
    unit_rows = [dict(dataset=d, normalization=m, unit=u, condition=c,
                      subspace_overlap=float(np.mean(v)), n_subjects=len(v))
                 for (d,m,u,c), v in sorted(units.items())]
    summaries, statistics = [], []
    for dataset, mode in sorted({(r['dataset'], r['normalization']) for r in rows}):
        selected = [r for r in unit_rows if r['dataset']==dataset and r['normalization']==mode]
        by_unit = defaultdict(dict)
        for r in selected:
            by_unit[r['unit']][r['condition']] = r['subspace_overlap']
        assert all(set(p)==set(CONDITIONS) for p in by_unit.values())
        a, b = np.asarray([[p['none'],p['derivative']] for p in by_unit.values()]).T
        delta = np.round(b-a, 12)
        n = len(delta)
        if n < 2:
            statistic, p = np.nan, np.nan
            status = 'insufficient_independent_recordings'
        elif np.all(delta == 0):
            statistic, p, status = 0., 1., 'all_differences_zero'
        else:
            result = wilcoxon(delta, alternative='two-sided', zero_method='wilcox', method='auto')
            statistic, p, status = float(result.statistic), float(result.pvalue), 'ok'
        statistics.append(dict(dataset=dataset, normalization=mode, n_independent=n,
            independent_unit='PBS configuration' if dataset=='swec_pbs' else ('patient' if dataset=='swec_synthetic' else 'recording'),
            median_paired_difference=float(np.median(b-a)), mean_paired_difference=float(np.mean(b-a)),
            statistic=statistic, p_value=p, status=status))
        for condition, values in zip(CONDITIONS, (a,b)):
            raw = [r for r in rows if r['dataset']==dataset and r['normalization']==mode and r['condition']==condition]
            summaries.append(dict(dataset=dataset, normalization=mode, condition=condition,
                n_samples=n, n_windows=len(raw), mean_overlap=float(values.mean()),
                std_overlap=float(values.std(ddof=1)) if n>1 else np.nan,
                median_overlap=float(np.median(values)), iqr_overlap=float(np.ptp(np.percentile(values,[25,75]))),
                window_mean_overlap=float(np.mean([r['subspace_overlap'] for r in raw]))))
    for row, adjusted in zip(statistics, _holm([r['p_value'] for r in statistics])):
        row['p_holm'] = float(adjusted)
        row['correction_family'] = 'all evaluated datasets and normalization modes'
    return unit_rows, summaries, statistics


def style():
    set_tbme_style()
    plt.rcParams.update({'font.family':'serif', 'mathtext.fontset':'stix', 'font.size':13,
        'axes.labelsize':13, 'xtick.labelsize':13, 'ytick.labelsize':13, 'grid.alpha':.18,
        'grid.linewidth':.5, 'lines.linewidth':1.4, 'pdf.fonttype':42})


def save_figure(fig, path):
    for ext in ('pdf','png'):
        fig.savefig(path.with_suffix('.'+ext), dpi=600, facecolor='white')
    plt.close(fig)


def plot_pairs(units, mode, path, real=False):
    selected = [r for r in units if r['normalization']==mode and (r['dataset'].startswith('swec_') != real)]
    datasets = sorted({r['dataset'] for r in selected},
                      key=lambda d: ({'swec_synthetic':0,'swec_pbs':1}.get(d,2),d))
    if not datasets:
        return
    style()
    # Keep each exploratory recording on its own one-column panel.
    panels = [[d] for d in datasets] if real else [datasets]
    for panel in panels:
        fig, ax = plt.subplots(figsize=(3.5,3.0))
        for i,dataset in enumerate(panel):
            pairs = defaultdict(dict)
            for r in selected:
                if r['dataset']==dataset:
                    pairs[r['unit']][r['condition']] = r['subspace_overlap']
            values = np.asarray([[p[c] for c in CONDITIONS] for p in pairs.values()])
            x = np.array([i-.15,i+.15])
            for pair in values:
                ax.plot(x,pair,color='.65',lw=.5,alpha=.55,zorder=1)
            for j,(label,color) in enumerate(zip(('No diff.','Differential'),(TBME_COLORS['raw'],TBME_COLORS['after_subspace']))):
                ax.scatter(np.full(len(values),x[j]),values[:,j],s=10,facecolors='none',edgecolors=color,linewidths=.6,zorder=2)
                ax.errorbar(x[j],values[:,j].mean(),yerr=values[:,j].std(ddof=1) if len(values)>1 else 0,
                    color=color,marker='o',markersize=4,capsize=3,lw=1.4,zorder=3,label=label if i==0 else None)
        labels = [{'swec_synthetic':'SWEC +\nSynthetic','swec_pbs':'SWEC +\nPBS'}.get(d,d.replace('_',' ')) for d in panel]
        ax.set(xticks=range(len(panel)),xticklabels=labels,ylim=(0,1),xlim=(-.5,len(panel)-.5),
               ylabel='Artifact–neural\nsubspace overlap')
        ax.set_axisbelow(True)
        ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=2,frameon=False,fontsize=9.5,
                  handlelength=1.0,handletextpad=.4,columnspacing=.8,borderaxespad=0)
        fig.subplots_adjust(left=.24,right=.97,top=.85,bottom=.23)
        destination = path.with_name(path.name+'_'+panel[0]) if real else path
        save_figure(fig,destination)


def plot_diagnostic(exp):
    spectrum_rows = []
    for mode,(meta,spectra) in exp.diagnostic.items():
        style()
        fig,ax = plt.subplots(figsize=(3.5,3.0))
        for condition,ls in (('none','-'),('derivative','--')):
            for i,(label,color) in enumerate((('Artifact',TBME_COLORS['raw']),('Neural',TBME_COLORS['after_subspace']))):
                s = spectra[condition][i]
                spectrum_rows.extend(dict(normalization=mode, condition=condition,
                    component=label.lower(), rank=k+1, relative_singular_value=float(value/s[0]))
                    for k,value in enumerate(s))
                ax.semilogy(np.arange(1,len(s)+1),np.maximum(s/s[0],1e-16),ls=ls,color=color,
                    label=label+(' diff.' if condition=='derivative' else ''))
        ax.set(xlabel='Spatial rank',ylabel=r'$\sigma_i/\sigma_1$',ylim=(1e-8,1.2))
        ax.legend(frameon=False,fontsize=9.5)
        fig.subplots_adjust(left=.24,right=.95,top=.95,bottom=.20)
        save_figure(fig,exp.args.output_dir/f'singular_value_diagnostic_{mode}')
        write_json(exp.args.output_dir/f'singular_value_diagnostic_{mode}.json',meta)
    if spectrum_rows:
        write_csv(exp.args.output_dir/'diagnostic_spectra.csv',spectrum_rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset',choices=['all','swec_synthetic','swec_pbs','eraasr','dictionary'],default='all')
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--patient-config',type=Path,default=ROOT/'experiments/seizure_preservation/config.json')
    parser.add_argument('--data-root',type=Path)
    parser.add_argument('--artifact-source',type=Path)
    parser.add_argument('--energy-threshold',type=float,default=.95)
    parser.add_argument('--window-ms',type=float,default=100.)
    parser.add_argument('--max-patients',type=int)
    parser.add_argument('--max-trials',type=int,default=6,help='Maximum SWEC clips per patient (default: 6)')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--normalization',nargs='+',choices=MODES,default=list(MODES))
    parser.add_argument('--pbs-configurations',nargs='+',type=int,default=list(range(14)))
    parser.add_argument('--dictionary-configurations',nargs='+',choices=['2','4','5','14'],default=['2','4','5','14'])
    parser.add_argument('--trials-per-block',type=int,default=5,help='Real-recording pulse-average group size')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'figures/derivative')
    args = parser.parse_args()
    if not np.isfinite(args.energy_threshold) or not 0<args.energy_threshold<1:
        parser.error('--energy-threshold must be in (0,1)')
    if not np.isfinite(args.window_ms) or args.window_ms<=0:
        parser.error('--window-ms must be finite and positive')
    if args.max_trials<1 or (args.max_patients is not None and args.max_patients<1) or args.trials_per_block<2:
        parser.error('Require positive patient/trial limits and trials-per-block >= 2')
    for key in ('normalization','pbs_configurations','dictionary_configurations'):
        if len(getattr(args,key)) != len(set(getattr(args,key))):
            parser.error(f'Duplicate values in --{key.replace("_","-")}')
    return args


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    exp = Experiment(args)
    with threadpool_limits(limits=1):
        if args.dataset in ('all','swec_synthetic','swec_pbs'):
            config = json.loads(args.patient_config.read_text())
            if round(args.window_ms*config['target_fs']/1000)<4:
                raise ValueError('Window needs at least four samples at the configured analysis rate')
            run_swec(exp)
        if args.dataset in ('all','eraasr','dictionary'):
            run_real(exp)
        units,summaries,statistics = summarize(exp.rows)
    write_csv(args.output_dir/'window_overlaps.csv',exp.rows,ROW_FIELDS)
    write_csv(args.output_dir/'excluded_windows.csv',exp.exclusions,(*PAIR_KEYS,'reason'))
    write_csv(args.output_dir/'skipped_datasets.csv',exp.skipped,('dataset','subject','reason'))
    if units:
        write_csv(args.output_dir/'independent_unit_overlaps.csv',units)
        write_csv(args.output_dir/'overlap_summary.csv',summaries)
        write_csv(args.output_dir/'paired_statistics.csv',statistics)
    if exp.events:
        write_csv(args.output_dir/'real_recording_events.csv',exp.events)
    for mode in args.normalization:
        suffix = '' if mode=='none' else '_global_per_channel'
        plot_pairs(units,mode,args.output_dir/f'artifact_neural_subspace_overlap{suffix}')
        plot_pairs(units,mode,args.output_dir/f'exploratory_subspace_overlap{suffix}',real=True)
    plot_diagnostic(exp)
    write_json(args.output_dir/'analysis_metadata.json',dict(
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        provenance=exp.provenance, notes=[
            'Both modes center each segment; none is unscaled, not uncentered.',
            'global_per_channel uses the SAME neural-reference SD for artifact and neural, fixed before differentiation.',
            'Undifferentiated segments retain T samples; differences contain T-1. Both originate from exactly the same window.',
            'Overlap always divides by min(artifact_rank, neural_rank); the numerator is also saved.',
            'Synthetic units are patients; PBS units are configurations after equal-weight subject averaging.',
            'PBS inference is conditional on the sampled neural cohort; configuration independence is assumed, not verified biologically.',
            'Real-data block estimates are descriptive; a single recording cannot support an independent-recording Wilcoxon test.',
            'Main figure uses centered unscaled signals; normalized sensitivity and exploratory real-data figures are separate.',
            'Negative paired difference means lower overlap after differentiation. No parameter selection on outcomes.',
            'Singular spectra are relative within each matrix; compute_spatial_svd rescales inputs for numerical stability.',
        ]))
    print('\nSuccessfully evaluated:',sorted({r['dataset'] for r in exp.rows}))
    print('Skipped:',exp.skipped)
    print('\nIndependent-unit summary:')
    for r in summaries:
        print(r)
    print('\nPaired two-sided Wilcoxon (derivative minus none):')
    for r in statistics:
        print(r)
    for dataset,mode in sorted({(r['dataset'],r['normalization']) for r in exp.rows}):
        rows = [r for r in exp.rows if r['dataset']==dataset and r['normalization']==mode]
        means = {c:float(np.mean([r['subspace_overlap'] for r in rows if r['condition']==c])) for c in CONDITIONS}
        print(f'{dataset}/{mode}: valid paired windows={len(rows)//2}; window means={means}; change={means["derivative"]-means["none"]:.6g}')
        for c in CONDITIONS:
            print(c,'artifact ranks',dict(sorted(Counter(r['artifact_rank'] for r in rows if r['condition']==c).items())),
                  'neural ranks',dict(sorted(Counter(r['neural_rank'] for r in rows if r['condition']==c).items())))
    print('Exclusions:',dict(Counter(r['reason'] for r in exp.exclusions)))
    print('\nFiles created:')
    for path in sorted(args.output_dir.rglob('*')):
        if path.is_file():
            print(path)
    if not exp.rows:
        raise SystemExit('No valid paired observations; see skipped_datasets.csv and excluded_windows.csv')


if __name__=='__main__':
    main()
