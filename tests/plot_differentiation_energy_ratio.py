"""Artifact-to-neural energy ratio before and after first-order differences.

Run: python tests/plot_differentiation_energy_ratio.py --dataset all

R0 = sum(A**2) / sum(N**2); R1 = sum(diff(A)**2) / sum(diff(N)**2).
Gain = R1/R0; gain_db = 10*log10(R1/R0). Gain > 1 emphasizes artifact.
No centering, filtering or channel normalization is added by default. Use
--demean for a separately labeled sensitivity analysis of fluctuation energy.

Only SWEC + synthetic and SWEC + PBS have separate artifact/neural components.
Existing patient loaders, grouped splits, clip selection, generator and PBS
reader are reused. Synthetic amplitudes reference separate training clips;
PBS uses load_swec_pbs's scale convention. Neither is rescaled between paired
conditions. Resampling follows the patient config (default analysis: 1024 Hz).

Window dB ratios are averaged within patient/configuration. Synthetic inference
uses patients; PBS inference averages patient means within each configuration.
Thus independent-unit ratios are geometric means of window ratios. Tables
give median [Q1,Q3] across independent units, with paired gains summarized
separately. Wilcoxon tests use unit dB changes, not dependent time windows.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import inspect
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

from algorithms import derivative_time
from data_loading import ROOT, load_swec_pbs
from plotting import TBME_COLORS, set_tbme_style
from experiments.seizure_preservation.common import derived_seed, write_csv, write_json
from experiments.seizure_preservation.data import load_patient_data, split_patient_data, extract_windows
from experiments.seizure_preservation.cancellation import generate_synthetic_artifact
from experiments.seizure_preservation.reporting import _holm
from plot_stable_low_dim_artifact_subspace import load_pbs_artifact

CONDITIONS = ('none', 'derivative')
KEYS = ('dataset', 'subject', 'configuration', 'trial', 'window')
ENERGY_FIELDS = ('artifact_energy', 'neural_energy', 'energy_ratio', 'energy_ratio_db', 'n_samples')
ROW_FIELDS = (*KEYS, 'unit', 'start_sample', 'fs', 'channels', 'demean', 'condition', *ENERGY_FIELDS)
PAIR_FIELDS = (*KEYS, 'unit', 'ratio_none', 'ratio_derivative', 'gain', 'gain_db')
LABELS = {'swec_synthetic': 'SWEC + Synthetic', 'swec_pbs': 'SWEC + PBS'}


def energy_comparison(artifact, neural, demean=False):
    """Return a matched pair and gain; never normalize away relative amplitude.

    Energies sum over all channels and samples. Each condition uses the same
    sample count for A and N, so ratios of sums equal ratios of mean powers.
    Differences shorten BOTH components by one sample and never cross windows.
    Zero-energy components exclude the entire pair: log ratios/gains would be
    undefined. No epsilon is added to turn an undefined ratio into a result.
    """
    a, n = (np.asarray(x, dtype=np.float64) for x in (artifact, neural))
    if a.ndim != 2 or a.shape != n.shape or a.shape[0] < 1 or a.shape[-1] < 3:
        raise ValueError('insufficient_samples_or_mismatched_shapes')
    if not np.isfinite(a).all() or not np.isfinite(n).all():
        raise ValueError('nonfinite_samples')
    results = []
    for condition in CONDITIONS:
        values = [a, n] if condition == 'none' else [derivative_time(a), derivative_time(n)]
        if demean:
            values = [x - x.mean(axis=-1, keepdims=True) for x in values]
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            energies = [float(np.sum(x*x, dtype=np.float64)) for x in values]
        for component, energy in zip(('artifact', 'neural'), energies):
            if not np.isfinite(energy) or energy <= np.finfo(float).tiny:
                raise ValueError(f'{condition}_{component}_zero_or_invalid_energy')
        ea, en = energies
        db = float(10 * (np.log10(ea) - np.log10(en)))
        with np.errstate(over='ignore', under='ignore'):
            ratio = float(np.power(10., db/10))
        if not np.isfinite(ratio) or ratio <= 0:
            raise ValueError('energy_ratio_outside_float_range')
        results.append(dict(condition=condition, artifact_energy=ea, neural_energy=en,
                            energy_ratio=ratio, energy_ratio_db=db, n_samples=values[0].shape[-1]))
    gain_db = results[1]['energy_ratio_db'] - results[0]['energy_ratio_db']
    with np.errstate(over='ignore', under='ignore'):
        gain = float(np.power(10., gain_db/10))
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError('gain_outside_float_range')
    return results, dict(ratio_none=results[0]['energy_ratio'], ratio_derivative=results[1]['energy_ratio'],
                         gain=gain, gain_db=float(gain_db))


class Experiment:
    def __init__(self, args):
        self.args = args
        self.rows, self.pairs, self.exclusions, self.skipped, self.provenance = [], [], [], [], []

    def skip(self, dataset, subject, reason):
        self.skipped.append(dict(dataset=dataset, subject=str(subject), reason=str(reason)))
        print('SKIP:', self.skipped[-1], flush=True)

    def windows(self, artifact, neural, meta):
        if artifact.shape != neural.shape:
            raise ValueError('Ground-truth components must have identical shapes')
        width = round(self.args.window_ms * meta['fs'] / 1000)
        for window, start in enumerate(range(0, neural.shape[-1], width)):
            identity = {**{k: meta[k] for k in KEYS if k != 'window'}, 'window': window}
            try:
                if start+width > neural.shape[-1]:
                    raise ValueError('incomplete_tail')
                rows, gain = energy_comparison(artifact[:,start:start+width], neural[:,start:start+width], self.args.demean)
            except ValueError as exc:
                self.exclusions.append(dict(**identity, reason=str(exc)))
                continue
            self.rows.extend(dict(**meta, window=window, start_sample=start, channels=len(neural),
                                  demean=self.args.demean, **r) for r in rows)
            self.pairs.append(dict(**identity, unit=meta['unit'], **gain))


def read_clips(patient, groups, split, config, temporary, destination):
    """Call the existing extractor and retain its exact selection manifest."""
    clean, _ = extract_windows(patient, groups[split], split, config, temporary)
    for name in (f'{split}_windows.csv', f'{split}_counts.json'):
        (destination/name).write_bytes((Path(temporary)/name).read_bytes())
    return clean


def run_datasets(exp, config):
    args = exp.args
    pbs = {}
    if args.dataset in ('all','swec_pbs'):
        for c in args.pbs_configurations:
            print(f'Loading PBS configuration {c}', flush=True)
            try:
                ds = load_pbs_artifact(args.root, c)
                divisor = np.gcd(int(ds.fs), config['target_fs'])
                pbs[c] = args.pbs_scale * resample_poly(ds.X[0], config['target_fs']//divisor,
                                                       int(ds.fs)//divisor, axis=-1)
                exp.provenance.append(dict(dataset='swec_pbs', configuration=c, scale=args.pbs_scale,
                    source_fs=ds.fs, analysis_fs=config['target_fs'], **ds.metadata))
            except (OSError, ValueError) as exc:
                exp.skip('swec_pbs', f'configuration_{c}', exc)
    for pid in config['patients'][:args.max_patients]:
        print(f'Loading SWEC patient {pid}', flush=True)
        with tempfile.TemporaryDirectory(prefix='energy_', dir=args.output_dir) as tmp:
            try:
                patient = load_patient_data(pid, config)
                destination = args.output_dir/'sampling'/patient.patient
                groups = split_patient_data(patient, config, destination)
                clean = read_clips(patient, groups, 'test', config, tmp, destination)
            except (OSError, ValueError) as exc:
                for dataset in LABELS:
                    if args.dataset in ('all',dataset):
                        exp.skip(dataset, pid, exc)
                continue
            train_rms = None
            if args.dataset in ('all','swec_synthetic'):
                try:
                    calibration = read_clips(patient, groups, 'train', config, tmp, destination)
                    train_rms = float(np.sqrt(np.mean(np.square(calibration, dtype=np.float64))))
                    if not np.isfinite(train_rms) or train_rms <= 0:
                        raise ValueError('Invalid training-reference RMS')
                    exp.provenance.append(dict(dataset='swec_synthetic', subject=patient.patient,
                        calibration='disjoint training clips', train_rms=train_rms,
                        calibration_clips=len(calibration), native_fs=patient.fs, analysis_fs=config['target_fs']))
                    del calibration
                except (OSError, ValueError) as exc:
                    train_rms = None
                    exp.skip('swec_synthetic', patient.patient, exc)
            for trial, neural in enumerate(clean):
                base = dict(subject=patient.patient, trial=trial, fs=config['target_fs'])
                if train_rms is not None:
                    # Same seed namespace as the preceding subspace experiment;
                    # amplitude now uses the existing generator's training-RMS
                    # convention because absolute energy ratios depend on it.
                    seed = derived_seed(args.seed, patient.patient, trial, 'subspace_artifact')
                    try:
                        _, artifact, params = generate_synthetic_artifact(neural, train_rms, config, seed)
                        exp.provenance.append(dict(dataset='swec_synthetic', subject=patient.patient, trial=trial, **params))
                        exp.windows(artifact, neural, dict(base, dataset='swec_synthetic',
                                    configuration='seeded_synthetic', unit=patient.patient))
                    except (OSError, ValueError) as exc:
                        exp.skip('swec_synthetic', f'{patient.patient}/trial{trial}', exc)
                for c, artifact in pbs.items():
                    if neural.shape[0] < artifact.shape[0]:
                        exp.skip('swec_pbs', patient.patient, 'Insufficient neural channels for PBS')
                        break
                    length = min(neural.shape[-1], artifact.shape[-1])
                    exp.windows(artifact[:,:length], neural[:len(artifact),:length],
                                dict(base, dataset='swec_pbs', configuration=str(c), unit=f'PBS_{c}'))
            del clean


def summarize(rows):
    """Equal-weight subjects; reused PBS recordings count once per configuration."""
    paired = defaultdict(dict)
    cells = defaultdict(list)
    for r in rows:
        key = tuple(r[k] for k in KEYS)
        if r['condition'] in paired[key]:
            raise ValueError('Duplicate observation')
        paired[key][r['condition']] = r
        if not np.isfinite(r['energy_ratio_db']):
            raise ValueError('Nonfinite ratio')
        cells[(r['dataset'],r['unit'],r['subject'],r['condition'])].append(r['energy_ratio_db'])
    if any(set(p) != set(CONDITIONS) for p in paired.values()):
        raise ValueError('Unmatched conditions')
    values = defaultdict(dict)
    for (dataset, unit, subject, condition), dbs in cells.items():
        values[(dataset,unit,condition)][subject] = float(np.mean(dbs))
    units = []
    for dataset, unit in sorted({(d,u) for d,u,c in values}):
        a, b = (values[(dataset,unit,c)] for c in CONDITIONS)
        if set(a) != set(b):
            raise ValueError('Unmatched subjects')
        before, after = float(np.mean(list(a.values()))), float(np.mean(list(b.values())))
        units.append(dict(dataset=dataset,unit=unit,n_subjects=len(a),ratio_none=10**(before/10),
            ratio_derivative=10**(after/10),gain=10**((after-before)/10),
            ratio_none_db=before,ratio_derivative_db=after,gain_db=after-before))
    summaries = []
    for dataset in sorted({u['dataset'] for u in units}):
        selected = [u for u in units if u['dataset']==dataset]
        delta = np.round([u['gain_db'] for u in selected],12)
        if len(delta)<2:
            statistic,p,status = np.nan,np.nan,'insufficient_independent_units'
        elif np.all(delta==0):
            statistic,p,status = 0.,1.,'all_differences_zero'
        else:
            result = wilcoxon(delta,alternative='two-sided',zero_method='wilcox',method='auto')
            statistic,p,status = float(result.statistic),float(result.pvalue),'ok'
        row = dict(dataset=dataset,n_independent=len(selected),
            independent_unit='patient' if dataset=='swec_synthetic' else 'PBS configuration',
            n_paired_windows=sum(key[0]==dataset for key in paired),
            n_subjects=len({r['subject'] for r in rows if r['dataset']==dataset}),
            mean_gain_db=float(np.mean(delta)),statistic=statistic,p_value=p,status=status)
        for field in ('ratio_none','ratio_derivative','gain','gain_db'):
            q1,median,q3 = np.percentile([u[field] for u in selected],[25,50,75])
            row.update({field+'_median':float(median),field+'_q1':float(q1),field+'_q3':float(q3)})
        summaries.append(row)
    for row,p in zip(summaries,_holm([r['p_value'] for r in summaries])):
        row.update(p_holm=float(p),correction_family='all evaluated ground-truth datasets')
    return units,summaries


def format_interval(row, key):
    return f'{row[key+"_median"]:.3g} [{row[key+"_q1"]:.3g}, {row[key+"_q3"]:.3g}]'


def write_tables(summary, output):
    header = ['Dataset','Independent units','No differentiation','First-order differentiation','Paired gain','Holm p']
    lines = ['| '+' | '.join(header)+' |','|---|---:|---:|---:|---:|---:|']
    latex = [r'\begin{table}[t]',r'\centering',r'\caption{Artifact-to-neural energy ratios before and after temporal differentiation. '
             r'Values are median [Q1, Q3] across independent units of geometric-mean window ratios. '
             r'Gain is summarized from paired unit ratios; gain above one indicates artifact emphasis.}',
             r'\label{tab:differentiation_energy}',r'\resizebox{\columnwidth}{!}{%',
             r'\begin{tabular}{lrrrrr}',r'\toprule',r'Dataset & $n$ & No diff. & Differential & Gain & Holm $p$ \\',r'\midrule']
    for row in summary:
        p = f'{row["p_holm"]:.3g}' if np.isfinite(row['p_holm']) else 'NA'
        cells = [LABELS[row['dataset']],str(row['n_independent']),
                 *(format_interval(row,key) for key in ('ratio_none','ratio_derivative','gain')),p]
        lines.append('| '+' | '.join(cells)+' |')
        latex.append(' & '.join(cells)+r' \\')
    latex.extend([r'\bottomrule',r'\end{tabular}}',r'\end{table}'])
    note = ('\n\nRatios are dimensionless. Each unit averages window ratios in dB (geometric mean in linear units). '
            'Synthetic units are patients; PBS units are configurations after equal-weight patient averaging. '
            'Gain is computed within each pair before aggregation. Tests are two-sided paired Wilcoxon on unit dB changes.\n')
    (output/'energy_ratio_table.md').write_text('\n'.join(lines)+note)
    (output/'energy_ratio_table.tex').write_text('\n'.join(latex)+'\n')
    print('\n'+'\n'.join(lines))


def plot_results(units, output):
    set_tbme_style()
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':13,
        'axes.labelsize':13,'xtick.labelsize':13,'ytick.labelsize':13,'grid.alpha':.18,
        'grid.linewidth':.5,'lines.linewidth':1.4,'pdf.fonttype':42})
    datasets = [d for d in LABELS if any(r['dataset']==d for r in units)]
    fig,ax = plt.subplots(figsize=(3.5,3.0))
    for i,dataset in enumerate(datasets):
        selected = [r for r in units if r['dataset']==dataset]
        v = np.array([[r['ratio_none_db'],r['ratio_derivative_db']] for r in selected])
        x = np.array([i-.15,i+.15])
        for pair in v:
            ax.plot(x,pair,color='.65',lw=.5,alpha=.55,zorder=1)
        for j,(label,color) in enumerate(zip(('No diff.','Differential'),(TBME_COLORS['raw'],TBME_COLORS['after_subspace']))):
            q1,median,q3 = np.percentile(v[:,j],[25,50,75])
            ax.scatter(np.full(len(v),x[j]),v[:,j],s=10,facecolors='none',edgecolors=color,linewidths=.6)
            ax.errorbar(x[j],median,yerr=[[median-q1],[q3-median]],color=color,marker='o',
                        markersize=4,capsize=3,lw=1.4,label=label if i==0 else None)
    ax.set(xticks=range(len(datasets)),xticklabels=[LABELS[d].replace(' + ',' +\n') for d in datasets],
           ylabel='Artifact-to-neural\nenergy ratio (dB)',xlim=(-.5,len(datasets)-.5))
    ax.set_axisbelow(True)
    ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=2,frameon=False,fontsize=9.5,
              handlelength=1.,handletextpad=.4,columnspacing=.8,borderaxespad=0)
    fig.subplots_adjust(left=.25,right=.97,top=.85,bottom=.23)
    # Negative/wide dB ticks need more space than the stable-subspace plot's
    # [0,1] ticks. Keep the physical column size while fitting the full label.
    fig.canvas.draw()
    label_left = ax.yaxis.label.get_window_extent(fig.canvas.get_renderer()).x0
    if label_left < 8:
        fig.subplots_adjust(left=.25+(8-label_left)/fig.bbox.width)
    for ext in ('pdf','png'):
        fig.savefig(output/f'artifact_neural_energy_ratio.{ext}',dpi=600,facecolor='white')
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset',choices=['all',*LABELS],default='all')
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--patient-config',type=Path,default=ROOT/'experiments/seizure_preservation/config.json')
    p.add_argument('--data-root',type=Path)
    p.add_argument('--artifact-source',type=Path)
    p.add_argument('--max-patients',type=int)
    p.add_argument('--max-trials',type=int,default=6,help='Maximum test clips per patient')
    p.add_argument('--calibration-trials',type=int,default=6,help='Maximum disjoint training clips for synthetic amplitude RMS')
    p.add_argument('--window-ms',type=float,default=100.)
    p.add_argument('--pbs-configurations',nargs='+',type=int,default=list(range(14)))
    p.add_argument('--pbs-scale',type=float,default=inspect.signature(load_swec_pbs).parameters['artifact_scale'].default)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--demean',action='store_true',help='Remove each channel mean in each condition; default is raw energy')
    p.add_argument('--output-dir',type=Path,default=ROOT/'figures/derivative/energy_ratio')
    args = p.parse_args()
    if args.max_trials<1 or args.calibration_trials<1 or (args.max_patients is not None and args.max_patients<1):
        p.error('Patient/trial limits must be positive')
    if not np.isfinite(args.window_ms) or args.window_ms<=0 or not np.isfinite(args.pbs_scale) or args.pbs_scale<=0:
        p.error('Window duration and PBS scale must be finite and positive')
    if len(set(args.pbs_configurations)) != len(args.pbs_configurations):
        p.error('Duplicate PBS configurations')
    return args


def main():
    args = parse_args()
    config = json.loads(args.patient_config.read_text())
    for key in ('data_root','artifact_source'):
        if getattr(args,key) is not None:
            config[key] = str(getattr(args,key))
    fs = config['target_fs']
    if not isinstance(fs,int) or fs<=0 or config['artifact']['high_fs'] % fs or round(args.window_ms*fs/1000)<3:
        raise ValueError('Require >=3 samples per window and a positive target_fs dividing artifact high_fs')
    config['seed'] = args.seed
    config['samples_per_patient'] = None
    for split,limit in [('test',args.max_trials),('train',args.calibration_trials)]:
        config['max_windows_per_class'][split] = [(limit+1)//2,limit//2]
    args.output_dir.mkdir(parents=True,exist_ok=True)
    exp = Experiment(args)
    if args.dataset=='all':
        for dataset in ('ERAASR','Dictionary Learning'):
            exp.skip(dataset,'all','No separately available artifact ground truth; pulse estimates and separate rest cannot provide this ground-truth energy ratio')
    run_datasets(exp,config)
    units,summary = summarize(exp.rows)
    write_csv(args.output_dir/'window_energy_ratios.csv',exp.rows,ROW_FIELDS)
    write_csv(args.output_dir/'paired_energy_gains.csv',exp.pairs,PAIR_FIELDS)
    write_csv(args.output_dir/'excluded_windows.csv',exp.exclusions,(*KEYS,'reason'))
    write_csv(args.output_dir/'skipped_datasets.csv',exp.skipped,('dataset','subject','reason'))
    write_json(args.output_dir/'analysis_metadata.json',dict(
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        config=config,provenance=exp.provenance,notes=[
            'R=sum(artifact**2)/sum(neural**2); gain=R_derivative/R_none; dB gain=10*log10(gain).',
            'Raw energy by default, including DC. --demean is an optional fluctuation-energy sensitivity analysis.',
            'No per-component normalization or condition-specific amplitude scaling.',
            'Undifferentiated signals have T samples; both differentiated components have T-1 samples.',
            'Fixed windows are measured in both conditions; a failed condition excludes the whole pair.',
            'Neural data are resampled by the existing extractor. Upsampling does not add neural bandwidth.',
            'Synthetic artifacts are generated once per test clip with training-reference amplitude; first eight channels are used for PBS.',
            'PBS scale follows the existing mixture loader; absolute ratios depend on this scale and neural recording units.',
            'PBS inference assumes independent configurations and is conditional on the sampled neural cohort.',
            'Unit means are in dB; tables summarize the corresponding geometric-mean linear ratios.',
            'Increased energy ratio measures artifact emphasis, not spatial orthogonality or verified artifact-recovery accuracy.',
        ]))
    if not units:
        raise SystemExit('No valid comparisons; see skipped_datasets.csv and excluded_windows.csv')
    write_csv(args.output_dir/'independent_unit_energy_ratios.csv',units)
    write_csv(args.output_dir/'energy_ratio_summary.csv',summary)
    write_tables(summary,args.output_dir)
    plot_results(units,args.output_dir)
    print('\nDatasets evaluated:',[r['dataset'] for r in summary])
    for r in summary:
        print(f"{r['dataset']}: {r['n_paired_windows']} paired windows; {r['n_independent']} {r['independent_unit']} units; "
              f"median gain {r['gain_median']:.6g}x ({r['gain_db_median']:.6g} dB); "
              f"Wilcoxon p={r['p_value']:.6g}, Holm p={r['p_holm']:.6g}")
    print('Excluded windows:',dict(Counter(r['reason'] for r in exp.exclusions)))
    print('Files created in',args.output_dir)
    for path in sorted(args.output_dir.glob('*')):
        if path.is_file():
            print(path.name)


if __name__=='__main__':
    main()
