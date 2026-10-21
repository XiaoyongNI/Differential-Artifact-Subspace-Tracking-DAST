"""Matched online PASTd u1 identification with and without differentiation.

python tests/plot_pastd_u1_convergence.py --dataset all

Reuses the energy experiment's complete patient/PBS sampling and amplitude
protocol. Each continuous clip is tracked once; there are NO 100-ms resets.
Both trackers start identically and update on original sample indices 1..T-1:
the raw arm sees X[:,1:], the differential arm sees derivative_time(X).
Sample zero in each saved curve is initialization, before any tracker update.

The PRIMARY reference for BOTH arms is u1 of the original artifact A[:,1:].
Similarity is absolute cosine (sign invariant). Full-clip SVD is used ONLY for
evaluation, never to initialize or update the online trackers. A secondary
condition-matched-reference trace and artifact reference rotation are saved
to distinguish a changed derivative target from slow online identification.

Plots show independent-unit mean +/- SD. Synthetic units are patients; PBS
units are configurations after equal-weight averaging over patients. Early
mean similarity (normalized area under the curve) and restricted convergence
delay are compared using paired unit statistics. Threshold failures are
explicitly censored, never reported as successful convergence at the horizon.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import inspect
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon
from threadpoolctl import threadpool_limits

from algorithms import PASTd, derivative_time
from data_loading import ROOT, load_swec_pbs
from plotting import TBME_COLORS, set_tbme_style
from experiments.seizure_preservation.common import derived_seed, write_csv, write_json
from experiments.seizure_preservation.reporting import _holm
from plot_differentiation_energy_ratio import run_datasets, LABELS, CONDITIONS
from plot_stable_low_dim_artifact_subspace import compute_spatial_svd, dominant_vector_stability

METRICS = ('early_mean_similarity', 'restricted_convergence_ms', 'converged_by_horizon', 'horizon_similarity')


def artifact_direction(artifact, min_gap=1e-6):
    """Uncentered spatial SVD, consistent with PASTd's raw second moment."""
    u, s = compute_spatial_svd(artifact)
    gap = float(1 - (s[1]/s[0])**2) if len(s)>1 else 1.
    if gap < min_gap:
        raise ValueError('Artifact leading singular direction is not identifiable (spectral tie)')
    return u[:,0], dict(leading_energy_fraction=float(s[0]**2/np.sum(s*s)), relative_eigenvalue_gap=gap)


def track_pair(artifact, neural, beta=.999, initial_power=1., initialization='canonical', seed=42):
    """Online updates access contaminated samples only; truth is evaluation-only."""
    a, n = (np.asarray(x,dtype=np.float64) for x in (artifact,neural))
    if a.ndim!=2 or a.shape!=n.shape or a.shape[0]<2 or a.shape[-1]<3:
        raise ValueError('Need matched channels x samples components with >=2 channels and >=3 samples')
    if not np.isfinite(a).all() or not np.isfinite(n).all():
        raise ValueError('Nonfinite component samples')
    if not np.isfinite(beta) or not 0<beta<=1 or not np.isfinite(initial_power) or initial_power<=0:
        raise ValueError('Require 0 < beta <= 1 and positive initial power')
    reference, info = artifact_direction(a[:,1:])
    try:
        differential_reference, differential_info = artifact_direction(derivative_time(a))
        reference_cosine = float(np.clip(abs(reference @ differential_reference),0,1))
        reference_status = 'ok'
    except ValueError as exc:
        # Constant/tied differential artifact has no unique u1, but its primary
        # comparison to the shared original-artifact direction remains valid.
        differential_reference = None
        differential_info = dict(leading_energy_fraction=np.nan,relative_eigenvalue_gap=np.nan)
        reference_cosine, reference_status = np.nan,str(exc)
    if initialization=='canonical':
        initial = np.eye(len(a),1)[:,0]
    elif initialization=='random':
        initial = np.random.default_rng(seed).normal(size=len(a))
        initial /= np.linalg.norm(initial)
    else:
        raise ValueError('Unknown initialization')
    mixture = a+n
    if not np.isfinite(mixture).all():
        raise ValueError('Nonfinite contaminated signal')
    inputs = (mixture[:,1:],derivative_time(mixture))
    vectors = np.empty((2,a.shape[-1],a.shape[0]),dtype=np.float64)
    primary = np.empty((2,a.shape[-1]),dtype=np.float64)
    matched = np.empty_like(primary)
    for condition,x in enumerate(inputs):
        tracker = PASTd(n_channels=len(a),rank=1,beta=beta)
        tracker.U[:,0] = initial
        tracker.d[:] = initial_power
        vectors[condition,0] = initial
        for sample in range(x.shape[1]):
            tracker.update(x[:,sample],track_delta=False)
            vectors[condition,sample+1] = tracker.get_components()[:,0]
        norms = np.linalg.norm(vectors[condition],axis=1)
        if not np.isfinite(vectors[condition]).all() or np.any(norms<=np.finfo(float).tiny):
            raise ValueError('PASTd produced an invalid basis')
        vectors[condition] /= norms[:,None]
        bases = [v[:,None] for v in vectors[condition]]
        primary[condition] = dominant_vector_stability(reference[:,None],bases)[0]
        target = reference if condition==0 else differential_reference
        matched[condition] = (dominant_vector_stability(target[:,None],bases)[0]
                              if target is not None else np.nan)
    assert np.array_equal(vectors[0,0],vectors[1,0])
    assert np.isfinite(primary).all() and np.all((primary>=0)&(primary<=1))
    return dict(primary=primary,matched=matched,vectors=vectors,reference=reference,
                differential_reference=differential_reference,reference_cosine=reference_cosine,
                reference_status=reference_status,reference_info=info,differential_reference_info=differential_info)


def first_sustained_crossing(similarity, threshold, hold_samples):
    """Return (run start, confirmation index); None denotes right censoring."""
    values = np.asarray(similarity,dtype=float)
    if values.ndim!=1 or not np.isfinite(values).all() or not 0<threshold<=1 or hold_samples<1:
        raise ValueError('Invalid convergence criterion')
    if len(values)<hold_samples:
        return None
    counts = np.convolve((values>=threshold).astype(int),np.ones(hold_samples,dtype=int),mode='valid')
    starts = np.flatnonzero(counts==hold_samples)
    return (int(starts[0]),int(starts[0]+hold_samples-1)) if len(starts) else None


def trace_metrics(similarity, fs, evaluation_ms, threshold, hold_ms):
    """Fixed early horizon avoids unequal follow-up across recordings.

    Restricted delay caps time at the horizon even for nonconverged clips;
    an accompanying event flag preserves their censoring status. It is NOT
    imputed successful convergence. Full-recording event times are also saved.
    """
    horizon = int(round(evaluation_ms*fs/1000))
    if horizon<1 or horizon>=len(similarity):
        raise ValueError('Clip shorter than the requested evaluation horizon')
    hold = int(np.ceil(hold_ms*fs/1000))+1
    crossing = first_sustained_crossing(similarity,threshold,hold)
    event = crossing is not None and crossing[1]<=horizon
    early = similarity[:horizon+1]
    return dict(initial_similarity=float(similarity[0]), final_similarity=float(similarity[-1]),
        early_mean_similarity=float(np.sum((early[:-1]+early[1:])*.5)/horizon),
        horizon_similarity=float(similarity[horizon]),evaluation_ms=horizon/fs*1000,
        restricted_convergence_ms=(crossing[1] if event else horizon)/fs*1000,
        converged_by_horizon=bool(event),converged_full_clip=crossing is not None,
        convergence_start_sample=crossing[0] if crossing else -1,
        convergence_confirmed_sample=crossing[1] if crossing else -1,
        convergence_confirmed_ms=crossing[1]/fs*1000 if crossing else np.nan,
        followup_ms=(len(similarity)-1)/fs*1000,hold_samples=hold)


class Experiment:
    """Adapter to the previous experiment's ground-truth dataset runner."""
    def __init__(self,args):
        self.args = args
        self.rows,self.index,self.references,self.provenance,self.skipped,self.exclusions = [],[],[],[],[],[]
        self.traces = {}
        self.representatives = set()

    def skip(self,dataset,subject,reason):
        self.skipped.append(dict(dataset=dataset,subject=str(subject),reason=str(reason)))
        print('SKIP:',self.skipped[-1],flush=True)

    def windows(self,artifact,neural,meta):
        """Runner callback receives a FULL clip; do not subdivide/reset it."""
        args = self.args
        try:
            horizon = round(args.evaluation_ms*meta['fs']/1000)
            if neural.shape[-1]<=horizon:
                raise ValueError('Clip shorter than evaluation horizon')
            seed = derived_seed(args.seed,meta['dataset'],meta['subject'],meta['configuration'],meta['trial'],'pastd_initialization')
            result = track_pair(artifact,neural,args.beta,args.initial_power,args.initialization,seed)
            metrics = [trace_metrics(curve,meta['fs'],args.evaluation_ms,args.similarity_threshold,args.hold_ms)
                       for curve in result['primary']]
        except (ValueError,np.linalg.LinAlgError) as exc:
            self.exclusions.append(dict(**meta,reason=str(exc)))
            return
        key = f'trace_{len(self.index):06d}'
        self.index.append(dict(**meta,trace_key=key,n_samples=neural.shape[-1],n_updates=neural.shape[-1]-1,
                               channels=neural.shape[0],initialization_seed=seed))
        self.traces[key+'_primary'] = result['primary']
        self.traces[key+'_condition_matched'] = result['matched']
        self.traces[key+'_references'] = np.stack([result['reference'],
            result['differential_reference'] if result['differential_reference'] is not None else np.full(neural.shape[0],np.nan)])
        for condition,metric in zip(CONDITIONS,metrics):
            self.rows.append(dict(**meta,trace_key=key,condition=condition,**metric))
        self.references.append(dict(**meta,trace_key=key,artifact_direction_cosine=result['reference_cosine'],
            differential_reference_status=result['reference_status'],
            **{f'original_{k}':v for k,v in result['reference_info'].items()},
            **{f'derivative_{k}':v for k,v in result['differential_reference_info'].items()}))
        if meta['dataset'] not in self.representatives:
            np.savez_compressed(args.output_dir/f'representative_{meta["dataset"]}.npz',
                tracked_vectors=result['vectors'],shared_reference=result['reference'],
                primary_similarity=result['primary'],condition_matched_similarity=result['matched'],
                sample=np.arange(neural.shape[-1]),time_ms=np.arange(neural.shape[-1])/meta['fs']*1000,
                conditions=np.asarray(CONDITIONS),trace_key=np.asarray(key))
            self.representatives.add(meta['dataset'])


def aggregate_units(exp):
    """Trials within subject first, then equal subjects within PBS configuration."""
    observations = defaultdict(dict)
    cells = defaultdict(list)
    for row in exp.rows:
        key = (row['dataset'],row['subject'],row['configuration'],row['trial'])
        if row['condition'] in observations[key]:
            raise ValueError('Duplicate paired observation')
        observations[key][row['condition']] = row
        cells[(row['dataset'],row['unit'],row['subject'],row['condition'])].append(row)
    if any(set(pair)!=set(CONDITIONS) for pair in observations.values()):
        raise ValueError('Unmatched conditions')
    subject_means = defaultdict(list)
    for (dataset,unit,subject,condition),rows in cells.items():
        subject_means[(dataset,unit,condition)].append(np.mean([[r[k] for k in METRICS] for r in rows],axis=0))
    unit_rows = []
    for (dataset,unit,condition),values in sorted(subject_means.items()):
        unit_rows.append(dict(dataset=dataset,unit=unit,condition=condition,n_subjects=len(values),
                             **dict(zip(METRICS,map(float,np.mean(values,axis=0))))))
    unit_curves,curve_rows = {},[]
    for dataset in LABELS:
        selected = [r for r in exp.index if r['dataset']==dataset]
        if not selected:
            continue
        length = min(r['n_samples'] for r in selected)
        rates = {r['fs'] for r in selected}
        if len(rates)!=1:
            raise ValueError('Curve aggregation requires a common sampling rate')
        fs = rates.pop()
        # Truncate to common follow-up so cohort composition never changes
        # along the displayed curve. Full individual traces remain in NPZ.
        for kind in ('primary','condition_matched'):
            by_subject = defaultdict(list)
            for r in selected:
                by_subject[(r['unit'],r['subject'])].append(exp.traces[r['trace_key']+'_'+kind][:,:length])
            by_unit = defaultdict(list)
            for (unit,subject),curves in by_subject.items():
                by_unit[unit].append(np.mean(curves,axis=0))
            curves = np.stack([np.mean(by_unit[u],axis=0) for u in sorted(by_unit)])
            unit_curves[(dataset,kind)] = (fs,sorted(by_unit),curves)
            for j,condition in enumerate(CONDITIONS):
                center = np.mean(curves[:,j,:],axis=0)
                sd = np.std(curves[:,j,:],axis=0,ddof=1) if len(curves)>1 else np.zeros(length)
                for sample in range(length):
                    curve_rows.append(dict(dataset=dataset,reference=kind,condition=condition,
                        sample=sample,time_ms=sample/fs*1000,n_independent=len(curves),
                        mean_similarity=float(center[sample]),std_similarity=float(sd[sample])))
    return unit_rows,unit_curves,curve_rows


def statistical_summary(units):
    summaries,tests = [],[]
    for dataset in LABELS:
        selected = [r for r in units if r['dataset']==dataset]
        if not selected:
            continue
        paired = defaultdict(dict)
        for r in selected:
            paired[r['unit']][r['condition']] = r
        if any(set(p)!=set(CONDITIONS) for p in paired.values()):
            raise ValueError('Unmatched independent units')
        for condition in CONDITIONS:
            rows = [p[condition] for p in paired.values()]
            row = dict(dataset=dataset,condition=condition,n_independent=len(rows),
                       independent_unit='patient' if dataset=='swec_synthetic' else 'PBS configuration')
            for metric in METRICS:
                values = [r[metric] for r in rows]
                row[metric+'_mean'] = float(np.mean(values))
                row[metric+'_median'] = float(np.median(values))
            summaries.append(row)
        for metric in ('early_mean_similarity','restricted_convergence_ms'):
            delta = np.round([p['derivative'][metric]-p['none'][metric] for p in paired.values()],12)
            if len(delta)<2:
                statistic,pvalue,status = np.nan,np.nan,'insufficient_independent_units'
            elif np.all(delta==0):
                statistic,pvalue,status = 0.,1.,'all_differences_zero'
            else:
                result = wilcoxon(delta,alternative='two-sided',zero_method='wilcox',method='auto')
                statistic,pvalue,status = float(result.statistic),float(result.pvalue),'ok'
            tests.append(dict(dataset=dataset,metric=metric,n_independent=len(delta),
                mean_derivative_minus_none=float(np.mean(delta)),median_derivative_minus_none=float(np.median(delta)),
                improvement_direction='positive' if metric=='early_mean_similarity' else 'negative',
                statistic=statistic,p_value=pvalue,status=status))
    for row,pvalue in zip(tests,_holm([r['p_value'] for r in tests])):
        row.update(p_holm=float(pvalue),correction_family='both metrics across evaluated datasets')
    return summaries,tests


def plot_curves(unit_curves,args):
    set_tbme_style()
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':13,
        'axes.labelsize':13,'xtick.labelsize':13,'ytick.labelsize':13,'grid.alpha':.18,
        'grid.linewidth':.5,'lines.linewidth':1.4,'pdf.fonttype':42})
    for (dataset,kind),(fs,unit_names,curves) in unit_curves.items():
        for full in (False,True) if kind=='primary' else (False,):
            length = curves.shape[-1] if full else min(curves.shape[-1],round(args.plot_ms*fs/1000)+1)
            time = np.arange(length)/fs*1000
            fig,ax = plt.subplots(figsize=(3.5,3.0))
            # CONDITIONS is ordered as ('none', 'derivative'):
            # orange = without time differentiation, blue = 1st-order differentiation.
            labels = ('Without time differentiation', '1st-order differentiation')
            colors = ('tab:orange', 'tab:blue')
            linestyles = ('--', '-')
            for j, (label, color, linestyle) in enumerate(zip(
                    ('No diff.', '1st-order differentiation'),
                    ('#D55E00', '#0072B2'),
                    linestyles)):
                values = curves[:, j, :length]
                center = np.mean(values, axis=0)
                sd = values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros(length)
                ax.plot(time,center,color=color,linestyle=linestyle,label=label)
                ax.fill_between(time,np.clip(center - sd, 0, 1),np.clip(center + sd, 0, 1),color=color,alpha=.15,lw=0)
            # ax.axhline(args.similarity_threshold,color='.55',ls=':',lw=.7)
            ax.set(xlabel='Time from clip start (ms)', ylim=(0,1), xlim=(0,float(time[-1])))
            ax.set_ylabel(r'$||\widehat{u}_1^{\mathsf{T}} u_{1}||$', fontsize=20)
            ax.set_axisbelow(True)
            # No legend in the manuscript figure.
            fig.subplots_adjust(left=.22,right=.95,top=.95,bottom=.22)
            suffix = '_full' if full else ('' if kind=='primary' else '_condition_matched')
            for ext in ('pdf','png'):
                fig.savefig(args.output_dir/f'pastd_u1_convergence_{dataset}{suffix}.{ext}',dpi=600,facecolor='white')
            plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset',choices=['all',*LABELS],default='all')
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--patient-config',type=Path,default=ROOT/'experiments/seizure_preservation/config.json')
    p.add_argument('--data-root',type=Path)
    p.add_argument('--artifact-source',type=Path)
    p.add_argument('--max-patients',type=int)
    p.add_argument('--max-trials',type=int,default=6)
    p.add_argument('--calibration-trials',type=int,default=6)
    p.add_argument('--pbs-configurations',nargs='+',type=int,default=list(range(14)))
    p.add_argument('--pbs-scale',type=float,default=inspect.signature(load_swec_pbs).parameters['artifact_scale'].default)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--beta',type=float,default=.999)
    p.add_argument('--initial-power',type=float,default=1.,help='Same initial PASTd d in both arms (repository default: 1)')
    p.add_argument('--initialization',choices=['canonical','random'],default='canonical')
    p.add_argument('--evaluation-ms',type=float,default=200.,help='Prespecified common horizon for early metrics')
    p.add_argument('--plot-ms',type=float,default=200.,help='Early-time figure extent; full curves are also plotted')
    p.add_argument('--similarity-threshold',type=float,default=.95)
    p.add_argument('--hold-ms',type=float,default=20.,help='Time continuously above threshold required to confirm convergence')
    p.add_argument('--output-dir',type=Path,default=ROOT/'figures/derivative/pastd_u1_convergence')
    args = p.parse_args()
    if args.max_trials<1 or args.calibration_trials<1 or (args.max_patients is not None and args.max_patients<1):
        p.error('Patient/trial limits must be positive')
    if not all(np.isfinite(getattr(args,k)) and getattr(args,k)>0 for k in ('pbs_scale','initial_power','evaluation_ms','plot_ms')):
        p.error('Scale, initial power and horizons must be finite and positive')
    if not np.isfinite(args.beta) or not 0<args.beta<=1 or not 0<args.similarity_threshold<=1:
        p.error('Require 0 < beta <= 1 and 0 < similarity-threshold <= 1')
    if not np.isfinite(args.hold_ms) or not 0<=args.hold_ms<args.evaluation_ms:
        p.error('Require 0 <= hold-ms < evaluation-ms')
    if len(args.pbs_configurations)!=len(set(args.pbs_configurations)):
        p.error('Duplicate PBS configurations')
    return args


def main():
    args = parse_args()
    config = json.loads(args.patient_config.read_text())
    for key in ('data_root','artifact_source'):
        if getattr(args,key) is not None:
            config[key] = str(getattr(args,key))
    fs = config['target_fs']
    if not isinstance(fs,int) or fs<=0 or config['artifact']['high_fs']%fs or min(round(args.plot_ms*fs/1000),round(args.evaluation_ms*fs/1000))<1:
        raise ValueError('Require valid analysis rate dividing artifact high_fs and horizons of >=1 sample')
    config['seed'] = args.seed
    config['samples_per_patient'] = None
    for split,limit in [('test',args.max_trials),('train',args.calibration_trials)]:
        config['max_windows_per_class'][split] = [(limit+1)//2,limit//2]
    args.output_dir.mkdir(parents=True,exist_ok=True)
    exp = Experiment(args)
    if args.dataset=='all':
        for dataset in ('ERAASR','Dictionary Learning'):
            exp.skip(dataset,'all','No separately available artifact ground truth for this experiment')
    with threadpool_limits(limits=1):
        run_datasets(exp,config)
        units,unit_curves,curve_rows = aggregate_units(exp)
    summary,tests = statistical_summary(units)
    write_csv(args.output_dir/'skipped_datasets.csv',exp.skipped,('dataset','subject','reason'))
    write_csv(args.output_dir/'excluded_clips.csv',exp.exclusions,('subject','trial','fs','dataset','configuration','unit','reason'))
    write_json(args.output_dir/'analysis_metadata.json',dict(
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},config=config,
        provenance=exp.provenance,notes=[
            'Primary reference is original-artifact u1 from uncentered SVD of A[:,1:], shared by both arms.',
            'Truth uses the full clip for evaluation only. PASTd sees contaminated samples only, without warm start, filtering or normalization.',
            'Raw updates use X[:,1:]; differential updates use diff(X). Both have T-1 updates at original indices 1..T-1.',
            'Index 0 is initialization; all later similarities are measured immediately after that sample update.',
            'One tracker per condition/clip, never reset at analysis-window boundaries; rank=1 and identical beta, U0 and d0.',
            'Primary similarity is absolute cosine. Sign flips are irrelevant. A spectral tie rejects a nonidentifiable original reference.',
            'Secondary condition-matched reference and original/derivative artifact-direction cosine diagnose target rotation.',
            'Early mean similarity is trapezoidal AUC divided by the fixed early horizon; higher is better.',
            'Convergence time is confirmation of a sustained threshold run, not the first instantaneous crossing.',
            'Restricted convergence delay caps time at the early horizon; a censoring/event flag is always retained. Lower is better.',
            'Statistical units are patients for synthetic and PBS configurations after equal-weight subject averaging.',
            'PBS configuration independence is assumed; inference is conditional on the sampled neural cohort.',
            'Plots show unit mean +/- sample SD, not confidence intervals. Full plots use common follow-up without changing cohort composition.',
            'The existing dataset runner supplies the same sampling, resampling and artifact-amplitude convention as the energy-ratio experiment.',
        ]))
    if not exp.rows:
        raise SystemExit('No valid tracking pairs; see exclusion/skip CSVs')
    for name,rows in [('clip_metrics',exp.rows),('trace_index',exp.index),('artifact_reference_diagnostics',exp.references),
                      ('independent_unit_metrics',units),('curve_summary',curve_rows),('convergence_summary',summary),('paired_statistics',tests)]:
        write_csv(args.output_dir/f'{name}.csv',rows)
    np.savez_compressed(args.output_dir/'tracking_traces.npz',conditions=np.asarray(CONDITIONS),**exp.traces)
    plot_curves(unit_curves,args)
    print('\nSuccessfully evaluated:',[d for d in LABELS if any(r['dataset']==d for r in exp.rows)])
    print('Valid paired clips:',len(exp.index),'excluded:',len(exp.exclusions))
    for row in summary:
        print(row)
    print('\nPaired statistics (derivative minus none):')
    for row in tests:
        print(row)
    print('\nFiles created in',args.output_dir)
    for path in sorted(args.output_dir.glob('*')):
        if path.is_file():
            print(path.name)


if __name__=='__main__':
    main()
