"""Run the existing DAST pipeline on PBS configurations 0--13 and fill the table.

python tests/run_pbs_dast_table.py

Matches the PBS derivative-ablation defaults: separated SWEC channels, rank 1,
beta .999, first difference, no filtering or normalization. Uses a common
clean reference and common available interval across configurations. Reports
pooled MSE and OUTPUT SNR, not SNR improvement; includes online startup.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import inspect
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
from threadpoolctl import threadpool_limits

from data_loading import ROOT, Dataset, load_swec_pbs
from pipeline import PipelineConfig, run_pipeline
from plot_stable_low_dim_artifact_subspace import load_pbs_artifact
from experiments.seizure_preservation.common import write_csv, write_json, array_sha256

CONDITIONS = [
    'Biphasic; no interphase gap',
    'Baseline biphasic configuration',
    r'Interphase gap: 1~ms',
    r'Triphasic; $D_1=0.5$, $D_2=1$~ms',
    r'$A_1=A_2=0.04$~mA',
    r'$D_1=D_2=1.5$~ms',
    r'Pulse period: 10~ms',
    r'Pulse period: 1.5~ms',
    r'Stimulator delays: $(1,2,3,4)$~ms',
    r'Stimulator delays: $(0.5,1,1.5,2)$~ms',
    r'Stimulation electrodes repositioned relative to Config.~1',
    r'Recording electrodes repositioned relative to Config.~10',
    r'Config.~11 with physical noise',
    r'Config.~11 without stimulation',
]


def reconstruction_metrics(estimate, reference):
    estimate,reference = (np.asarray(x,dtype=np.float64) for x in (estimate,reference))
    if estimate.shape!=reference.shape or not reference.size:
        raise ValueError('Estimate and reference must have matching nonempty shapes')
    if not np.isfinite(estimate).all() or not np.isfinite(reference).all():
        raise ValueError('Nonfinite signals')
    mse = float(np.mean((estimate-reference)**2))
    signal_power = float(np.mean(reference**2))
    if signal_power<=0:
        raise ValueError('Zero-energy neural reference')
    snr = float(10*np.log10(signal_power/mse)) if mse else np.inf
    return dict(mse_uv2=mse,snr_db=snr,reference_power_uv2=signal_power)


def render_table(rows):
    indexed = {r['configuration']:r for r in rows}
    if set(indexed)!=set(range(14)) or len(rows)!=14:
        raise ValueError('Need exactly configurations 0--13, including the control')
    lines = [r'\begin{table}[t]',r'\centering',
        r'\caption{PBS configurations and cancellation performance.',
        r'MSE and SNR are evaluated against the clean neural signal',
        r'used to construct the semi-synthetic recordings.}',
        r'\label{tab:pbs_configurations}',r'\footnotesize',
        r'\setlength{\tabcolsep}{3pt}',r'\renewcommand{\arraystretch}{1.08}',
        r'\begin{tabularx}{\columnwidth}{@{}cXcc@{}}',r'\toprule',
        r'Config. & Condition & MSE & SNR \\',
        r'        &           & ($\mu$V$^2$) & (dB) \\',r'\midrule']
    for c,condition in enumerate(CONDITIONS):
        if c==13:
            lines.append(r'\midrule')
        row = indexed[c]
        snr = f'{row["snr_db"]:.2f}' if np.isfinite(row['snr_db']) else r'$\infty$'
        lines.append(f'{c:<2} & {condition} & {row["mse_uv2"]:.2f} & {snr} '+r'\\')
    lines.extend([r'\bottomrule',r'\end{tabularx}',r'\smallskip',r'\begin{minipage}{\columnwidth}',r'\scriptsize',
        r'Baseline (Config.~1): $A_1=A_2=0.16$~mA,',
        r'$D_1=D_2=0.5$~ms, interphase gap $0.3$~ms,',
        r'and pulse period $4$~ms.',
        r'Unless specified, pulse parameters follow the baseline.',
        r'Config.~0 has no gap; Config.~3 uses the triphasic',
        r'sequence $D_1$--$D_2$--$D_1$ without gaps.',
        r'Config.~13 has $A_1=A_2=0$.',r'\end{minipage}',r'\end{table}'])
    return '\n'.join(lines)+'\n'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--output-dir',type=Path,default=ROOT/'results/pbs_dast_table')
    p.add_argument('--artifact-scale',type=float,default=inspect.signature(load_swec_pbs).parameters['artifact_scale'].default)
    p.add_argument('--channels',choices=['separated','adjacent'],default='separated')
    p.add_argument('--rank',type=int,default=1)
    p.add_argument('--beta',type=float,default=.999)
    p.add_argument('--skip-ms',type=float,default=0.,help='Optional fixed evaluation burn-in; default includes all startup')
    args = p.parse_args()
    if not np.isfinite(args.beta) or not 0<args.beta<=1 or not 1<=args.rank<=8:
        p.error('Require 0 < beta <= 1 and rank in 1..8')
    if not np.isfinite(args.artifact_scale) or args.artifact_scale<=0 or not np.isfinite(args.skip_ms) or args.skip_ms<0:
        p.error('Artifact scale must be positive and skip-ms nonnegative, both finite')
    return args


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    # Load the existing mixture ONCE to obtain its exact neural trial, time
    # crop and channel mapping. Reuse the isolated PBS loader for the sweep,
    # avoiding fourteen loads of the multi-gigabyte synthetic archive.
    print('Loading common clean reference through load_swec_pbs',flush=True)
    base = load_swec_pbs(root=args.root,configuration_number=0,artifact_scale=args.artifact_scale,
                         separated_recording_channels=args.channels=='separated')
    clean = base.y_clean.copy()
    fs = base.fs
    artifacts = {}
    source_rows = []
    for c in range(14):
        print(f'Loading PBS {c}',flush=True)
        ds = load_pbs_artifact(args.root,c)
        if ds.fs!=fs or ds.X.shape[:2]!=clean.shape[:2] or not np.isfinite(ds.X).all():
            raise ValueError(f'PBS {c}: incompatible or nonfinite artifact')
        artifacts[c] = args.artifact_scale*ds.X
        source_rows.append(dict(configuration=c,original_samples=ds.X.shape[-1],**ds.metadata))
    length = min(clean.shape[-1],*(x.shape[-1] for x in artifacts.values()))
    clean = clean[...,:length]
    np.testing.assert_allclose(base.X[...,:length],clean+artifacts[0][...,:length],rtol=1e-12,atol=1e-12)
    del base
    config = PipelineConfig(n_test=1,preprocessing='none',normalization='none',
        use_derivative_for_tracking=True,derivative_order=1,tracker='PASTd',rank=args.rank,beta=args.beta,
        adaptive_k=False,harmonic_filter=False,compute_ideal_u=False,removal_method='projection')
    first = max(1,int(np.ceil(args.skip_ms*fs/1000)))
    if first>=length:
        raise ValueError('No samples remain after evaluation burn-in')
    rows,channels = [],[]
    with threadpool_limits(limits=1):
        for c in range(14):
            print(f'Running DAST: PBS {c}',flush=True)
            artifact = artifacts[c][...,:length]
            mixture = clean+artifact
            rate = {6:100.,7:1000/1.5,13:0.}.get(c,250.)
            dataset = Dataset(name=f'swec_pbs_config{c}',X=mixture,y_clean=clean,fs=fs,stim_rate=rate,
                stim_channels=[0,1,2,3],metadata=dict(configuration=c,artifact_scale=args.artifact_scale))
            result = run_pipeline(dataset,config)
            # Pipeline output j corresponds to original sample j+1 after the
            # first difference; no lag fitting, demeaning or gain fitting.
            estimate = result.stages['cleaned'][...,first-1:]
            reference = clean[...,first:]
            raw = mixture[...,first:]
            metrics = reconstruction_metrics(estimate,reference)
            before = reconstruction_metrics(raw,reference)
            rows.append(dict(configuration=c,condition=CONDITIONS[c],**metrics,
                input_mse_uv2=before['mse_uv2'],input_snr_db=before['snr_db'],
                snr_improvement_db=metrics['snr_db']-before['snr_db'],
                n_channels=clean.shape[1],n_samples=reference.shape[-1],fs=fs,
                first_sample=first,last_sample=length-1))
            for ch in range(clean.shape[1]):
                channels.append(dict(configuration=c,recording_channel=ch,
                    **reconstruction_metrics(estimate[:,ch],reference[:,ch])))
            np.savez_compressed(args.output_dir/f'configuration_{c:02d}.npz',
                cleaned=estimate,reference=reference,contaminated=raw,artifact=artifact[...,first:],
                fs=np.array(fs),original_sample_indices=np.arange(first,length),configuration=np.array(c))
            source_rows[c].update(clean_sha256=array_sha256(reference),artifact_sha256=array_sha256(artifact),
                                  used_samples=length,stim_rate_hz=rate)
            print(f"  MSE={metrics['mse_uv2']:.6f} uV^2; output SNR={metrics['snr_db']:.6f} dB",flush=True)
            del result
    write_csv(args.output_dir/'pbs_dast_metrics.csv',rows)
    write_csv(args.output_dir/'pbs_dast_per_channel.csv',channels)
    (args.output_dir/'pbs_configurations.tex').write_text(render_table(rows))
    write_json(args.output_dir/'protocol.json',dict(
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        pipeline=asdict(config),sources=source_rows,
        neural_reference=dict(loader='load_swec_pbs(configuration_number=0)',trial=0,
            channels=[0,11,22,33,44,55,66,77] if args.channels=='separated' else list(range(8)),
            source_crop_seconds=[.1,1.0],signal_unit='uV, following existing loader and PBS ablation convention'),
        evaluation=dict(fs=fs,first_sample=first,last_sample=length-1,n_samples=length-first,
            start_ms=first/fs*1000,stop_exclusive_ms=length/fs*1000,
            mse='mean((cleaned-reference)**2), pooled across all eight channels and evaluated samples',
            snr='10*log10(sum(reference**2)/sum((cleaned-reference)**2)); output SNR, not improvement'),
        notes=['Same clean trial, channel mapping and common available interval in every configuration.',
               'DAST resets independently for each configuration. No clean neural signal enters tracker updates or projection.',
               'Only the first sample is lost to differentiation by default; online startup is included.',
               'Config 13 retains the recorded no-stimulation PBS signal/noise; it is not replaced with zeros and DAST is not bypassed.',
               'Conditions and pulse parameters follow the user-supplied table; they are not inferred from waveforms.',
               'MSE and output SNR are pooled, not averages of per-channel dB SNRs. Per-channel values are separately available.',
               'Results are descriptive for the existing loader neural trial, not a patient-cohort average.']))
    protocol = (f'DAST: rank {args.rank}, beta {args.beta:g}, first-order differences, no filtering/normalization/harmonic stage.\n'
        f'Clean trial 0; channels {args.channels}; PBS amplitude multiplier {args.artifact_scale:g}; fs={fs:g} Hz.\n'
        f'All configurations use {length-first} samples/channel, original indices {first}..{length-1}, after matching the common interval.\n'
        'MSE and output SNR are pooled across all eight channels; startup is included unless --skip-ms is specified.\n'
        'SNR = 10 log10(sum(clean^2)/sum((cleaned-clean)^2)). No mean removal, lag optimization or amplitude fitting.\n'
        'Configuration 13 is the recorded no-stimulation control, with the same cancellation applied.\n')
    (args.output_dir/'README.txt').write_text(protocol)
    print('\n'+protocol)
    print('Completed table:',args.output_dir/'pbs_configurations.tex')
    print('Full-precision metrics:',args.output_dir/'pbs_dast_metrics.csv')


if __name__=='__main__':
    main()
