"""python -m experiments.seizure_preservation.run --help"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess
from pathlib import Path
import numpy as np
import torch
import scipy
import sklearn
import h5py
from benchmarks.methods import METHODS, BenchmarkConfig
from pipeline import PipelineConfig
from .common import write_json, write_csv, derived_seed, seed_everything, array_sha256
from .data import load_patient_data, split_patient_data, extract_windows
from .training import train_eegnet, train_gru, train_chrononet, fit_normalization, evaluate_decoder
from .cancellation import generate_synthetic_artifact, apply_artifact_cancellation, calibrate_lrr
from .checkpoints import DATA_SETTINGS, load_patient_decoders
from .reporting import add_restoration, aggregate_results, run_statistical_analysis, plot_results

DEFAULT_CONFIG = Path(__file__).with_name('config.json')


def validate_config(c):
    if c['target_fs'] != 1024:
        raise ValueError('This protocol requires target_fs=1024')
    if not c['patients'] or len(set(c['patients'])) != len(c['patients']) or any(p not in range(1,19) for p in c['patients']):
        raise ValueError('Choose unique patients in 1..18')
    if not 0 <= c['overlap'] < 1 or c['window_seconds'] <= 0:
        raise ValueError('Invalid window length/overlap')
    ratios = np.asarray(c['split_ratios'])
    if ratios.shape != (3,) or np.any(ratios<=0) or not np.isclose(ratios.sum(),1):
        raise ValueError('Three positive split ratios must sum to one')
    sample_limit = c.get('samples_per_patient')
    if sample_limit is not None and (isinstance(sample_limit, bool) or not isinstance(sample_limit, int) or sample_limit < 6):
        raise ValueError('samples_per_patient must be null or an integer >= 6')
    for split in ('train','validation','test'):
        caps = c['max_windows_per_class'][split]
        if len(caps)!=2 or any(v is not None and (not isinstance(v,int) or v<1) for v in caps):
            raise ValueError('Window caps must be null or positive integers for each class')
    if c['split_boundary_guard_seconds'] < 0 or c['seizure_group_margin_seconds'] < c['split_boundary_guard_seconds']:
        raise ValueError('Seizure grouping margin must cover the nonnegative split guard')
    if c['two_seizure_policy'] not in ('negative_only_validation','error'):
        raise ValueError('Unknown two-seizure policy')
    if not {'Clean','Contaminated','DAST','SVD','ERAASR'} <= set(c['methods']):
        raise ValueError('The five primary conditions are required')
    if len(c['methods']) != len(set(c['methods'])) or any(m not in {'Clean','Contaminated','DAST','SVD','ERAASR','LRR',*METHODS} for m in c['methods']):
        raise ValueError('Unknown/duplicate cancellation method')
    if 'LRR' in c['methods'] or 'lrr' in c['methods']:
        if 'LRR' in c['methods'] and 'lrr' in c['methods']:
            raise ValueError('Use only one LRR spelling, preferably LRR')
        count = c['cancellation'].get('lrr', {}).get('calibration_windows')
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError('LRR requires a positive calibration_windows count')
    if not c['decoders'] or len(set(c['decoders']))!=len(c['decoders']) or any(d not in ('EEGNet','GRU','ChronoNet') for d in c['decoders']):
        raise ValueError('Unknown/duplicate decoder')
    dast = PipelineConfig(**c['cancellation']['dast'])
    if dast.tracker != 'PASTd' or not dast.use_derivative_for_tracking or dast.derivative_order < 1:
        raise ValueError('DAST is defined here as derivative-driven PASTd')
    if dast.removal_method != 'projection' or dast.carry_tracker_across_trials:
        raise ValueError('Primary DAST uses per-window projection without cross-window state')
    c['evaluation_offset_samples'] = dast.derivative_order
    if round(c['window_seconds']*1024)-dast.derivative_order < 32:
        raise ValueError('Window too short for the decoder architectures')
    baseline = BenchmarkConfig(**c['cancellation']['baseline'])
    if baseline.template_history is None or baseline.template_history < 1:
        raise ValueError('Use backward template subtraction; no clean/test reference template fitting')
    a = c['artifact']
    if a['high_fs'] % 1024 or a['high_fs'] < 1024 or a['n_stim_channels'] < 1:
        raise ValueError('Invalid artifact sampling rate/channel count')
    if not a['period_samples'] or any(not isinstance(p,int) or p<2 for p in a['period_samples']):
        raise ValueError('Artifact pulse periods must be integer samples >=2')
    for key in ('current_range','pulse_frequency_range','rms_ratio_range','first_pulse_seconds'):
        pair = a[key]
        if len(pair)!=2 or not np.isfinite(pair).all() or not 0<pair[0]<=pair[1]:
            raise ValueError(f'Invalid artifact {key}')
    if a['end_margin_seconds'] < 0 or a['first_pulse_seconds'][1]+a['end_margin_seconds']+2*max(a['period_samples'])/1024 >= c['window_seconds']:
        raise ValueError('Window must fit at least two complete pulse periods')
    if max(a['pulse_frequency_range']) >= a['high_fs']/2:
        raise ValueError('Pulse carrier must be below generation Nyquist')
    for key in ('channel_rank','pulse_rank','omit_channels','omit_pulses'):
        value = c['cancellation']['eraasr'][key]
        if not isinstance(value,int) or value < (1 if key.startswith('omit') else 0):
            raise ValueError(f'Invalid ERAASR {key}')
    t = c['training']
    if any(t[k] < 1 for k in ('epochs','patience','batch_size','torch_threads')) or not t['learning_rates'] or any(v<=0 for v in t['learning_rates']):
        raise ValueError('Training counts and learning rates must be positive')
    if t['normalization_epsilon'] <= 0 or t['gradient_clip'] <= 0 or c['restoration_epsilon'] <= 0:
        raise ValueError('Numerical tolerances must be positive')
    if c['statistics']['correction'] != 'holm' or c['statistics']['family'] != 'all_decoders_baselines_and_metrics':
        raise ValueError('Only global Holm correction is implemented')
    if not 0 < c['statistics']['alpha'] < 1:
        raise ValueError('Invalid statistical alpha')
    return c


def source_manifest(config):
    project = Path(__file__).resolve().parents[2]
    paths = list(Path(__file__).parent.glob('*.py')) + [project/p for p in (
        'algorithms.py','pipeline.py','covariance.py','data_loading.py','plotting.py','utils.py',
        'benchmarks/methods.py','benchmarks/lrr.py','benchmarks/_pulse_low_rank_tv.py')]
    paths += [Path(config['eegnet_source']), Path(config['artifact_source'])]
    return {str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def _save_provenance(config, directory, sources):
    directory = Path(directory)
    project = Path(__file__).resolve().parents[2]
    for source in sources:
        path = Path(source)
        relative = path.relative_to(project) if path.is_relative_to(project) else Path('external')/path.name
        target = directory/'sources'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path,target)
    write_json(directory/'source_hashes.json',sources)
    try:
        revision = subprocess.check_output(['git','rev-parse','HEAD'], cwd=project, text=True).strip()
        diff = subprocess.check_output(['git','diff'],cwd=project,text=True)
        (directory/'working_tree.patch').write_text(diff)
    except (OSError,subprocess.CalledProcessError):
        revision = None
    write_json(directory/'environment.json', dict(python=platform.python_version(), platform=platform.platform(),
        numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__, torch=str(torch.__version__),
        h5py=h5py.__version__, cuda=torch.version.cuda, cuda_available=torch.cuda.is_available(),
        gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        git_revision=revision, deterministic_algorithms=True))


def _load_results(path):
    with Path(path).open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in row:
            if key not in ('patient','decoder','method'):
                row[key] = float(row[key])
    return rows


def run_patient(patient_id, config, directory):
    patient = load_patient_data(patient_id, config)
    directory = Path(directory)/patient.patient
    directory.mkdir(parents=True, exist_ok=True)
    print(f'{patient.patient}: splitting clean recordings before extraction', flush=True)
    groups = split_patient_data(patient, config, directory)
    cache = directory/'signals'
    offset = config['evaluation_offset_samples']
    if config.get('load_decoders'):
        models, normalization, val_y = load_patient_decoders(patient,config,directory)
        test_x,test_y = extract_windows(patient,groups['test'],'test',config,cache)
        source = Path(config['load_decoders'])/patient.patient/'signals'
        if (source/'test_windows.csv').read_bytes() != (cache/'test_windows.csv').read_bytes():
            raise ValueError('Selected test windows differ from the checkpoint run')
        # Keep validation labels for provenance and future checkpoint reuse.
        np.save(cache/'validation_labels.npy',val_y)
        print(f'{patient.patient}: loaded frozen decoders; skipping training/validation signal extraction and training',flush=True)
    else:
        splits = {s: extract_windows(patient, groups[s], s, config, cache) for s in groups}
        train_x,train_y = splits['train']
        val_x,val_y = splits['validation']
        test_x,test_y = splits['test']
        normalization = fit_normalization(train_x,offset,config['training']['normalization_epsilon'])
        trainers = dict(EEGNet=train_eegnet,GRU=train_gru,ChronoNet=train_chrononet)
        models = {name:trainers[name](train_x,train_y,val_x,val_y,normalization,config,directory,patient.patient)
                  for name in config['decoders']}
        del splits,train_x,val_x
    if set(np.unique(test_y)) != {0,1}:
        raise ValueError(f'{patient.patient}: test set must contain both classes')
    mean,scale = normalization
    np.savez(directory/'normalization.npz',mean=mean,scale=scale,fitted_on='clean_train_only',offset=offset)
    train_rms = float(np.sqrt(np.mean(mean.astype(float)**2+scale.astype(float)**2)))
    # Decoder selection is complete before any test artifact is generated.
    lrr_weights = None
    if any(m in config["methods"] for m in ("LRR", "lrr")):
        lrr_weights = calibrate_lrr(test_x.shape[1], test_x.shape[2], train_rms,
                                    config, patient.patient, directory)
    shape = test_x.shape
    mixed = np.lib.format.open_memmap(cache/'Contaminated.npy',mode='w+',dtype=np.float32,shape=shape)
    artifacts = np.lib.format.open_memmap(cache/'artifact.npy',mode='w+',dtype=np.float32,shape=shape)
    methods = [m for m in config['methods'] if m not in ('Clean','Contaminated')]
    outputs = {m:np.lib.format.open_memmap(cache/f'{m}.npy',mode='w+',dtype=np.float32,
        shape=(shape[0],shape[1],shape[2]-offset)) for m in methods}
    with (directory/'artifact_parameters.jsonl').open('w') as provenance:
        for i,clean in enumerate(test_x):
            seed = derived_seed(config['seed'],patient.patient,'heldout_artifact',i)
            contaminated, artifact, params = generate_synthetic_artifact(clean,train_rms,config,seed)
            mixed[i], artifacts[i] = contaminated,artifact
            contaminated.flags.writeable = False
            input_hash = array_sha256(contaminated)
            for method in methods:
                outputs[method][i] = apply_artifact_cancellation(contaminated,method,params,config,lrr_weights=lrr_weights)
                if array_sha256(contaminated) != input_hash:
                    raise RuntimeError(f'{method} modified shared contaminated input')
            provenance.write(json.dumps(dict(window=i,**params,clean_sha256=array_sha256(clean),
                contaminated_sha256=input_hash,artifact_sha256=array_sha256(artifact)))+'\n')
            if i % 50 == 0 or i==len(test_x)-1:
                print(f'{patient.patient}: matched cancellation {i+1}/{len(test_x)}',flush=True)
    mixed.flush(); artifacts.flush()
    for x in outputs.values():
        x.flush()
    rows = []
    prediction_dir = directory/'predictions'
    prediction_dir.mkdir(exist_ok=True)
    for name,model in models.items():
        for method in config['methods']:
            x = test_x if method=='Clean' else mixed if method=='Contaminated' else outputs[method]
            metrics, probabilities = evaluate_decoder(model,x,test_y,normalization,config,
                                                offset if method in ('Clean','Contaminated') else 0)
            rows.append(dict(patient=patient.patient,decoder=name,method=method,**metrics))
            np.savez_compressed(prediction_dir/f'{name}_{method}.npz', labels=test_y,probabilities=probabilities,
                window_indices=np.arange(len(test_y)), sample_offset=offset, sampling_rate=config['target_fs'])
    add_restoration(rows,config['restoration_epsilon'])
    write_csv(directory/'results.csv',rows)
    del test_x,mixed,artifacts,outputs,models,x,clean
    # Remove only regenerable caches created in this patient directory.
    for path in cache.glob('*.npy'):
        remove = (path.name.endswith('_clean.npy') and not config['storage']['keep_clean_windows'])
        remove |= (path.stem in methods and not config['storage']['keep_cancelled_windows'])
        remove |= (path.stem in ('artifact','Contaminated') and not config['storage']['keep_artifacts'])
        if remove:
            path.unlink()
    write_json(directory/'completed.json',dict(patient=patient.patient,completed=True,
        test_windows=len(test_y), validation_classes=np.unique(val_y).tolist(),
        artifact_realizations='Held-out only; no artifact-based hyperparameter tuning',
        eraasr_variant='single_trial_channel_then_pulse_PCR; nPC_trials=0'))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def run(config, resume=False, plan_only=False):
    validate_config(config)
    torch.set_num_threads(config['training']['torch_threads'])
    seed_everything(config['seed'])
    directory = Path(config['output_dir']).resolve()
    if config.get('load_decoders') and Path(config['load_decoders']).resolve() == directory:
        raise ValueError('Checkpoint source and output must be different directories')
    if config.get('load_decoders'):
        config['loaded_checkpoint_hashes'] = {}
        for patient_id in config['patients']:
            for name in config['decoders']:
                path = Path(config['load_decoders'])/f'ID{patient_id:02d}'/name/'best.pt'
                config['loaded_checkpoint_hashes'][str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    exact = dict(config, plan_only=plan_only)
    sources = source_manifest(config)
    config_file = directory/'config.json'
    if directory.exists() and any(directory.iterdir()):
        if not resume:
            raise FileExistsError(f'{directory} is not empty; choose a new output directory or --resume')
        if not config_file.exists() or json.loads(config_file.read_text()) != exact:
            raise ValueError('Resume configuration differs from saved configuration')
        if json.loads((directory/'source_hashes.json').read_text()) != sources:
            raise ValueError('Resume code differs from saved source hashes; use a new output directory')
    else:
        directory.mkdir(parents=True,exist_ok=True)
        write_json(config_file,exact)
        _save_provenance(config,directory,sources)
    all_rows, audit = [], []
    for patient_id in config['patients']:
        patient = load_patient_data(patient_id,config)
        patient_dir = directory/patient.patient
        stat = patient.path.stat()
        fingerprint = dict(path=str(patient.path),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,
            channels=patient.channels,samples=patient.samples,fs=patient.fs,
            annotations_sha256=array_sha256(patient.seizures))
        fingerprint_path = patient_dir/'input_fingerprint.json'
        if fingerprint_path.exists() and json.loads(fingerprint_path.read_text()) != fingerprint:
            raise ValueError(f'{patient.patient}: source data changed since prior run')
        write_json(fingerprint_path,fingerprint)
        if plan_only:
            groups = split_patient_data(patient,config,patient_dir)
            audit.append(dict(patient=patient.patient, channels=patient.channels,native_fs=patient.fs,
                target_fs=config['target_fs'],seizures=len(patient.seizures),
                **{s+'_groups':len(gs) for s,gs in groups.items()},
                **{s+'_seizures':sum(len(g['seizures']) for g in gs) for s,gs in groups.items()}))
            print(audit[-1],flush=True)
            continue
        if (patient_dir/'completed.json').exists():
            all_rows.extend(_load_results(patient_dir/'results.csv'))
        else:
            all_rows.extend(run_patient(patient_id,config,directory))
        write_csv(directory/'results.csv',all_rows)
    if plan_only:
        write_csv(directory/'split_audit.csv',audit)
        return
    write_csv(directory/'aggregated_results.csv',aggregate_results(all_rows))
    run_statistical_analysis(all_rows,config,directory)
    plot_results(all_rows,config,directory)
    write_json(directory/'completed.json',dict(completed=True,patients=config['patients'],
        n_patients=len(config['patients']),smoke_test=config.get('smoke_test',False)))
    print(f'Finished {len(config["patients"])} patients: {directory}',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--load-decoders',type=Path,help='Prior run directory: reuse frozen best.pt checkpoints; skip decoder training')
    parser.add_argument('--patients',nargs='+',type=int)
    parser.add_argument('--samples-per-patient',type=int,help='Total window budget per patient across train/validation/test; >=6. Overrides class caps, targets balanced classes.')
    parser.add_argument('--device',choices=['auto','cpu','cuda','cuda:0','cuda:1'])
    parser.add_argument('--resume',action='store_true',help='Skip finished patients only; partial patients restart')
    parser.add_argument('--plan-only',action='store_true',help='Audit/snapshot splits without signal extraction or training')
    parser.add_argument('--smoke',action='store_true',help='One-epoch, tiny-data integration check; NOT scientific results')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.smoke:
        config['smoke_test'] = True
        config['patients'] = [4]
        config['output_dir'] = 'results/seizure_preservation_smoke'
        config['max_windows_per_class'] = dict(train=[8,8],validation=[4,4],test=[4,4])
        config['training'].update(epochs=1,patience=1,batch_size=4,learning_rates=[0.001])
        config['training']['eegnet'].update(F1=4,D=2,F2=8)
        config['training']['gru'].update(hidden_size=8,layers=1)
        config['training']['chrononet'].update(filters=4,hidden_size=8)
        config['storage'] = dict(keep_clean_windows=True,keep_cancelled_windows=True,keep_artifacts=True)
    if args.load_decoders is not None:
        config['load_decoders'] = str(args.load_decoders.resolve())
    if config.get('load_decoders'):
        if args.smoke:
            parser.error('--smoke cannot be combined with checkpoint loading')
        source_config = json.loads((Path(config['load_decoders'])/'config.json').read_text())
        for key in DATA_SETTINGS:
            config[key] = source_config.get(key)
        config['smoke_test'] = source_config.get('smoke_test',False)
    if args.output_dir is not None:
        config['output_dir'] = str(args.output_dir)
    if args.patients is not None:
        config['patients'] = args.patients
    if args.samples_per_patient is not None:
        config['samples_per_patient'] = args.samples_per_patient
    if args.device is not None:
        config['training']['device'] = args.device
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=config["training"]["torch_threads"]):
        run(config,args.resume,args.plan_only)


if __name__ == '__main__':
    main()
