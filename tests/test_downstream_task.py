"""Protocol invariants for the SWEC seizure-preservation experiment.

Run: python -m pytest tests/test_downstream_task.py
"""
import copy
import json
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest
import torch

from experiments.seizure_preservation.data import load_patient_data, split_patient_data, extract_windows
from experiments.seizure_preservation.common import derived_seed
from experiments.seizure_preservation.run import DEFAULT_CONFIG, validate_config
from experiments.seizure_preservation.cancellation import generate_synthetic_artifact, apply_artifact_cancellation, _pca_regression
from experiments.seizure_preservation.models import build_model
from experiments.seizure_preservation.training import fit_normalization, evaluate_decoder, train_gru
from experiments.seizure_preservation.reporting import compute_restoration_ratio, run_statistical_analysis, aggregate_results, _holm


@pytest.fixture
def config():
    return validate_config(json.loads(DEFAULT_CONFIG.read_text()))


def make_patient(tmp_path, config, seizures):
    config.update(data_root=str(tmp_path),recording_seconds=20,seizure_group_margin_seconds=0,
                  split_boundary_guard_seconds=0,window_seconds=1,overlap=0)
    config['max_windows_per_class'] = {s:[3,3] for s in ('train','validation','test')}
    fs, channels, n = 256, 3, 10
    t = np.arange(n*20*fs)/fs
    signal = np.stack([np.sin(2*np.pi*(j+3)*t) for j in range(channels)]).astype(np.float32)
    with h5py.File(tmp_path/'ID01.h5','w') as f:
        g = f.create_group('ID01')
        g.create_dataset('EEG',data=signal)
        g.create_dataset('seizure_begin',data=[s for s,e in seizures])
        g.create_dataset('seizure_end',data=[e for s,e in seizures])
        g.attrs.update(fs_target=fs,n_channels=channels,n_files=n,
                       file_order='\n'.join(f'ID01_{i}h.mat' for i in range(1,n+1)))
    return load_patient_data(1,config)


def test_split_whole_recordings_and_cross_boundary_seizure(tmp_path, config):
    patient = make_patient(tmp_path,config,[(19,22),(65,67),(125,127)])
    splits = split_patient_data(patient,config,tmp_path/'out')
    assert splits == split_patient_data(patient,config)
    group_ids = [g['group'] for groups in splits.values() for g in groups]
    assert len(group_ids)==len(set(group_ids))
    ownership = {}
    for split,groups in splits.items():
        for g in groups:
            for seizure in g['seizures']:
                assert seizure not in ownership
                ownership[seizure]=split
    assert set(ownership.values()) == {'train','validation','test'}
    cross = next(g for groups in splits.values() for g in groups if 0 in g['seizures'])
    assert cross['recording_indices']==[0,1]
    bounds = []
    for split,groups in splits.items():
        x,y = extract_windows(patient,groups,split,config,tmp_path/'out')
        assert x.dtype==np.float32 and x.shape[1:]==(3,1024)
        assert set(y)=={0,1}
        import csv
        for row in csv.DictReader((tmp_path/'out'/f'{split}_windows.csv').open()):
            start,stop = int(row['start']),int(row['stop'])
            assert any(g['start']<=start<stop<=g['stop'] for g in groups)
            expected = np.any((patient.seizures[:,0]<stop)&(patient.seizures[:,1]>start))
            assert int(row['label'])==int(expected)
            bounds.append((split,start,stop))
    for split,a,b in bounds:
        assert not any(split!=other and a<d and b>c for other,c,d in bounds)


def test_two_seizures_flag_validation_without_leakage(tmp_path,config):
    patient=make_patient(tmp_path,config,[(21,23),(121,123)])
    splits=split_patient_data(patient,config,tmp_path/'out')
    assert sum(len(g['seizures']) for g in splits['train'])==1
    assert sum(len(g['seizures']) for g in splits['test'])==1
    assert all(not g['seizures'] for g in splits['validation'])
    assert 'Validation loss' in (tmp_path/'out'/'splits.json').read_text()
    config['two_seizure_policy']='error'
    with pytest.raises(ValueError):
        split_patient_data(patient,config)


def test_single_seizure_group_refused(tmp_path,config):
    patient=make_patient(tmp_path,config,[(21,23),(25,27)])
    with pytest.raises(ValueError,match='fewer than two'):
        split_patient_data(patient,config)


def test_normalization_train_only_and_stable():
    rng=np.random.default_rng(8)
    clean=rng.normal(size=(5,3,50)).astype(np.float32)
    mean,scale=fit_normalization(clean,1,1e-6)
    np.testing.assert_allclose(mean,clean[:,:,1:].mean(axis=(0,2)),rtol=1e-5)
    np.testing.assert_allclose(scale,clean[:,:,1:].std(axis=(0,2)),rtol=1e-5)


def test_restoration_unclipped_and_near_zero():
    assert compute_restoration_ratio(.9,.8,.4)==pytest.approx(1.25)
    assert compute_restoration_ratio(.2,.8,.4)==pytest.approx(-.5)
    assert compute_restoration_ratio(.6,.4,.8)==pytest.approx(.5)
    assert np.isnan(compute_restoration_ratio(.7,.8,.8+1e-8))
    assert np.isnan(compute_restoration_ratio(np.nan,.8,.4))
    np.testing.assert_allclose(_holm([.01,.04,.03,np.nan])[:3],[.03,.06,.06])


def test_patient_paired_statistics_and_sample_sd(tmp_path,config):
    rows=[]
    for patient,delta in [('ID01',.1),('ID02',.2),('ID03',.3)]:
        for decoder in config['decoders']:
            for method in config['methods']:
                value=.5+delta if method=='DAST' else .5
                rows.append(dict(patient=patient,decoder=decoder,method=method,accuracy=value,
                    weighted_f1=value,kappa=value,restoration_accuracy=np.nan,restoration_f1=np.nan,restoration_kappa=np.nan))
    stats=run_statistical_analysis(rows,config,tmp_path)
    assert all(r['n_pairs']==3 for r in stats)
    assert all(r['p_corrected']>=r['p_raw'] for r in stats)
    dast=next(r for r in aggregate_results(rows) if r['method']=='DAST')
    assert dast['accuracy_sd']==pytest.approx(.1)
    for row in rows:
        row['accuracy']=.5
    stats=run_statistical_analysis(rows,config,tmp_path)
    assert all(r['p_raw']==1 for r in stats if r['metric']=='accuracy')
    with pytest.raises(ValueError,match='Duplicate'):
        run_statistical_analysis(rows+[rows[0]],config,tmp_path)


def test_all_cancellers_same_input_and_grid(config):
    config=copy.deepcopy(config)
    t=np.arange(256)/1024
    x=np.stack([np.sin(2*np.pi*(7+c)*t) for c in range(3)]).astype(np.float32)
    triggers=list(range(16,240,16))
    x[:,triggers]+=10
    before=x.copy()
    parameters=dict(stim_rate=64.,first_pulse_sample=16,period_samples=16,trigger_samples=triggers)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        for method in config['methods'][2:]:
            y=apply_artifact_cancellation(x,method,parameters,config,lrr_weights=np.zeros((3,3)))
            assert y.shape==(3,255) and np.isfinite(y).all()
            np.testing.assert_array_equal(x,before)


def test_eraasr_regresses_other_columns_not_self():
    t=np.arange(100)
    artifact=np.sin(t*.2)
    x=np.column_stack([artifact,2*artifact,-3*artifact])
    for omitted in (True,False):
        cleaned=_pca_regression(x,1,1,omitted)
        assert np.linalg.norm(cleaned)<1e-10
    # Singleton has no external predictor: retain its centered waveform.
    np.testing.assert_allclose(_pca_regression(x[:,:1],1,1),x[:,:1]-x[:,:1].mean(axis=0))


def test_generator_seed_isolation_and_reconstruction(config):
    class FakeGenerator:
        def __init__(self,native_fs,high_fs,**kwargs):
            self.up=high_fs//native_fs
        def generate_artifact_clip(self,zeros,*args,**kwargs):
            assert np.all(zeros==0)
            return np.random.normal(size=(zeros.shape[0],zeros.shape[1]*self.up)).astype(np.float32)
    clean=np.ones((3,2048),np.float32)
    np.random.seed(72)
    state=np.random.get_state()
    with patch('experiments.seizure_preservation.cancellation._artifact_class',return_value=FakeGenerator):
        mixed,artifact,params=generate_synthetic_artifact(clean,2.,config,5)
        mixed2,artifact2,params2=generate_synthetic_artifact(clean,2.,config,5)
    after=np.random.get_state()
    assert all(np.array_equal(a,b) for a,b in zip(state,after))
    np.testing.assert_array_equal(mixed,mixed2)
    np.testing.assert_array_equal(mixed,clean+artifact)
    assert params==params2
    assert np.sqrt(np.mean(artifact.astype(float)**2))==pytest.approx(params['artifact_rms_ratio']*2,rel=1e-6)
    assert derived_seed(1,'train')!=derived_seed(1,'heldout_artifact')


@pytest.mark.parametrize('name',['EEGNet','GRU','ChronoNet'])
def test_models_parameters_initialized_and_frozen_evaluation(name,config):
    torch.set_num_threads(1)
    if name=='EEGNet' and not Path(config['eegnet_source']).exists():
        pytest.skip('External reused EEGNet source unavailable')
    model=build_model(name,3,127,config)
    assert not any(isinstance(p,torch.nn.parameter.UninitializedParameter) for p in model.parameters())
    optimizer=torch.optim.Adam(model.parameters())
    registered={id(p) for group in optimizer.param_groups for p in group['params']}
    model.eval().requires_grad_(False)
    before={k:v.clone() for k,v in model.state_dict().items()}
    x=np.random.default_rng(7).normal(size=(4,3,127)).astype(np.float32)
    metrics,probability=evaluate_decoder(model,x,np.array([0,1,0,1]),(np.zeros(3),np.ones(3)),config)
    assert probability.shape==(4,2) and set(metrics)=={'accuracy','weighted_f1','kappa'}
    assert registered=={id(p) for p in model.parameters()}
    for k,v in model.state_dict().items():
        torch.testing.assert_close(v,before[k],rtol=0,atol=0)


def test_training_checkpoint_and_selection_only_clean(tmp_path,config):
    torch.set_num_threads(1)
    config=copy.deepcopy(config)
    config['training'].update(device='cpu',epochs=2,patience=1,batch_size=2,learning_rates=[.001,.0003])
    config['training']['gru'].update(hidden_size=3,layers=1)
    rng=np.random.default_rng(9)
    train=rng.normal(size=(4,2,40)).astype(np.float32)
    val=rng.normal(size=(2,2,40)).astype(np.float32)
    y=np.array([0,1,0,1])
    norm=fit_normalization(train,1,1e-6)
    model=train_gru(train,y,val,np.array([0,1]),norm,config,tmp_path,'ID01')
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    saved=torch.load(tmp_path/'GRU'/'best.pt',weights_only=True)
    selection=json.loads((tmp_path/'GRU'/'selection.json').read_text())
    assert saved['validation_loss']==selection['validation_loss']
    assert (tmp_path/'GRU'/'candidate_0_best.pt').exists()
    assert (tmp_path/'GRU'/'candidate_1_best.pt').exists()


def test_lrr_calibration_and_frozen_adapter(tmp_path, config):
    from experiments.seizure_preservation.cancellation import calibrate_lrr
    from benchmarks.lrr import apply_lrr_batch
    config = copy.deepcopy(config)
    config['cancellation']['lrr']['calibration_windows'] = 2
    seen = []
    def generate(clean, train_rms, cfg, seed):
        assert np.all(clean == 0)
        seen.append(seed)
        artifact = np.random.default_rng(seed).normal(size=clean.shape).astype(np.float32)
        return artifact, artifact, dict(seed=seed, trigger_samples=[10, 30], stim_rate=64.)
    with patch('experiments.seizure_preservation.cancellation.generate_synthetic_artifact', side_effect=generate):
        weights = calibrate_lrr(3, 64, 2., config, 'ID01', tmp_path)
    assert not weights.flags.writeable
    assert seen == [derived_seed(config['seed'], 'ID01', 'lrr_calibration', i) for i in range(2)]
    assert not set(seen) & {derived_seed(config['seed'], 'ID01', 'heldout_artifact', i) for i in range(2)}
    np.testing.assert_array_equal(weights, np.load(tmp_path/'lrr_weights.npy'))
    x = np.random.default_rng(3).normal(size=(3,64)).astype(np.float32)
    before = weights.copy()
    result = apply_artifact_cancellation(x, 'LRR', {'stim_rate':64., 'trigger_samples':[]}, config, weights)
    np.testing.assert_allclose(result, apply_lrr_batch(x,weights)[:,1:], rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(weights,before)
    with pytest.raises(ValueError,match='pretrained W'):
        apply_artifact_cancellation(x, 'LRR', {'stim_rate':64., 'trigger_samples':[]}, config)


def test_patient_window_budget_is_deterministic_and_split_first(tmp_path, config):
    patient = make_patient(tmp_path, config, [(21,23),(65,67),(125,127)])
    config['samples_per_patient'] = 20
    splits = split_patient_data(patient, config)
    total = 0
    import csv
    for split, expected in [('train',12),('validation',4),('test',4)]:
        x,y = extract_windows(patient,splits[split],split,config,tmp_path/'first')
        assert len(x)==expected and set(y)=={0,1}
        total += len(x)
        extract_windows(patient,splits[split],split,config,tmp_path/'second')
        assert (tmp_path/'first'/f'{split}_windows.csv').read_bytes()==(tmp_path/'second'/f'{split}_windows.csv').read_bytes()
        for row in csv.DictReader((tmp_path/'first'/f'{split}_windows.csv').open()):
            assert any(g['start']<=int(row['start'])<int(row['stop'])<=g['stop'] for g in splits[split])
    assert total==20


def test_patient_budget_negative_only_validation(tmp_path,config):
    patient=make_patient(tmp_path,config,[(21,23),(121,123)])
    config['samples_per_patient']=100
    groups=split_patient_data(patient,config)
    x,y=extract_windows(patient,groups['validation'],'validation',config,tmp_path/'out')
    assert len(x)==20 and np.all(y==0)


@pytest.mark.parametrize('limit',[0,5,-1,1.5,True])
def test_invalid_patient_budget(config,limit):
    config['samples_per_patient']=limit
    with pytest.raises(ValueError,match='samples_per_patient'):
        validate_config(config)


def test_load_frozen_checkpoint_rejects_split_and_patient_mismatch(tmp_path,config):
    from types import SimpleNamespace
    from experiments.seizure_preservation.checkpoints import load_patient_decoders
    from experiments.seizure_preservation.common import write_json
    config=copy.deepcopy(config)
    config['decoders']=['GRU']
    config['training']['device']='cpu'
    config['training']['gru'].update(hidden_size=3,layers=1)
    config['load_decoders']=str(tmp_path/'source')
    source=tmp_path/'source'/'ID01'
    destination=tmp_path/'destination'/'ID01'
    manifest=dict(patient='ID01',native_fs=256,target_fs=1024,groups={'test':[1]},seizure_intervals_samples=[[2,3]])
    fingerprint=dict(bytes=100,mtime_ns=2,channels=2,samples=2048,fs=256,annotations_sha256='abc')
    for folder in (source,destination):
        write_json(folder/'splits.json',manifest)
        write_json(folder/'input_fingerprint.json',fingerprint)
    (source/'signals').mkdir()
    (source/'GRU').mkdir()
    np.save(source/'signals/validation_labels.npy',[0,1])
    model=build_model('GRU',2,2047,config)
    saved=dict(model=model.state_dict(),decoder='GRU',channels=2,samples=2047,mean=torch.zeros(2),scale=torch.ones(2),
               epoch=2,validation_loss=.3,config=config,seed=derived_seed(config['seed'],'ID01','GRU','training',0))
    path=source/'GRU'/'best.pt'
    torch.save(saved,path)
    patient=SimpleNamespace(patient='ID01',channels=2)
    models,norm,labels=load_patient_decoders(patient,config,destination)
    assert not models['GRU'].training and not any(p.requires_grad for p in models['GRU'].parameters())
    np.testing.assert_array_equal(norm[1],[1,1])
    np.testing.assert_array_equal(labels,[0,1])
    assert path.read_bytes()==(destination/'GRU'/'best.pt').read_bytes()
    saved['seed']=derived_seed(config['seed'],'ID02','GRU','training',0)
    torch.save(saved,path)
    with pytest.raises(ValueError,match='different patient'):
        load_patient_decoders(patient,config,destination)
    manifest['groups']={'test':[2]}
    write_json(destination/'splits.json',manifest)
    with pytest.raises(ValueError,match='split mismatch'):
        load_patient_decoders(patient,config,destination)
