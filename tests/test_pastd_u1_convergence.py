"""Online update alignment, no future leakage, reference geometry and censoring."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from plot_pastd_u1_convergence import (
    track_pair,artifact_direction,first_sustained_crossing,trace_metrics,
    aggregate_units,statistical_summary,CONDITIONS,METRICS,
)
from algorithms import PASTd


def components(length=50):
    t=np.arange(length)
    a=np.array([1.,2.,3.])[:,None]*((-1.)**t)[None,:]
    n=np.random.default_rng(7).normal(size=a.shape)*.1
    return a,n


def test_first_updates_match_existing_pastd_and_same_timestamps():
    a,n=components()
    result=track_pair(a,n)
    x=a+n
    assert result['vectors'].shape==(2,50,3)
    np.testing.assert_array_equal(result['vectors'][:,0],[[1,0,0],[1,0,0]])
    for condition,sample in enumerate([x[:,1],x[:,1]-x[:,0]]):
        tracker=PASTd(3,1)
        tracker.update(sample,track_delta=False)
        np.testing.assert_allclose(result['vectors'][condition,1],tracker.U[:,0])


def test_future_samples_do_not_change_past_estimates():
    a,n=components(100)
    short=track_pair(a[:,:30],n[:,:30],initialization='random',seed=3)
    long=track_pair(a,n,initialization='random',seed=3)
    np.testing.assert_array_equal(short['vectors'],long['vectors'][:,:30])


def test_similarity_is_absolute_cosine_and_artifact_only_converges():
    a,_=components(100)
    result=track_pair(a,np.zeros_like(a))
    expected=np.abs(result['vectors']@(-result['reference']))
    np.testing.assert_allclose(result['primary'],expected,atol=1e-14)
    assert np.all(result['primary'][:,-1]>.999)
    assert result['reference_cosine']==pytest.approx(1)


def test_same_seed_repeats_initialization_and_different_seed_changes_it():
    a,n=components()
    first=track_pair(a,n,initialization='random',seed=4)
    second=track_pair(a,n,initialization='random',seed=4)
    third=track_pair(a,n,initialization='random',seed=5)
    np.testing.assert_array_equal(first['vectors'],second['vectors'])
    assert not np.allclose(first['vectors'][0,0],third['vectors'][0,0])


def test_shared_reference_is_not_silently_changed_for_derivative():
    t=np.arange(200)
    a=np.array([20*np.sin(2*np.pi*t/200),(-1.)**t])
    result=track_pair(a,np.zeros_like(a))
    assert result['reference_cosine']<.01
    assert result['primary'][1,-1]<.1
    assert result['matched'][1,-1]>.99


def test_constant_artifact_does_not_disappear_from_primary_evaluation():
    a=np.array([1.,2.])[:,None]*np.ones((1,20))
    n=np.zeros_like(a)
    result=track_pair(a,n)
    assert np.isfinite(result['primary']).all()
    assert np.isnan(result['matched'][1]).all()
    assert result['reference_status']!='ok'


def test_ground_truth_tie_and_invalid_data_are_rejected():
    with pytest.raises(ValueError,match='spectral tie'):
        artifact_direction(np.eye(3))
    a,n=components()
    a[0,0]=np.nan
    with pytest.raises(ValueError,match='Nonfinite'):
        track_pair(a,n)


def test_sustained_threshold_requires_confirmation_not_instantaneous_hit():
    curve=np.array([.2,.96,.7,.96,.97,.98,.6])
    assert first_sustained_crossing(curve,.95,3)==(3,5)
    assert first_sustained_crossing(curve,.95,4) is None
    assert first_sustained_crossing(curve,.99,1) is None


def test_censored_delay_is_not_reported_as_convergence():
    curve=np.full(11,.5)
    m=trace_metrics(curve,1000,10,.95,2)
    assert m['early_mean_similarity']==.5
    assert not m['converged_by_horizon']
    assert not m['converged_full_clip']
    assert m['restricted_convergence_ms']==10
    assert np.isnan(m['convergence_confirmed_ms'])
    assert m['convergence_confirmed_sample']==-1
    m=trace_metrics(np.ones(11),1000,10,.95,2)
    assert m['converged_by_horizon']
    assert m['convergence_confirmed_ms']==2


def test_event_after_horizon_remains_censored_for_early_metric():
    curve=np.array([.1]*10+[.99]*10)
    m=trace_metrics(curve,1000,5,.95,2)
    assert m['converged_full_clip']
    assert not m['converged_by_horizon']
    assert m['restricted_convergence_ms']==5
    assert m['convergence_confirmed_ms']==12


def test_aggregation_uses_patients_and_pbs_configurations_not_trial_counts():
    rows,index,traces=[],[],{}
    for dataset in ['swec_synthetic','swec_pbs']:
        for subject,count,value in [('p1',10,.2),('p2',1,.8)]:
            unit=subject if dataset=='swec_synthetic' else 'PBS0'
            for trial in range(count):
                key=f'{dataset}_{subject}_{trial}'
                meta=dict(dataset=dataset,subject=subject,unit=unit,configuration=unit,trial=trial)
                index.append(dict(**meta,trace_key=key,n_samples=4,fs=1000))
                for kind in ['primary','condition_matched']:
                    traces[key+'_'+kind]=np.full((2,4),value)
                for condition in CONDITIONS:
                    rows.append(dict(**meta,condition=condition,**{m:value for m in METRICS}))
    units,curves,_=aggregate_units(SimpleNamespace(rows=rows,index=index,traces=traces))
    pbs=[r for r in units if r['dataset']=='swec_pbs']
    assert len(pbs)==2
    assert pbs[0]['early_mean_similarity']==pytest.approx(.5)
    assert curves[('swec_pbs','primary')][2].shape==(1,2,4)
    np.testing.assert_allclose(curves[('swec_pbs','primary')][2],.5)
    _,stats=statistical_summary(units)
    assert all(r['n_independent']==2 and r['p_value']==1 for r in stats if r['dataset']=='swec_synthetic')
    assert all(np.isnan(r['p_value']) for r in stats if r['dataset']=='swec_pbs')
    with pytest.raises(ValueError,match='Unmatched'):
        aggregate_units(SimpleNamespace(rows=rows[:-1],index=index,traces=traces))
