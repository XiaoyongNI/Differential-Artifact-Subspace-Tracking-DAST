"""Analytic energy-ratio checks and independent-unit aggregation checks."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from plot_differentiation_energy_ratio import energy_comparison, summarize


def test_proportional_components_have_no_gain():
    n = np.array([[1.,2.,4.,8.],[3.,1.,2.,5.]])
    rows,pair = energy_comparison(3*n,n)
    for r in rows:
        assert r['energy_ratio']==pytest.approx(9)
        assert r['energy_ratio_db']==pytest.approx(10*np.log10(9))
    assert pair['gain']==pytest.approx(1)
    assert pair['gain_db']==pytest.approx(0,abs=1e-12)
    assert [r['n_samples'] for r in rows]==[4,3]


def test_known_high_frequency_artifact_gain():
    a = np.array([[1.,-1.,1.,-1.]])
    n = np.array([[0.,1.,2.,3.]])
    rows,pair = energy_comparison(a,n)
    assert rows[0]['artifact_energy']==4
    assert rows[0]['neural_energy']==14
    assert rows[1]['artifact_energy']==12
    assert rows[1]['neural_energy']==3
    assert pair['ratio_none']==pytest.approx(2/7)
    assert pair['ratio_derivative']==pytest.approx(4)
    assert pair['gain']==pytest.approx(14)
    assert pair['gain_db']==pytest.approx(10*np.log10(14))


def test_scaling_changes_ratios_but_not_gain():
    a = np.array([[1.,-1.,1.,-1.]])
    n = np.array([[0.,1.,2.,3.]])
    _,original = energy_comparison(a,n)
    _,scaled = energy_comparison(10*a,n)
    assert scaled['ratio_none']==pytest.approx(100*original['ratio_none'])
    assert scaled['ratio_derivative']==pytest.approx(100*original['ratio_derivative'])
    assert scaled['gain']==pytest.approx(original['gain'])


def test_slow_artifact_is_allowed_to_lose_relative_energy():
    a = np.array([[0.,1.,2.,3.]])
    n = np.array([[1.,-1.,1.,-1.]])
    _,pair = energy_comparison(a,n)
    assert pair['gain']==pytest.approx(1/14)
    assert pair['gain_db']<0


def test_default_preserves_dc_and_demeaning_is_explicit():
    x = np.array([[1.,3.,2.,5.,4.]])
    _,raw = energy_comparison(x+100,x)
    _,centered = energy_comparison(x+100,x,demean=True)
    assert raw['ratio_none']>100
    assert raw['ratio_derivative']==pytest.approx(1)
    assert centered['ratio_none']==pytest.approx(1)
    assert centered['ratio_derivative']==pytest.approx(1)


@pytest.mark.parametrize('a,n',[
    (np.zeros((2,10)),np.ones((2,10))),
    (np.ones((2,10)),np.arange(20).reshape(2,10)),
    (np.full((2,10),np.nan),np.ones((2,10))),
    (np.ones((2,2)),np.ones((2,2))),
    (np.ones((2,10)),np.ones((3,10))),
])
def test_undefined_comparisons_exclude_both_conditions(a,n):
    with pytest.raises(ValueError):
        energy_comparison(a,n)


def row(subject,unit,window,condition,db,dataset='swec_synthetic'):
    return dict(dataset=dataset,subject=subject,configuration=unit,unit=unit,
                trial=0,window=window,condition=condition,energy_ratio_db=db)


def test_patient_aggregation_is_equal_weight_not_window_weighted():
    rows=[]
    for subject,count,before,after in [('p1',20,0,10),('p2',1,20,30)]:
        for w in range(count):
            rows.extend([row(subject,subject,w,'none',before),row(subject,subject,w,'derivative',after)])
    units,summary=summarize(rows)
    assert len(units)==2
    assert [u['ratio_none'] for u in units]==pytest.approx([1,100])
    assert summary[0]['ratio_none_median']==pytest.approx(50.5)
    assert summary[0]['gain_median']==pytest.approx(10)
    assert summary[0]['n_independent']==2
    assert summary[0]['n_paired_windows']==21


def test_pbs_averages_subject_db_means_before_configuration_inference():
    rows=[]
    for subject,count,db in [('p1',20,0),('p2',1,20)]:
        for w in range(count):
            for unit in ['config0','config1']:
                for condition in ['none','derivative']:
                    rows.append(row(subject,unit,w,condition,db,'swec_pbs'))
    units,summary=summarize(rows)
    assert all(u['ratio_none']==pytest.approx(10) for u in units)
    assert summary[0]['n_independent']==2
    assert summary[0]['p_value']==1
    assert summary[0]['gain_median']==1


def test_holm_family_and_missing_pair():
    rows=[]
    for subject in ['p1','p2','p3']:
        for dataset in ['swec_synthetic','swec_pbs']:
            rows.extend([row(subject,subject,0,'none',0,dataset),row(subject,subject,0,'derivative',10,dataset)])
    _,summary=summarize(rows)
    assert all(r['p_value']==pytest.approx(.25) for r in summary)
    assert all(r['p_holm']==pytest.approx(.5) for r in summary)
    with pytest.raises(ValueError,match='Unmatched'):
        summarize(rows[:-1])
    with pytest.raises(ValueError,match='Duplicate'):
        summarize(rows+rows[:1])


def test_single_unit_has_no_inferential_pvalue():
    _,summary=summarize([row('p1','p1',0,c,0) for c in ['none','derivative']])
    assert np.isnan(summary[0]['p_value'])
