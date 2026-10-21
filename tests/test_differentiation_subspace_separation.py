"""Analytic geometry and experimental-unit checks for overlap experiment."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_differentiation_subspace_separation import (
    projection_overlap, estimate, evaluate_pair, summarize, paired_observations,
)


def test_unequal_ranks_retain_all_selected_directions():
    eye = np.eye(4)
    # Truncation to the first neural direction would incorrectly give zero.
    overlap, numerator = projection_overlap(eye[:, 1:2], eye[:, :3])
    assert overlap == pytest.approx(1)
    assert numerator == pytest.approx(1)
    assert projection_overlap(eye[:, 3:], eye[:, :3])[0] == 0
    rotated = eye[:, :2] @ np.array([[0., -1.], [1., 0.]])
    assert projection_overlap(eye[:, :2], rotated)[0] == pytest.approx(1)


def test_energy_rank_and_scale_invariance():
    t = np.arange(100) * 2*np.pi/100
    x = np.array([4*np.sin(t), 2*np.cos(t), .1*np.sin(2*t)])
    u, _ = estimate(x, .95)
    assert u.shape[1] == 2
    scaled, _ = estimate(x*1e80, .95)
    assert projection_overlap(u, scaled)[0] == pytest.approx(1)


def test_differentiation_can_reduce_overlap_without_changing_artifact_direction():
    t = np.arange(1000)
    artifact = np.array([(-1.)**t, np.zeros(len(t))])
    neural = np.array([20*np.sin(2*np.pi*t/1000), np.sin(2*np.pi*t/5)])
    rows, _ = evaluate_pair([artifact], [neural], np.ones(2), .95)
    assert rows[0]['subspace_overlap'] > .999
    assert rows[1]['subspace_overlap'] < .001
    assert rows[0]['artifact_samples'] == 1000
    assert rows[1]['artifact_samples'] == 999


def test_no_derivative_across_disjoint_epoch_boundaries():
    t = np.arange(30)
    x = np.array([np.sin(t), np.cos(t)])
    rows, _ = evaluate_pair([x], [x, x+1000], np.ones(2), .95)
    assert rows[1]['neural_samples'] == 58
    assert rows[1]['subspace_overlap'] == pytest.approx(1)


@pytest.mark.parametrize('x', [np.zeros((2,20)), np.full((2,20),np.nan), np.ones((2,2))])
def test_invalid_segments_rejected(x):
    with pytest.raises(ValueError):
        evaluate_pair([x],[x],np.ones(2),.95)


def make_row(dataset, subject, unit, window, condition, value):
    return dict(dataset=dataset, subject=subject, unit=unit, configuration=unit,
                trial=0, window=window, normalization='none', condition=condition,
                subspace_overlap=value)


def test_statistics_weight_patients_not_window_counts():
    rows = []
    for subject, count, a, b in [('p1',100,.8,.2),('p2',1,.4,.3)]:
        for w in range(count):
            rows.extend([make_row('swec_synthetic',subject,subject,w,'none',a),
                         make_row('swec_synthetic',subject,subject,w,'derivative',b)])
    units, summary, stats = summarize(rows)
    assert len(units) == 4
    assert stats[0]['n_independent'] == 2
    assert summary[0]['mean_overlap'] == pytest.approx(.6)
    assert stats[0]['median_paired_difference'] == pytest.approx(-.35)
    assert stats[0]['p_holm'] == stats[0]['p_value']


def test_pbs_reuse_is_not_independent_replication():
    rows = []
    for subject, count, value in [('p1',10,.8),('p2',1,.2)]:
        for config in ('PBS0','PBS1'):
            for w in range(count):
                for condition in ('none','derivative'):
                    rows.append(make_row('swec_pbs',subject,config,w,condition,value))
    units, summary, stats = summarize(rows)
    assert stats[0]['n_independent'] == 2
    assert stats[0]['p_value'] == 1
    assert all(r['subspace_overlap'] == pytest.approx(.5) for r in units)


def test_single_recording_and_missing_pair():
    rows = [make_row('ERAASR','rec','rec',0,c,.5) for c in ('none','derivative')]
    assert np.isnan(summarize(rows)[2][0]['p_value'])
    with pytest.raises(AssertionError,match='Unmatched'):
        paired_observations(rows[:1])
    with pytest.raises(AssertionError,match='Duplicate'):
        paired_observations(rows+rows)
