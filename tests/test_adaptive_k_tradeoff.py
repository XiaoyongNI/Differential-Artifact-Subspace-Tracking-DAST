"""The trade-off decomposition reproduces run_pipeline exactly."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest

from data_loading import Dataset
from pipeline import PipelineConfig, run_pipeline
from tests.plot_adaptive_k_tradeoff import tradeoff_energies, metrics_from_energies

K_MAX, BETA, THRESHOLD, PERIOD = 4, .99, .1, 50


def mixture(seed=0, channels=8, n_time=1200):
    rng = np.random.default_rng(seed)
    neural = np.cumsum(rng.standard_normal((channels, n_time)), axis=1) * .1
    spikes = (rng.random((2, n_time)) < .02) * rng.standard_normal((2, n_time)) * 20
    artifact = rng.standard_normal((channels, 2)) @ spikes
    return neural, artifact


def pipeline_output_error(neural, artifact, rank, adaptive):
    ds = Dataset(name='toy', X=(neural + artifact)[None], y_clean=neural[None], fs=1000., stim_rate=50.)
    config = PipelineConfig(n_test=0, preprocessing='none', rank=rank, beta=BETA,
                            adaptive_k=adaptive, adaptive_k_threshold=THRESHOLD,
                            adaptive_k_check_period=PERIOD, compute_ideal_u=False)
    result = run_pipeline(ds, config)
    return float(np.sum((result.stages['cleaned'][0] - neural[:, 1:]) ** 2)), result


@pytest.mark.parametrize('k', [0, 1, 2, K_MAX])
def test_fixed_k_matches_pipeline_including_rank_prefix(k):
    neural, artifact = mixture()
    energy = tradeoff_energies(neural, artifact, K_MAX, BETA, THRESHOLD, PERIOD, skip=0)
    if k == 0:
        expected = float(np.sum(artifact[:, 1:] ** 2))
    else:
        expected, _ = pipeline_output_error(neural, artifact, rank=k, adaptive=False)
    np.testing.assert_allclose(energy['output_error'][k], expected, rtol=1e-9)


def test_adaptive_matches_pipeline():
    neural, artifact = mixture(seed=1)
    energy = tradeoff_energies(neural, artifact, K_MAX, BETA, THRESHOLD, PERIOD, skip=0)
    expected, result = pipeline_output_error(neural, artifact, rank=K_MAX, adaptive=True)
    np.testing.assert_array_equal(energy['chosen_k'], result.diagnostics['chosen_k'][0])
    np.testing.assert_allclose(energy['output_error'][-1], expected, rtol=1e-9)


def test_k0_has_no_suppression_and_no_distortion():
    neural, artifact = mixture(seed=2)
    metrics = metrics_from_energies(tradeoff_energies(neural, artifact, K_MAX, BETA, THRESHOLD, PERIOD, skip=0))
    assert metrics['artifact_suppression_db'][0] == pytest.approx(0)
    assert metrics['neural_distortion_pct'][0] == 0


def test_orthonormal_projection_is_nested_and_monotone():
    """Under nested orthogonal projections, residual artifact cannot grow with k."""
    neural, artifact = mixture(seed=3)
    energy = tradeoff_energies(neural, artifact, K_MAX, BETA, THRESHOLD, PERIOD, skip=0)
    fixed = energy['artifact_residual'][:K_MAX + 1]
    assert np.all(np.diff(fixed) <= 1e-9 * fixed[0])
    # Neural error is the removed neural energy, which can only grow with k.
    assert np.all(np.diff(energy['neural_error'][:K_MAX + 1]) >= -1e-9)


def test_stability_gap_matches_pipeline():
    neural, artifact = mixture(seed=4)
    energy = tradeoff_energies(neural, artifact, K_MAX, BETA, THRESHOLD, PERIOD, skip=0, rule='stability_gap')
    ds = Dataset(name='toy', X=(neural + artifact)[None], y_clean=neural[None], fs=1000., stim_rate=50.)
    config = PipelineConfig(n_test=0, preprocessing='none', rank=K_MAX, beta=BETA, adaptive_k=True,
                            adaptive_k_rule='stability_gap', adaptive_k_check_period=PERIOD,
                            compute_ideal_u=False)
    result = run_pipeline(ds, config)
    np.testing.assert_array_equal(energy['chosen_k'], result.diagnostics['chosen_k'][0])
    expected = float(np.sum((result.stages['cleaned'][0] - neural[:, 1:]) ** 2))
    np.testing.assert_allclose(energy['output_error'][-1], expected, rtol=1e-9)


def test_stability_gap_finds_largest_log_jump():
    from algorithms import choose_k_from_stability_gap
    assert choose_k_from_stability_gap([1e-4, 1.1, 1.3, 1.5]) == 1
    assert choose_k_from_stability_gap([.04, .24, .48, .14, 1.08, 1.3]) == 4
    assert choose_k_from_stability_gap([0., 1., 1.2]) == 1
    assert choose_k_from_stability_gap([.5]) == 1


@pytest.mark.parametrize('rule', ['stability_gap', 'relative'])
def test_warm_start_matches_pipeline(rule):
    neural, artifact = mixture(seed=5)
    W = 64
    energy = tradeoff_energies(neural, artifact, K_MAX, BETA, THRESHOLD, PERIOD, skip=0, rule=rule, warm_start=W)
    ds = Dataset(name='toy', X=(neural + artifact)[None], y_clean=neural[None], fs=1000., stim_rate=50.)
    config = PipelineConfig(n_test=0, preprocessing='none', rank=K_MAX, beta=BETA, adaptive_k=True,
                            adaptive_k_rule=rule, adaptive_k_check_period=PERIOD, warm_start_samples=W,
                            compute_ideal_u=False)
    result = run_pipeline(ds, config)
    np.testing.assert_array_equal(energy['chosen_k'], result.diagnostics['chosen_k'][0])
    expected = float(np.sum((result.stages['cleaned'][0] - neural[:, 1:]) ** 2))
    np.testing.assert_allclose(energy['output_error'][-1], expected, rtol=1e-9)


def test_warm_start_is_causal():
    """Changing samples after time t0 must not change any output before t0."""
    neural, artifact = mixture(seed=6)
    W, t0 = 64, 300
    config = PipelineConfig(n_test=0, preprocessing='none', rank=K_MAX, beta=BETA, adaptive_k=True,
                            adaptive_k_rule='stability_gap', adaptive_k_check_period=PERIOD,
                            warm_start_samples=W, compute_ideal_u=False)
    outputs = []
    for seed in (0, 1):
        x = neural + artifact
        x[:, t0:] += np.random.default_rng(seed).standard_normal((x.shape[0], x.shape[1] - t0)) * 50
        ds = Dataset(name='toy', X=x[None], y_clean=neural[None], fs=1000., stim_rate=50.)
        outputs.append(run_pipeline(ds, config).stages['cleaned'][0])
    # cleaned[:, i] corresponds to input sample i + 1 (first difference)
    np.testing.assert_array_equal(outputs[0][:, :t0 - 1], outputs[1][:, :t0 - 1])
    assert not np.allclose(outputs[0][:, t0:], outputs[1][:, t0:])


def test_warm_start_initializes_dominant_subspace():
    from algorithms import PASTd
    rng = np.random.default_rng(7)
    basis = np.linalg.qr(rng.standard_normal((8, 2)))[0]
    X = basis @ rng.standard_normal((2, 100)) * 10 + rng.standard_normal((8, 100)) * .01
    tracker = PASTd(8, 3, beta=BETA)
    tracker.warm_start(X)
    overlap = np.sum((basis.T @ tracker.U[:, :2]) ** 2)
    assert overlap > 2 - 1e-6
    np.testing.assert_allclose(tracker.d, np.linalg.svd(X, compute_uv=False)[:3] ** 2)
