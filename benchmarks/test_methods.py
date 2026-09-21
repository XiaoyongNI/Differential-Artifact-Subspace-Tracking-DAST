"""Numerical and boundary checks for benchmark contracts."""
import unittest
import warnings
import numpy as np
from .methods import (BenchmarkConfig, linear_interpolation, average_template_subtraction,
                      window_svd, window_ica, low_rank_tv, pulse_times)


class BenchmarkTests(unittest.TestCase):
    def test_interpolation_merges_overlap_and_preserves_ramp(self):
        clean = np.broadcast_to(np.arange(20.), (1, 2, 20)).copy()
        mixed = clean.copy()
        mixed[:, :, 5:10] += 100
        result = linear_interpolation(mixed, [5, 7], 1000, BenchmarkConfig(pre_ms=0, post_ms=3))
        np.testing.assert_allclose(result.cleaned, clean)
        np.testing.assert_array_equal(mixed, result.cleaned+result.artifact)

    def test_interpolation_leaves_unanchored_edges(self):
        mixed = np.arange(10.).reshape(1, 1, 10)
        result = linear_interpolation(mixed, [0, 9], 1000, BenchmarkConfig(pre_ms=0, post_ms=1))
        np.testing.assert_array_equal(result.cleaned, mixed)
        self.assertEqual(result.diagnostics['skipped_edge_windows'], 2)

    def test_backward_template_uses_previous_raw_pulses(self):
        x = np.zeros((1, 1, 20))
        for t, value in zip([2, 7, 12], [3., 5., 9.]):
            x[:, :, t:t+2] = value
        result = average_template_subtraction(x, [2, 7, 12], 1000,
                                     BenchmarkConfig(pre_ms=0, post_ms=2, template_history=2))
        np.testing.assert_allclose(result.cleaned[0, 0, [2, 7, 12]], [3, 2, 5])
        np.testing.assert_array_equal(x, result.cleaned+result.artifact)

    def test_fixed_template_uses_separate_training(self):
        train = np.zeros((1, 2, 10))
        train[:, :, 2:4] = 7
        test = np.zeros((1, 2, 10))
        test[:, :, 5:7] = 9
        result = average_template_subtraction(test, [5], 1000,
                                     BenchmarkConfig(pre_ms=0, post_ms=2, template_history=None),
                                     training_data=train, training_stim_times=[2])
        np.testing.assert_allclose(result.cleaned[:, :, 5:7], 2)
        np.testing.assert_array_equal(result.cleaned[:, :, :5], test[:, :, :5])

    def test_svd_removes_rank_one_preserving_mean_and_outside(self):
        x = np.ones((1, 3, 20))*4
        wave = np.array([-3, -1, 1, 3.])
        x[0, :, 5:9] += np.array([1, 2, 3])[:, None]*wave
        result = window_svd(x, [5], 1000, BenchmarkConfig(pre_ms=0, post_ms=4))
        np.testing.assert_allclose(result.cleaned, 4, atol=1e-13)
        np.testing.assert_array_equal(result.cleaned[:, :, :5], x[:, :, :5])

    def test_ica_selected_component_and_subspace_preservation(self):
        rng = np.random.default_rng(0)
        neural = rng.uniform(-1, 1, 2000)
        artifact = rng.laplace(0, 10, 2000)
        x = np.stack([neural+artifact, neural-artifact, 2*neural])[None]
        config = BenchmarkConfig(pre_ms=0, post_ms=2000, rank=1, ica_components=2,
                                 ica_selection='energy', max_iters=1000, tol=1e-6)
        result = window_ica(x, [0], 1000, config)
        expected = np.stack([neural, neural, 2*neural])[None]
        self.assertLess(np.mean((result.cleaned-expected)**2), np.mean((x-expected)**2)/100)
        np.testing.assert_allclose(result.cleaned+result.artifact, x, atol=1e-14)

    def test_low_rank_tv_shape_finite_reconstruction_and_no_mutation(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(2, 3, 40))
        original = x.copy()
        result = low_rank_tv(x, config=BenchmarkConfig(max_iters=5))
        self.assertEqual(result.cleaned.shape, x.shape)
        self.assertTrue(np.isfinite(result.cleaned).all())
        np.testing.assert_allclose(result.cleaned+result.artifact, x, atol=1e-14)
        np.testing.assert_array_equal(x, original)

    def test_invalid_and_ragged_markers(self):
        with self.assertRaises(ValueError):
            pulse_times([1.5], 1, 20)
        with self.assertRaises(ValueError):
            pulse_times([20], 1, 20)
        times = pulse_times([[2, 4], []], 2, 20)
        np.testing.assert_array_equal(times[0], [2, 4])
        self.assertEqual(times[1].size, 0)


if __name__ == '__main__':
    unittest.main()
