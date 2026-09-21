"""Sequential SVD/template cancellation and frozen rest evaluation."""
from dataclasses import replace
import unittest
import numpy as np

from data_loading import Dataset
from . import BenchmarkConfig, run_benchmark, svd_template_subtraction
from .methods import window_svd, average_template_subtraction
from .metrics import clean_rest_output


class HybridTests(unittest.TestCase):
    def setUp(self):
        self.config = BenchmarkConfig(method='svd_template_subtraction', pre_ms=0, post_ms=4)
        self.times = [3, 10, 17]
        self.x = np.zeros((1, 3, 24))
        for t in self.times:
            self.x[0, :, t:t+4] = np.array([2, 3, 4])[:, None] + np.array([1, 2, 3])[:, None]*[-3, -1, 1, 3]

    def test_svd_removes_wave_then_template_removes_residual_mean(self):
        original = self.x.copy()
        result = run_benchmark(Dataset('test', self.x, 1000, 20), self.config, self.times)
        np.testing.assert_allclose(result.cleaned[0, :, 3:7], np.tile([[2], [3], [4]], (1, 4)), atol=1e-13)
        np.testing.assert_allclose(result.cleaned[0, :, 10:14], 0, atol=1e-13)
        np.testing.assert_allclose(result.cleaned[0, :, 17:21], 0, atol=1e-13)
        stages = result.diagnostics
        np.testing.assert_allclose(result.artifact, stages['svd'].artifact + stages['template_subtraction'].artifact)
        np.testing.assert_allclose(result.cleaned+result.artifact, self.x)
        np.testing.assert_array_equal(self.x, original)

    def test_fixed_training_template_is_fitted_after_svd(self):
        config = replace(self.config, template_history=None)
        train = self.x*2
        train_before = train.copy()
        result = svd_template_subtraction(self.x, self.times, 1000, config,
                                          training_data=train, training_stim_times=self.times)
        np.testing.assert_allclose(result.diagnostics['template_subtraction'].diagnostics['template'],
                                   np.tile([[4], [6], [8]], (1, 4)), atol=1e-13)
        for t in self.times:
            np.testing.assert_allclose(result.cleaned[0, :, t:t+4], np.tile([[-2], [-3], [-4]], (1, 4)), atol=1e-13)
        np.testing.assert_array_equal(train, train_before)

    def test_ragged_overlap_edges_and_rank_zero(self):
        x = np.random.default_rng(4).normal(size=(2, 3, 30))
        times = [[0, 5, 7, 28], [4, 16]]
        cfg = replace(self.config, pre_ms=1, template_history=2)
        result = svd_template_subtraction(x, times, 1000, cfg)
        expected = average_template_subtraction(window_svd(x, times, 1000, cfg).cleaned, times, 1000, cfg)
        np.testing.assert_array_equal(result.cleaned, expected.cleaned)
        np.testing.assert_array_equal(result.cleaned[1, :, 22:], x[1, :, 22:])
        zero = replace(cfg, rank=0)
        np.testing.assert_array_equal(svd_template_subtraction(x, times, 1000, zero).cleaned,
                                      average_template_subtraction(x, times, 1000, zero).cleaned)

    def test_rest_uses_same_stage_order_without_refitting(self):
        result = svd_template_subtraction(self.x, self.times, 1000, self.config)
        rest = np.random.default_rng(7).normal(size=self.x.shape)
        reference, cleaned = clean_rest_output(rest, self.config.method, result, self.config, 1000)
        _, after_svd = clean_rest_output(rest, 'window_svd', result.diagnostics['svd'], self.config, 1000)
        np.testing.assert_array_equal(reference, rest)
        np.testing.assert_allclose(cleaned, after_svd-result.diagnostics['template_subtraction'].artifact)


if __name__ == '__main__':
    unittest.main()
