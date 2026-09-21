"""Checks for rest correlation and the existing clean-energy convention."""
from dataclasses import replace
import unittest
import numpy as np

from data_loading import Dataset
from pipeline import PipelineConfig, run_pipeline
from .methods import BenchmarkConfig, window_svd
from .metrics import (clean_energy_loss, clean_energy_loss_from_output, summarize_clean_energy,
                      rest_correlations, clean_rest_output)


class RestMetricTests(unittest.TestCase):
    def test_clean_projection_matches_existing_function_exactly(self):
        from test_CleanRemoval_and_Covariance import clean_energy_loss as existing
        rng = np.random.default_rng(4)
        clean = rng.normal(size=(3, 4, 40))
        u = rng.normal(size=(3, 4))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        expected_rows, expected_cleaned = existing(clean, u)
        rows, cleaned = clean_energy_loss(clean, u)
        self.assertEqual(rows, expected_rows)
        np.testing.assert_array_equal(cleaned, expected_cleaned)

    def test_trial_fractions_are_averaged_without_energy_weighting(self):
        clean = np.array([[[1.]], [[100.]]])
        cleaned = np.array([[[0.]], [[100.]]])
        rows, _ = clean_energy_loss_from_output(clean, cleaned)
        self.assertEqual(rows[0]['energy_loss_fraction'], 1.)
        self.assertEqual(rows[1]['energy_loss_fraction'], 0.)
        self.assertEqual(summarize_clean_energy(rows)['energy_loss_fraction'], .5)

    def test_covariance_loss_is_correction_power_not_net_decrease(self):
        clean = np.ones((1, 2, 4))
        rows, _ = clean_energy_loss_from_output(clean, .5*clean)
        self.assertEqual(rows[0]['energy_loss_fraction'], .25)
        self.assertAlmostEqual(rows[0]['energy_loss_db'], 10*np.log10(.25))
        rows, _ = clean_energy_loss_from_output(np.zeros_like(clean), np.zeros_like(clean))
        self.assertTrue(np.isnan(rows[0]['energy_loss_fraction']))

    def test_time_and_spectral_correlations_have_distinct_meanings(self):
        t = np.arange(1000)/1000
        baseline = np.sin(2*np.pi*100*t)[None, None]
        shifted = np.cos(2*np.pi*100*t)[None, None]
        metrics, time, freq = rest_correlations(shifted, baseline[:, :, :800], 1000)
        self.assertAlmostEqual(metrics['rest_correlation_time'], 0, places=12)
        self.assertAlmostEqual(metrics['rest_correlation_freq'], 1, places=12)
        self.assertEqual(time.shape, (1, 1))
        self.assertEqual(freq.shape, (1, 1))

    def test_constant_or_short_rest_has_undefined_correlation(self):
        metrics, _, _ = rest_correlations(np.ones((1, 2, 40)), np.ones((1, 2, 30)), 1000)
        self.assertTrue(np.isnan(metrics['rest_correlation_time']))
        self.assertTrue(np.isnan(metrics['rest_correlation_freq']))
        metrics, _, _ = rest_correlations(np.ones((1, 2, 40)), np.ones((1, 2, 2)), 1000)
        self.assertTrue(np.isnan(metrics['rest_correlation_freq']))

    def test_past_rest_uses_learned_u_without_dc_offset_or_retraining(self):
        rng = np.random.default_rng(5)
        dataset = Dataset('test', rng.normal(size=(1, 3, 50))+100., 1000, 50)
        config = PipelineConfig(n_test=0, preprocessing='none', use_derivative_for_tracking=False,
                                normalization='per_trial')
        result = run_pipeline(dataset, config)
        before = result.diagnostics['u_trace'].copy()
        rest = rng.normal(size=(1, 3, 40))
        reference, cleaned = clean_rest_output(rest, 'past', result, config, 1000)
        u = result.diagnostics['u_trace'][:, -1, :, 0]
        _, expected = clean_energy_loss(rest, u)
        np.testing.assert_allclose(cleaned, expected, atol=1e-14)
        np.testing.assert_array_equal(before, result.diagnostics['u_trace'])

    def test_covariance_rest_reuses_stimulation_rx(self):
        rng = np.random.default_rng(6)
        dataset = Dataset('test', rng.normal(size=(1, 3, 50)), 1000, 50,
                          baseline=rng.normal(size=(1, 3, 40)))
        config = PipelineConfig(n_test=0, preprocessing='none', removal_method='covariance')
        result = run_pipeline(dataset, config)
        before = result.diagnostics['mixed_covariance_final'].copy()
        _, cleaned = clean_rest_output(dataset.baseline, 'past_covariance_rest', result, config, 1000)
        np.testing.assert_array_equal(before, result.diagnostics['mixed_covariance_final'])
        self.assertTrue(np.isfinite(cleaned).all())

    def test_svd_rest_uses_stimulation_basis_without_refitting(self):
        stimulus = np.zeros((1, 2, 10))
        stimulus[0, 0, 2:6] = [-3, -1, 1, 3]
        config = BenchmarkConfig(method='window_svd', pre_ms=0, post_ms=4)
        result = window_svd(stimulus, [2], 1000, config)
        rest = np.zeros_like(stimulus)
        rest[0, 1, 2:6] = [-3, -1, 1, 3]  # orthogonal to the learned artifact
        _, cleaned = clean_rest_output(rest, 'window_svd', result, config, 1000, [[2]])
        np.testing.assert_allclose(cleaned, rest, atol=1e-14)


if __name__ == '__main__':
    unittest.main()
