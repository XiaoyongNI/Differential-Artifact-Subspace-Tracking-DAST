"""Numerical checks for covariance removal without changing tracker behavior."""
from dataclasses import replace
import unittest
import warnings
import numpy as np

from algorithms import (covariance_aware_subspace_suppression, normalize_channels,
                        project_and_remove, apply_lpf_3d, apply_bpf_3d)
from covariance import interpulse_mask, prepare_neural_covariance, resolve_stim_times
from data_loading import Dataset
from pipeline import PipelineConfig, run_pipeline


class SuppressionTests(unittest.TestCase):
    def test_noncommuting_covariance_matches_formula(self):
        Rn = np.array([[2., .6], [.6, 1.]])
        Rx = Rn + np.array([[4., 0], [0, 0]])
        U = np.array([[1.], [0.]])
        x = np.array([[1., 2.], [3., 4.]])
        A = Rn + 2 * U @ np.diag([4.]) @ U.T + .01*np.eye(2)
        expected = np.linalg.solve(A.T, Rn.T).T @ x
        output = covariance_aware_subspace_suppression(x, U, Rn, Rx, 2., .01)
        np.testing.assert_allclose(output, expected, atol=1e-14)

    def test_neural_overlap_does_not_count_as_artifact(self):
        Rn = np.diag([10., 2.])
        U = np.array([[1.], [0.]])
        x = np.array([4., 1.])
        expected = Rn @ np.linalg.solve(Rn + 1e-6*np.eye(2), x)
        for Rx in (Rn, .5*Rn):
            np.testing.assert_allclose(covariance_aware_subspace_suppression(x, U, Rn, Rx), expected)
        self.assertAlmostEqual(project_and_remove(x, U)[0], 0)

    def test_current_basis_and_empty_or_changed_k(self):
        Rn, Rx, x = np.eye(3), np.diag([5., 3., 1.]), np.ones(3)
        for U in (np.eye(3)[:, :0], np.eye(3)[:, :1], np.eye(3)[:, :2], np.eye(3)[:, [1, 0]]):
            power = np.maximum(np.diag(U.T @ (Rx-Rn) @ U), 0)
            expected = np.linalg.solve(Rn+(U*power)@U.T+1e-6*np.eye(3), x)
            np.testing.assert_allclose(covariance_aware_subspace_suppression(x, U, Rn, Rx), expected)

    def test_singular_covariance_and_roundoff(self):
        Rn = np.diag([1., 0.])
        x = np.ones(2)
        out = covariance_aware_subspace_suppression(x, np.eye(2), Rn, np.diag([2., -1e-13]))
        self.assertTrue(np.isfinite(out).all())
        self.assertEqual(out[1], 0.)

    def test_invalid_dimensions_psd_and_parameters(self):
        x, U, R = np.ones(2), np.eye(2), np.eye(2)
        cases = [(x, np.eye(3), R, R, 1, 1e-6),
                 (x, U, np.eye(3), R, 1, 1e-6),
                 (x, U, np.diag([-1., 1.]), R, 1, 1e-6),
                 (x, U, R, R, -1, 1e-6), (x, U, R, R, 1, 0),
                 (np.array([np.nan, 1]), U, R, R, 1, 1e-6)]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                covariance_aware_subspace_suppression(*case)


class CovarianceSourceTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(10)
        self.x = rng.normal(size=(2, 3, 80)) + np.array([2., -1., .5])[None, :, None]
        self.rest = rng.normal(size=(2, 3, 60)) + np.array([1., 3., -2.])[None, :, None]
        self.dataset = Dataset('test', self.x, 1000., 50., baseline=self.rest)

    def test_rest_matches_signal_normalization_including_streaming(self):
        pooled = self.rest.transpose(1, 0, 2).reshape(3, -1)
        for mode in ('none', 'global_per_channel', 'per_trial', 'streaming_causal'):
            config = PipelineConfig(n_test=0, preprocessing='none', removal_method='covariance', normalization=mode)
            norm, stats = normalize_channels(self.x, mode)
            source = prepare_neural_covariance(self.dataset, config, self.x, stats)
            tr, t = 1, 25
            if mode == 'none':
                samples = pooled
            elif mode == 'streaming_causal':
                samples = (pooled-stats.debug['mu_hist'][tr, :, t, None])/stats.debug['scale_hist'][tr, :, t, None]
            elif mode == 'per_trial':
                samples = (pooled-stats.mean[tr, 0, 0])/stats.scale[tr, 0, 0]
            else:
                samples = (pooled-stats.mean[0, :, 0, None])/stats.scale[0, :, 0, None]
            np.testing.assert_allclose(source.at(tr, t), samples @ samples.T/samples.shape[1], atol=1e-13)

    def test_rest_has_identical_filtering_but_no_derivative(self):
        for method in ('lpf', 'bpf'):
            config = PipelineConfig(n_test=0, preprocessing=method, removal_method='covariance',
                                    lpf_cutoff_hz=100, bpf_low_hz=30, bpf_high_hz=200)
            filtered = (apply_lpf_3d(self.rest, 100, 1000) if method == 'lpf'
                        else apply_bpf_3d(self.rest, 30, 200, 1000))
            pooled = filtered.transpose(1, 0, 2).reshape(3, -1)
            _, stats = normalize_channels(self.x, 'none')
            source = prepare_neural_covariance(self.dataset, config, self.x, stats)
            np.testing.assert_allclose(source.at(0, 0), pooled@pooled.T/pooled.shape[1], atol=1e-14)

    def test_middle_gap_masks_noninteger_periods_and_overlaps(self):
        mask = interpulse_mask([10, 30, 50], 80, 1000, pre_ms=2, post_ms=4, middle_fraction=.5)
        expected = np.zeros(80, dtype=bool)
        expected[18:24] = True
        expected[38:44] = True
        np.testing.assert_array_equal(mask, expected)
        overlap = interpulse_mask([10, 12, 30], 80, 1000, pre_ms=2, post_ms=4, middle_fraction=.5)
        self.assertFalse(overlap[10:16].any())
        config = PipelineConfig(first_pulse_sample=2)
        dataset = replace(self.dataset, stim_rate=33.)
        times = resolve_stim_times(dataset, config)[0]
        np.testing.assert_array_equal(times, np.rint(2+np.arange(3)*1000/33).astype(int))

    def test_interpulse_normalization_and_selected_trial_timing(self):
        markers = [[10, 30, 50], [5, 25, 45, 65]]
        config = PipelineConfig(n_test=1, preprocessing='none', removal_method='covariance',
                                neural_cov_source='interpulse', pulse_pre_ms=2, pulse_post_ms=4,
                                interpulse_min_samples=10, normalization='global_per_channel')
        selected = self.x[-1:]
        normalized, stats = normalize_channels(selected, config.normalization)
        source = prepare_neural_covariance(self.dataset, config, selected, stats, markers)
        mask = source.diagnostics['safe_mask'][0]
        expected_mask = interpulse_mask(markers[1], 80, 1000, 2, 4, .5)
        np.testing.assert_array_equal(mask, expected_mask)
        samples = normalized[0][:, mask]
        np.testing.assert_allclose(source.at(0, 10), samples@samples.T/samples.shape[1], atol=1e-13)

    def test_fallback_and_missing_sources(self):
        _, stats = normalize_channels(self.x)
        config = PipelineConfig(preprocessing='none', removal_method='covariance', neural_cov_source='interpulse')
        with self.assertWarns(RuntimeWarning):
            source = prepare_neural_covariance(self.dataset, config, self.x, stats, [5])
        self.assertEqual(source.diagnostics['source'], 'rest_fallback')
        with warnings.catch_warnings(record=True) as caught:
            source = prepare_neural_covariance(replace(self.dataset, baseline=None), config, self.x, stats, [])
        self.assertEqual(source.diagnostics['source'], 'bypass')
        self.assertIsNone(source.at(0, 0))
        self.assertEqual(len(caught), 2)
        with self.assertRaises(ValueError):
            prepare_neural_covariance(replace(self.dataset, baseline=None), replace(config, neural_cov_source='rest'), self.x, stats)

    def test_loader_y_clean_fallback(self):
        dataset = replace(self.dataset, baseline=None, y_clean=self.rest)
        config = PipelineConfig(preprocessing='none', removal_method='covariance')
        _, stats = normalize_channels(self.x)
        source = prepare_neural_covariance(dataset, config, self.x, stats)
        self.assertEqual(source.diagnostics['rest_data'], 'y_clean')


class PipelineChecks(unittest.TestCase):
    def test_identification_adaptive_k_and_harmonic_unchanged(self):
        rng = np.random.default_rng(12)
        dataset = Dataset('test', rng.normal(size=(2, 3, 120)), 1000, 50,
                          baseline=rng.normal(size=(2, 3, 150)))
        markers = [10, 30, 50, 70, 90, 110]
        config = PipelineConfig(n_test=0, rank=2, adaptive_k=True, adaptive_k_check_period=10,
                                normalization='global_per_channel', preprocessing='lpf', lpf_cutoff_hz=100,
                                harmonic_filter=True, harmonic_max_harmonics=2, carry_tracker_across_trials=True)
        original = run_pipeline(dataset, config)
        for source in ('rest', 'interpulse'):
            result = run_pipeline(dataset, replace(config, removal_method='covariance', neural_cov_source=source), markers)
            for key in ('chosen_k', 'delta_u', 'delta_u_ema', 'u_trace', 'angle_trace_deg'):
                np.testing.assert_array_equal(result.diagnostics[key], original.diagnostics[key])
            for key in ('raw', 'preprocessed', 'normalized', 'tracking_input'):
                np.testing.assert_array_equal(result.stages[key], original.stages[key])
            self.assertIn('after_harmonic', result.stages)
            self.assertEqual(result.stages['cleaned'].shape, original.stages['cleaned'].shape)
            self.assertTrue(np.isfinite(result.stages['cleaned']).all())

    def test_rx_recursion_and_covariance_each_current_basis(self):
        rng = np.random.default_rng(13)
        dataset = Dataset('test', rng.normal(size=(1, 3, 20)), 1000, 50,
                          baseline=rng.normal(size=(1, 3, 30)))
        config = PipelineConfig(n_test=0, rank=2, preprocessing='none', removal_method='covariance',
                                use_derivative_for_tracking=False, covariance_beta=.8)
        result = run_pipeline(dataset, config)
        Rn = result.diagnostics['neural_covariance_initial'][0]
        Rx = Rn.copy()
        for t in range(20):
            x = dataset.X[0, :, t]
            Rx = .8*Rx+.2*np.outer(x, x)
            U = result.diagnostics['u_trace'][0, t]
            k = result.diagnostics['chosen_k'][0, t]
            expected = covariance_aware_subspace_suppression(x, U[:, :k], Rn, Rx)
            np.testing.assert_allclose(result.stages['after_subspace'][0, :, t], expected, atol=1e-14)
        np.testing.assert_allclose(result.diagnostics['mixed_covariance_final'][0], Rx, atol=1e-14)

    def test_streaming_and_mcc_shapes(self):
        rng = np.random.default_rng(3)
        dataset = Dataset('test', rng.normal(size=(1, 3, 30)), 1000, 50,
                          baseline=rng.normal(size=(1, 3, 40)))
        for mode in ('none', 'global_per_channel', 'per_trial', 'streaming_causal'):
            for tracker in ('PASTd', 'MCC_OPAST'):
                config = PipelineConfig(n_test=0, rank=1, preprocessing='none', removal_method='covariance',
                                        normalization=mode, tracker=tracker)
                output = run_pipeline(dataset, config).stages['cleaned']
                self.assertEqual(output.shape, (1, 3, 29))
                self.assertTrue(np.isfinite(output).all())


if __name__ == '__main__':
    unittest.main()
