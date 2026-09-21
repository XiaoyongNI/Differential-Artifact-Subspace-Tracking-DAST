"""Dictionary numerical stages, held-out fitting, boundaries, and runner contracts."""
from dataclasses import replace
import unittest
from unittest.mock import patch
import numpy as np

from data_loading import Dataset
from . import BenchmarkConfig, dictionary_learning, fit_dictionary, run_benchmark
from .dictionary_learning import _best_template, _cluster, _normalize, dictionary_windows
from .metrics import clean_rest_output
from .run import synthetic_dataset


class DictionaryTests(unittest.TestCase):
    def setUp(self):
        self.config = BenchmarkConfig(method='dictionary_learning', pre_ms=3, post_ms=5)
        self.wave = np.array([0., 0., 0., 10., -5., 2., 0., 0.])
        self.times = np.arange(10, 310, 10)

    def signal(self, waves, baseline=4.):
        x = np.full((1, 2, 320), baseline)
        for i, t in enumerate(self.times):
            x[0, :, t-3:t+5] += waves[i % len(waves)]
        return x

    def test_hdbscan_discovers_two_waveform_families(self):
        rng = np.random.default_rng(9)
        other = np.array([0., 0., 0., 2., -10., 6., 0., 0.])
        x = self.signal([self.wave, other])
        x += rng.normal(0, .001, x.shape)
        model = fit_dictionary(x, self.times, 1000, self.config)
        for atoms in model.templates:
            self.assertEqual(atoms.shape, (8, 2))
            for wave in (self.wave, other):
                self.assertLess(np.min(np.linalg.norm(atoms-wave[:, None], axis=0)), .01)
        result = dictionary_learning(x, self.times, 1000, self.config, model=model)
        self.assertLess(np.mean((result.cleaned-4.)**2), .001)
        self.assertEqual(result.diagnostics['skipped_windows'], 0)

    def test_scaled_subtraction_retains_baseline_and_outside_samples(self):
        train = self.signal([self.wave])
        with patch('benchmarks.dictionary_learning._cluster', return_value=(np.full(30, -1), np.ones(30))):
            model = fit_dictionary(train, self.times, 1000, self.config)
        self.assertEqual(model.diagnostics['fallbacks'], ['mean_all_pulses']*2)
        np.testing.assert_array_equal(model.templates[0][:, 0], self.wave)
        test = self.signal([self.wave*2.5], baseline=7.)
        before = test.copy()
        result = dictionary_learning(test, self.times, 1000, self.config, model=model)
        np.testing.assert_allclose(result.cleaned, 7., atol=1e-13)
        np.testing.assert_array_equal(test, before)
        np.testing.assert_allclose(result.cleaned+result.artifact, test)
        self.assertTrue(all(abs(s['scale']-2.5) < 1e-12 for s in result.diagnostics['selections']))

    def test_held_out_training_does_not_refit_and_reuses_benchmark_api(self):
        train, test = self.signal([self.wave]), self.signal([3*self.wave])
        model = fit_dictionary(train, self.times, 1000, self.config)
        snapshots = [a.copy() for a in model.templates]
        with patch('benchmarks.dictionary_learning.fit_dictionary', side_effect=AssertionError('refit')):
            result = run_benchmark(Dataset('test', test, 1000, 100), self.config, self.times, model=model)
        self.assertEqual(result.cleaned.shape, test.shape)
        for a, b in zip(snapshots, model.templates):
            np.testing.assert_array_equal(a, b)
        other = dictionary_learning(test, self.times, 1000, self.config,
                                    training_data=train, training_stim_times=self.times)
        np.testing.assert_allclose(other.cleaned, result.cleaned)
        self.assertEqual(other.diagnostics['training_source'], 'provided_training')
        reference, cleaned = clean_rest_output(np.ones_like(test), 'dictionary_learning', result,
                                               self.config, 1000, [self.times])
        np.testing.assert_allclose(cleaned, reference-result.artifact)

    def test_normalization_and_matching_follow_source(self):
        epoch = np.array([1., 2., 6., 3., 10.])
        np.testing.assert_allclose(_normalize(epoch, self.config), epoch-3.)
        np.testing.assert_allclose(_normalize(epoch, replace(self.config, dictionary_normalize='firstSamp')), epoch-1.)
        np.testing.assert_allclose(_normalize(epoch, replace(self.config, dictionary_normalize='mean')), epoch-epoch.mean())
        np.testing.assert_array_equal(_normalize(epoch, replace(self.config, dictionary_normalize='none')), epoch)
        atoms = np.column_stack([epoch, -epoch])
        self.assertEqual(_best_template(-epoch, atoms, 2, self.config), 1)
        self.assertEqual(_best_template(-epoch, atoms, 2, replace(self.config, dictionary_match='eucl')), 1)
        self.assertEqual(_best_template(-epoch, atoms, 2, replace(self.config, dictionary_match='cosine')), 0)
        self.assertEqual(_best_template(np.ones(5)*2, np.column_stack([np.ones(5), np.ones(5)*2]),
                                        2, self.config), 1)

    def test_variable_windows_are_zero_padded_and_overlaps_merged(self):
        cfg = replace(self.config, pre_ms=0, post_ms=3, dictionary_normalize='none')
        x = np.array([[[1., 2., 3., 4., 0., 5.]]])
        model = fit_dictionary(x, [0, 1, 5], 1000, cfg)
        self.assertEqual(model.diagnostics['training_windows'], [[[(0, 4), (5, 6)]]])
        np.testing.assert_allclose(model.templates[0][:, 0], [3, 1, 1.5, 2])
        result = dictionary_learning(x, [0, 1, 5], 1000, cfg, model=model)
        self.assertTrue(np.isfinite(result.cleaned).all())
        self.assertEqual(result.cleaned[0, 0, 4], x[0, 0, 4])

    def test_constant_empty_short_and_missing_templates_are_safe(self):
        x = np.ones((1, 2, 30))
        for markers in (None, [], [0, 15, 29]):
            result = dictionary_learning(x, markers, 1000, self.config)
            np.testing.assert_array_equal(result.cleaned, x)
        short = dictionary_learning(x[:, :, :3], None, 1000, self.config)
        np.testing.assert_array_equal(short.cleaned, x[:, :, :3])
        empty_model = fit_dictionary(x, [], 1000, self.config)
        result = dictionary_learning(x, [15], 1000, self.config, model=empty_model)
        self.assertEqual(result.diagnostics['skipped_windows'], 2)

    def test_automatic_detection_and_marker_override(self):
        dataset, markers = synthetic_dataset()
        cfg = replace(self.config, pre_ms=.1, post_ms=1.1)
        bounds = dictionary_windows(dataset.X, None, dataset.fs, cfg)
        self.assertEqual(len(bounds), 3)
        for trial in bounds:
            for channel in trial:
                self.assertEqual(len(channel), len(markers))
                self.assertTrue(all(0 <= a < b <= 600 for a, b in channel))
        result = dictionary_learning(dataset.X, None, dataset.fs, cfg)
        self.assertEqual(result.diagnostics['window_source'], 'detected')
        self.assertTrue(np.isfinite(result.cleaned).all())
        np.testing.assert_array_equal(dictionary_learning(dataset.X, [], dataset.fs, cfg).cleaned, dataset.X)

    def test_glosh_rejection_and_minpoints_mapping(self):
        class Clusterer:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.labels_ = np.array([0, 0, 0, 1])
                self.outlier_scores_ = np.array([0., .95, .96, 0.])
            def fit(self, features):
                self.features = features
                return self
        with patch('hdbscan.HDBSCAN', side_effect=Clusterer) as constructor:
            labels, scores = _cluster(np.arange(8).reshape(4, 2), self.config)
        np.testing.assert_array_equal(labels, [0, 0, -1, 1])
        self.assertEqual(constructor.call_args.kwargs['min_samples'], 1)

    def test_invalid_configuration_and_model_contracts(self):
        x = self.signal([self.wave])
        for kwargs in [dict(dictionary_min_points=1), dict(dictionary_min_cluster_size=0),
                       dict(dictionary_bracket=-1), dict(dictionary_baseline_samples=0),
                       dict(dictionary_normalize='bad'), dict(dictionary_match='dtw'),
                       dict(dictionary_outlier_threshold=2), dict(dictionary_detection_ms=np.nan)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                fit_dictionary(x, self.times, 1000, replace(self.config, **kwargs))
        model = fit_dictionary(x, self.times, 1000, self.config)
        for kwargs in [dict(model=model, training_data=x), dict(training_stim_times=self.times),
                       dict(model=model, config=replace(self.config, pre_ms=2))]:
            with self.assertRaises(ValueError):
                dictionary_learning(x, self.times, 1000, **kwargs)
        with self.assertRaises(ValueError):
            dictionary_learning(x, self.times, 2000, model=model)
        with self.assertRaises(ValueError):
            dictionary_learning(x[:, :1], self.times, 1000, model=model)


if __name__ == '__main__':
    unittest.main()
