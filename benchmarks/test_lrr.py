"""Offline fitting, strict online causality, and benchmark integration checks."""
import unittest
from unittest.mock import patch
import numpy as np

from data_loading import Dataset
from . import fit_lrr, OnlineLRR, apply_lrr_batch, BenchmarkConfig, run_benchmark
from .methods import fit_lrr_trials, lrr
from .metrics import clean_rest_output


class LRRTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(123)
        self.train = self.rng.normal(size=(4, 300))
        self.mask = np.arange(300) % 3 == 0
        self.W = fit_lrr(self.train, self.mask)

    def test_offline_ols_excludes_self_and_unmasked_samples(self):
        self.assertEqual(self.W.shape, (4, 4))
        np.testing.assert_array_equal(np.diag(self.W), 0)
        for c in range(4):
            others = np.arange(4) != c
            expected = np.linalg.lstsq(self.train[others][:, self.mask].T,
                                      self.train[c, self.mask], rcond=None)[0]
            np.testing.assert_allclose(self.W[c, others], expected)
        changed = self.train.copy()
        changed[:, ~self.mask] = 1e6
        np.testing.assert_array_equal(fit_lrr(changed, self.mask), self.W)
        model = OnlineLRR(self.W)
        for c in range(4):
            impulse = np.eye(4)[c]
            self.assertEqual(model.process_sample(impulse)[c], 1.)

    def test_exact_sample_shape(self):
        model = OnlineLRR(self.W)
        self.assertEqual(model.process_sample(np.ones(4)).shape, (4,))
        for shape in [(), (3,), (5,), (4, 1), (1, 4), (4, 2), (1, 4, 1)]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                model.process_sample(np.ones(shape))
        with self.assertRaises(ValueError):
            model.process_sample([0, 0, np.nan, 0])

    def test_history_independence_and_fixed_owned_weights(self):
        weights = self.W.copy()
        model = OnlineLRR(weights)
        weights[:] = 0
        model.W[:] = 99
        x = self.rng.normal(size=4)
        expected = x - self.W @ x
        with patch('numpy.linalg.lstsq', side_effect=AssertionError('online fitting')):
            for _ in range(20):
                model.process_sample(self.rng.normal(size=4))
                np.testing.assert_array_equal(model.process_sample(x), expected)
            model.reset()
            np.testing.assert_array_equal(model.process_sample(x), expected)
        np.testing.assert_array_equal(model.W, self.W)

    def test_sample_batch_equivalence_and_no_mutation(self):
        X = self.rng.normal(size=(4, 137))
        original = X.copy()
        model = OnlineLRR(self.W)
        streamed = np.column_stack([model.process_sample(X[:, t]) for t in range(X.shape[1])])
        np.testing.assert_allclose(streamed, apply_lrr_batch(X, self.W), atol=1e-14)
        np.testing.assert_array_equal(X, original)
        np.testing.assert_array_equal(model.W, self.W)

    def test_artifact_reduction_and_independent_neural_preservation(self):
        C, T = 16, 6000
        gains = np.linspace(.8, 1.2, C)[:, None]
        training = self.rng.normal(size=(C, T)) + gains*self.rng.normal(0, 20, (1, T))
        W = fit_lrr(training, np.ones(T, dtype=bool))
        neural = self.rng.normal(size=(C, 1500))
        artifact = gains*self.rng.normal(0, 20, (1, 1500))
        model = OnlineLRR(W)
        clean = np.column_stack([model.process_sample(x) for x in (neural+artifact).T])
        residual_artifact = apply_lrr_batch(artifact, W)
        self.assertLess(np.mean(residual_artifact**2), np.mean(artifact**2)*.001)
        self.assertLess(np.mean((clean-neural)**2), np.mean(artifact**2)*.001)
        preserved = apply_lrr_batch(neural, W)
        self.assertLess(np.mean((preserved-neural)**2), np.mean(neural**2)*.15)
        self.assertGreater(np.mean([np.corrcoef(a, b)[0, 1] for a, b in zip(neural, preserved)]), .94)

    def test_training_adapter_and_online_benchmark_and_rest(self):
        train = self.rng.normal(size=(2, 4, 30))
        config = BenchmarkConfig(method='lrr', pre_ms=1, post_ms=3)
        W = fit_lrr_trials(train, [[0, 2], [28]], 1000, config)
        mask = np.zeros((2, 30), dtype=bool)
        mask[0, :5] = True
        mask[1, 27:] = True
        np.testing.assert_array_equal(W, fit_lrr(np.concatenate(train, axis=1), mask.ravel()))
        X = self.rng.normal(size=(2, 4, 20))
        dataset = Dataset('test', X, 1000, 20)
        calls = []
        original = OnlineLRR.process_sample
        def process(model, sample):
            calls.append(sample.copy())
            return original(model, sample)
        with patch.object(OnlineLRR, 'process_sample', process):
            result = run_benchmark(dataset, config, W=W)
        self.assertEqual(len(calls), 40)
        np.testing.assert_array_equal(np.stack(calls), X.transpose(0, 2, 1).reshape(-1, 4))
        np.testing.assert_allclose(result.cleaned, np.stack([apply_lrr_batch(tr, W) for tr in X]))
        np.testing.assert_allclose(result.cleaned+result.artifact, X)
        reference, rest = clean_rest_output(X, 'lrr', result, config, 1000, None)
        np.testing.assert_array_equal(reference, X)
        np.testing.assert_allclose(rest, result.cleaned)
        with self.assertRaises(ValueError):
            lrr(X)

    def test_invalid_training_and_weights(self):
        for mask in [np.ones(300), np.ones(299, dtype=bool), np.zeros(300, dtype=bool), self.mask[None]]:
            with self.assertRaises(ValueError):
                fit_lrr(self.train, mask)
        for X in [np.zeros((0, 3)), np.zeros((3, 0)), np.ones(3), np.full((2, 3), np.inf)]:
            with self.assertRaises(ValueError):
                fit_lrr(X, np.ones(3, dtype=bool))
        for W in [np.eye(4), np.zeros((4, 3)), np.zeros(4), np.zeros((0, 0)), np.full((4, 4), np.nan)]:
            with self.assertRaises(ValueError):
                OnlineLRR(W)
        with self.assertRaises(ValueError):
            apply_lrr_batch(np.ones((3, 5)), self.W)

    def test_single_channel_and_rank_deficient_training(self):
        np.testing.assert_array_equal(fit_lrr(np.ones((1, 5)), np.ones(5, dtype=bool)), [[0.]])
        W = fit_lrr(np.ones((3, 5)), np.ones(5, dtype=bool))
        np.testing.assert_allclose(OnlineLRR(W).process_sample(np.ones(3)), 0, atol=1e-14)


if __name__ == '__main__':
    unittest.main()
