"""ASAR parity with the supplied MATLAB loop and streaming contracts."""
import unittest
from unittest.mock import patch
import numpy as np

from data_loading import Dataset
from . import OnlineASAR, calibrate_asar, asar, BenchmarkConfig, run_benchmark
from .metrics import clean_rest_output


def matlab_reference(x, length, mu, alpha, epsilon, n_stats, refs=None):
    """Literal scalar-channel transcription, independent of the vectorized engine."""
    out = np.zeros_like(x)
    weights = []
    for ch, d in enumerate(x):
        reference = x[ch if refs is None else refs[ch]]
        S, T = 0., 0.
        for n in range(n_stats):
            S += reference[n]
            T += reference[n]**2
        avg = S/n_stats
        std = np.sqrt((T - n_stats*avg**2)/(n_stats-1))
        w, u = np.zeros(length), np.zeros(length)
        for n in range(d.size):
            u[1:] = u[:-1]
            u[0] = reference[n] if abs(reference[n]-avg) >= alpha*std else 0.
            estimate = u @ w
            error = d[n] - estimate
            w = w + mu*u*error/(u @ u + epsilon)
            out[ch, n] = d[n] - u @ w
        weights.append(w)
    return out, np.stack(weights)


class ASARTests(unittest.TestCase):
    def test_neighbor_reference_parity_uses_reference_statistics(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=(4, 650)) * np.arange(1, 5)[:, None]
        x[:, 520:530] += np.array([30, -40, 50, -60])[:, None]
        result = asar(x[None])
        refs = [1, 2, 3, 2]
        expected, weights = matlab_reference(x, 201, .1, 5, 1e-4, 500, refs)
        np.testing.assert_array_equal(result.diagnostics['reference_channels'], refs)
        np.testing.assert_allclose(result.cleaned[0], expected, atol=1e-12)
        np.testing.assert_allclose(result.diagnostics['weights'][0], weights, atol=1e-12)

    def test_matlab_parity_with_explicit_self_references_and_trial_reset(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0, .2, (2, 7, 650))
        x[:, :, 520:530] += 30
        before = x.copy()
        config = BenchmarkConfig(method='asar', asar_reference_channels=tuple(range(7)))
        result = asar(x, config=config)
        for trial in range(2):
            expected, weights = matlab_reference(x[trial], 201, .1, 5, 1e-4, 500)
            np.testing.assert_allclose(result.cleaned[trial], expected, atol=1e-12)
            np.testing.assert_allclose(result.diagnostics['weights'][trial], weights, atol=1e-12)
            np.testing.assert_array_equal(asar(x[trial:trial+1], config=config).cleaned[0], result.cleaned[trial])
        np.testing.assert_allclose(result.cleaned+result.artifact, x)
        np.testing.assert_array_equal(x, before)
        self.assertEqual(result.diagnostics['calibration_source'], 'test_prefix_replay')

    def test_post_update_output_and_inclusive_gate(self):
        model = OnlineASAR([0.], [1.], filter_length=1, threshold=5, reference_channels=[0])
        self.assertEqual(model.process_sample([4.])[0], 4.)
        out = model.process_sample([5.])[0]
        weight = .1*25/(25+1e-4)
        self.assertAlmostEqual(model.weights[0, 0], weight)
        self.assertAlmostEqual(out, 5-5*weight)
        self.assertNotEqual(out, 5.)
        # A zero current reference still allows a previous gated sample to act.
        model = OnlineASAR([0.], [1.], filter_length=2, threshold=5, reference_channels=[0])
        model.process_sample([10.])
        self.assertLess(model.process_sample([1.])[0], 1.)

    def test_online_causality_reset_and_shape(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(2, 80))
        model = OnlineASAR([0, 0], [1, 1], filter_length=8, threshold=.5)
        clean = np.column_stack([model.process_sample(s) for s in x.T])
        model.reset()
        x[:, 40:] += 100
        changed = np.column_stack([model.process_sample(s) for s in x.T])
        np.testing.assert_array_equal(clean[:, :40], changed[:, :40])
        for sample in [1., [1.], np.ones((2, 1)), np.ones((1, 2)), [np.nan, 0]]:
            with self.assertRaises(ValueError):
                model.process_sample(sample)
        model.reset()
        np.testing.assert_array_equal(model.weights, 0)

    def test_reference_mapping_and_channel_independence(self):
        model = OnlineASAR([0, 0], [1, 1], filter_length=1, reference_channels=[1, 0])
        out = model.process_sample([2., 10.])
        self.assertLess(out[0], 2.)
        self.assertEqual(out[1], 10.)
        model = OnlineASAR([0, 0], [1, 1], filter_length=3)
        out = model.process_sample([2., 10.])
        self.assertLess(out[0], 2.)
        self.assertEqual(out[1], 10.)
        # Each target defaults to a different neighboring input channel.
        model = OnlineASAR(np.zeros(4), np.ones(4), filter_length=1)
        np.testing.assert_array_equal(model.reference_channels, [1, 2, 3, 2])
        for ch in range(4):
            model.reset()
            impulse = np.eye(4)[ch]*10
            self.assertEqual(model.process_sample(impulse)[ch], 10.)
        with self.assertRaises(ValueError):
            OnlineASAR([0], [1])

    def test_statistics_and_constant_calibration(self):
        x = np.array([[1., 2., 3., 1000.]])
        mean, std = calibrate_asar(x, 3)
        np.testing.assert_allclose(mean, [2])
        np.testing.assert_allclose(std, [1])
        mean, std = calibrate_asar(np.ones((2, 500)))
        model = OnlineASAR(mean, std)
        self.assertTrue(np.isfinite(model.process_sample([1, 1])).all())
        zero = OnlineASAR([0], [0], reference_channels=[0])
        np.testing.assert_array_equal(zero.process_sample([0]), [0])

    def test_provided_calibration_integration_and_frozen_rest(self):
        rng = np.random.default_rng(3)
        calibration = rng.normal(0, .1, (1, 3, 500))
        x = np.full((2, 3, 20), 10.)
        config = BenchmarkConfig(method='asar', asar_filter_length=8)
        calls = []
        original = OnlineASAR.process_sample
        def process(model, sample, **kwargs):
            calls.append(sample.shape)
            return original(model, sample, **kwargs)
        with patch.object(OnlineASAR, 'process_sample', process):
            result = run_benchmark(Dataset('test', x, 1000, 20), config,
                                   calibration_data=calibration)
        self.assertEqual(calls, [(3,)]*40)
        self.assertEqual(result.diagnostics['calibration_source'], 'provided')
        np.testing.assert_allclose(result.diagnostics['mean'][0], calibration[0].mean(axis=1))
        saved = result.diagnostics['weights'].copy()
        # Force rest samples above the threshold to exercise frozen FIR removal.
        rest = np.full((1, 3, 12), 3.)
        reference, cleaned = clean_rest_output(rest, 'asar', result, config, 1000)
        for tr in range(2):
            for ch in range(3):
                expected = reference[tr, ch] - np.convolve(reference[tr, ch], saved[tr, ch])[:12]
                np.testing.assert_allclose(cleaned[tr, ch], expected)
        np.testing.assert_array_equal(result.diagnostics['weights'], saved)

    def test_artifact_reduction_and_subthreshold_preservation(self):
        rng = np.random.default_rng(4)
        neural = rng.uniform(-.1, .1, (1, 2, 2000))
        artifact = np.zeros_like(neural)
        artifact[:, :, 50::40] = 20
        calibration = rng.normal(0, .1, (1, 2, 500))
        config = BenchmarkConfig(method='asar', asar_filter_length=1)
        result = asar(neural+artifact, config=config, calibration_data=calibration)
        self.assertLess(np.mean((result.cleaned-neural)**2), np.mean(artifact**2)*.12)
        np.testing.assert_array_equal(result.cleaned[artifact == 0], neural[artifact == 0])

    def test_validation(self):
        for kwargs in [dict(filter_length=0), dict(filter_length=1.5), dict(mu=0), dict(mu=2),
                       dict(epsilon=0), dict(threshold=-1), dict(mu=np.nan),
                       dict(reference_channels=[2]), dict(reference_channels=[.0])]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                OnlineASAR([0], [1], **kwargs)
        for mean, std in [([], []), ([0], [-1]), ([np.inf], [1]), ([0], [1, 2])]:
            with self.assertRaises(ValueError):
                OnlineASAR(mean, std)
        for n in [1, 501, 2.5]:
            with self.assertRaises(ValueError):
                calibrate_asar(np.ones((1, 500)), n)
        with self.assertRaises(ValueError):
            asar(np.ones((1, 2, 499)))
        with self.assertRaises(ValueError):
            asar(np.ones((2, 2, 500)), calibration_data=np.ones((3, 2, 500)))


if __name__ == '__main__':
    unittest.main()
