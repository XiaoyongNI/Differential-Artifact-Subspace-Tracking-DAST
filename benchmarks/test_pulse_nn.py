"""PULSE source parity, unsupervised training, inference and checkpoint contracts."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

from data_loading import Dataset
from . import BenchmarkConfig, METHODS, pulse, low_rank_tv, run_benchmark
from .pulse_nn import (UNet1D, PhysicsLoss, LOSS_KEYS, fit_pulse, save_pulse, load_pulse,
                       pulse_artifact_mask, pulse_trace_from_markers)
from .metrics import clean_rest_output


SOURCE = Path(__file__).resolve().parents[1] / 'paper/pulse/sparc-nn'


def source_module(name, path):
    spec = importlib.util.spec_from_file_location(name, SOURCE/path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PULSETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.config = BenchmarkConfig(method='pulse', pulse_epochs=1, pulse_device='cpu',
                                    pulse_artifact_duration_ms=8, pulse_blur_samples=2)
        cls.x = np.random.default_rng(5).normal(size=(2, 3, 64))
        cls.trace = pulse_trace_from_markers([16, 40], 2, 64)
        cls.model = fit_pulse(cls.x, cls.trace, 1000, cls.config)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_network_matches_local_source_including_odd_length(self):
        original = source_module('source_unet', 'models/unet1d.py').UNet1D(4, 3).eval()
        original.load_state_dict(self.model.network.state_dict())
        x, trace = torch.randn(1, 3, 65), torch.randn(1, 1, 65)
        with torch.no_grad():
            np.testing.assert_array_equal(self.model.network(x, trace).numpy(), original(x, trace).numpy())

    def test_all_five_losses_match_source_and_have_gradients(self):
        original = source_module('source_loss', 'loss.py').PhysicsLoss(sampling_rate=1000, f_cutoff=10)
        local = PhysicsLoss(sampling_rate=1000, f_cutoff=10)
        s, a = torch.randn(2, 3, 300, requires_grad=True), torch.randn(2, 3, 300, requires_grad=True)
        expected, actual = original(s, a), local(s, a)
        self.assertEqual(tuple(actual), LOSS_KEYS)
        for key in LOSS_KEYS:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        sum(actual.values()).backward()
        self.assertTrue(torch.isfinite(s.grad).all() and torch.isfinite(a.grad).all())

    def test_soft_masks_match_source_and_preserve_overlaps(self):
        original = source_module('source_data', 'data_utils.py')
        actual = pulse_artifact_mask([[0, 5, 30], [10]], 2, 32, 1000, duration_ms=8, blur_samples=5)
        for tr, times in enumerate([[0, 5, 30], [10]]):
            expected = original.create_soft_artifact_mask(times, [8]*len(times), 32, blur_radius=5)
            np.testing.assert_array_equal(actual[tr], expected.numpy())
        self.assertEqual(actual.max(), 1.)

    def test_training_is_unsupervised_and_normalization_uses_training_only(self):
        original = source_module('source_stats', 'data_utils.py')
        mean, std = original.compute_robust_stats(self.x.astype(np.float32))
        torch.testing.assert_close(self.model.mean, mean)
        torch.testing.assert_close(self.model.std, std)
        self.assertEqual(len(self.model.history), 1)
        self.assertTrue(all(np.isfinite(v) for v in self.model.history[0].values()))
        # Recreate initialization to prove an optimizer step changed weights.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.config.random_state)
            initial = UNet1D(4, 3)
        self.assertFalse(torch.equal(initial.inc.conv1[0].weight, self.model.network.inc.conv1[0].weight))
        self.assertTrue(all(not p.requires_grad for p in self.model.network.parameters()))

    def test_frozen_inference_blending_and_reconstruction(self):
        before = {k: v.clone() for k, v in self.model.network.state_dict().items()}
        x = self.x+50
        original = x.copy()
        with patch('benchmarks.pulse_nn.fit_pulse', side_effect=AssertionError('refit')):
            result = run_benchmark(Dataset('test', x, 1000, 20), self.config, [16, 40],
                                   model=self.model, stim_trace=self.trace)
        mask = result.diagnostics['artifact_mask']
        outside = np.broadcast_to(mask == 0, x.shape)
        np.testing.assert_array_equal(result.cleaned[outside], x[outside])
        np.testing.assert_allclose(result.cleaned+result.artifact, x)
        np.testing.assert_array_equal(x, original)
        with torch.no_grad():
            scale = self.model.std+1e-8
            norm = (torch.tensor(x, dtype=torch.float32)-self.model.mean)/scale
            a = self.model.network(norm, torch.from_numpy(self.trace))
            s = ((norm-a)*scale+self.model.mean).numpy()
        np.testing.assert_allclose(result.cleaned, x+mask*(s-x), atol=2e-5)
        for key, value in self.model.network.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.pt'
            save_pulse(self.model, path)
            loaded = load_pulse(path, 'cpu')
            restored = pulse(x, [16, 40], 1000, model=loaded, stim_trace=self.trace)
        np.testing.assert_array_equal(restored.cleaned, result.cleaned)

    def test_predict_neural_uncertainty_and_rest_frozen_evaluation(self):
        config = replace(self.config, pulse_predict_neural=True, pulse_uncertainty=True)
        model = fit_pulse(self.x[:1], self.trace[:1], 1000, config)
        self.assertIsNotNone(model.uncertainty_state)
        self.assertFalse(torch.equal(model.uncertainty_state['log_vars'], torch.zeros(5)))
        result = pulse(self.x[:1], [16, 40], 1000, model=model, stim_trace=self.trace[:1])
        before = {k: v.clone() for k, v in model.network.state_dict().items()}
        rest = np.random.default_rng(1).normal(size=(1, 3, 80))
        reference, clean = clean_rest_output(rest, 'pulse', result, config, 1000)
        self.assertEqual(reference.shape, clean.shape)
        np.testing.assert_array_equal(clean[..., 64:], rest[..., 64:])
        for key, value in model.network.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_source_checkpoint_loads_strictly_without_test_statistics(self):
        source = {'model_state_dict': self.model.network.state_dict(),
                  'model_config': {'in_channels': 4, 'out_channels': 3},
                  'training_config': {'sampling_rate': 1000, 'predict_neural': False},
                  'data_mean': self.model.mean, 'data_std': self.model.std}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'source.pt'
            torch.save(source, path)
            model = load_pulse(path, 'cpu')
            torch.testing.assert_close(model.mean, self.model.mean)
            del source['data_std']
            torch.save(source, path)
            with self.assertRaisesRegex(ValueError, 'data_std'):
                load_pulse(path, 'cpu')

    def test_validation_and_distinct_registry_entries(self):
        self.assertIs(METHODS['pulse'], pulse)
        self.assertIs(METHODS['low_rank_tv'], low_rank_tv)
        for kwargs in [dict(), dict(model=self.model), dict(model=self.model, stim_trace=np.ones((2, 64))),
                       dict(model=self.model, stim_trace=self.trace, artifact_mask=np.ones((2, 1, 64))*2)]:
            with self.assertRaises(ValueError):
                pulse(self.x, [16, 40], 1000, **kwargs)
        with self.assertRaises(ValueError):
            pulse(self.x[..., :8], [], 1000, model=self.model, stim_trace=self.trace[..., :8])
        with self.assertRaises(ValueError):
            pulse(self.x, [16], 2000, model=self.model, stim_trace=self.trace)
        with self.assertRaises(ValueError):
            fit_pulse(self.x, self.trace, 1000, replace(self.config, pulse_epochs=0))
        with self.assertRaises(ValueError):
            fit_pulse(self.x, self.trace, 1000, replace(self.config, pulse_f_cutoff=500))


if __name__ == '__main__':
    unittest.main()
