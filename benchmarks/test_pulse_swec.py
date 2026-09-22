"""SWEC component mixing and parameter-derived stimulation timing."""
import unittest
import numpy as np

from .train_pulse_swec import FS, make_example, configuration_markers, pad_channels


class SWECPreparationTests(unittest.TestCase):
    def test_channel_padding(self):
        for channels in (1, 88, 128):
            original = np.ones((2, channels, 80), dtype=np.float32)
            padded = pad_channels(original)
            self.assertEqual(padded.shape, (2, 128, 80))
            self.assertEqual(padded.dtype, original.dtype)
            np.testing.assert_array_equal(padded[:, :channels], original)
            np.testing.assert_array_equal(padded[:, channels:], 0)
            self.assertEqual(pad_channels(original[0]).shape, (128, 80))
        with self.assertRaises(ValueError):
            pad_channels(np.zeros((2, 129, 80)))

    def test_components_and_staggered_stimulation_sites(self):
        clean = np.ones((2, 3, 80), dtype=np.float32)
        artifacts = np.stack([np.full_like(clean, 2), np.full_like(clean, 5)])
        params = np.array([[[129, 73, 0, 2500, 1., 1.005]]*2,
                           [[129, 73, 8, 2500, 1.0002, 1.0052]]*2])
        original = clean.copy()
        single, trace = make_example(clean, artifacts, params, 1, [0], int(FS))
        combined, combined_trace = make_example(clean, artifacts, params, 1, [0, 8], int(FS))
        np.testing.assert_array_equal(single, 3)
        np.testing.assert_array_equal(combined, 8)
        np.testing.assert_array_equal(clean, original)
        self.assertEqual(trace.shape, (1, 80))
        np.testing.assert_array_equal(trace[0, :5], [-1, -1, -1, 1, 1])
        self.assertTrue(np.isfinite(combined_trace).all())
        np.testing.assert_array_equal(configuration_markers(params, 1, [0, 8], int(FS), 80), [0, 3])
        with self.assertRaises(ValueError):
            make_example(clean, artifacts, params, 1, [2], int(FS))


if __name__ == '__main__':
    unittest.main()
