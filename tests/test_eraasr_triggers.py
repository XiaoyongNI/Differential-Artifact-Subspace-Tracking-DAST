"""Check onset indexing and missing-pulse behavior on known synthetic events."""
import unittest
import warnings
import numpy as np
from label_eraasr_triggers import detect_pulse_triggers


class TriggerTests(unittest.TestCase):
    def signal(self, onsets):
        signal = np.zeros((1, 3, 100))
        # Strongest edge is later than the onset, as with biphasic stimulation.
        waveform = np.array([1., 2., 6., -5., -3., 0.])
        for onset in onsets:
            signal[0, :, onset:onset+len(waveform)] += np.array([1., 2., 3.])[:, None]*waveform
        return signal

    def test_leading_edge_indices_not_late_peak(self):
        triggers, _, diagnostics = detect_pulse_triggers(self.signal([20, 40, 60, 80]), 1000, 50)
        np.testing.assert_array_equal(triggers[0], [20, 40, 60, 80])
        self.assertNotEqual(diagnostics[0]['peak_samples'][0], 20)

    def test_missing_pulses_are_not_filled(self):
        with self.assertWarns(RuntimeWarning):
            triggers, _, _ = detect_pulse_triggers(self.signal([20, 60, 80]), 1000, 50)
        np.testing.assert_array_equal(triggers[0], [20, 60, 80])

    def test_no_signal_yields_no_triggers(self):
        with self.assertWarns(RuntimeWarning):
            triggers, _, _ = detect_pulse_triggers(np.zeros((1, 3, 100)), 1000, 50)
        self.assertEqual(triggers[0].size, 0)

    def test_channel_median_rejects_isolated_channel_transient(self):
        signal = self.signal([20, 40, 60, 80])
        signal[0, 0, 5] = 1000
        triggers, _, _ = detect_pulse_triggers(signal, 1000, 50)
        np.testing.assert_array_equal(triggers[0], [20, 40, 60, 80])


if __name__ == '__main__':
    unittest.main()
