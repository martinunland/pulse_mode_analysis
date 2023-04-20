# run tests with 'python -m unittest test_pulse_mode_analysis.py'
import unittest
import numpy as np
from pulse_mode_analysis import PulseModeAnalysis, INVALID_VALUE


class TestPulseModeAnalysis(unittest.TestCase):
    def setUp(self):
        self.sampling_interval = 1e-9
        self.baseline_tmin = 0
        self.baseline_tmax = 10e-9
        self.analysis = PulseModeAnalysis(
            self.sampling_interval, self.baseline_tmin, self.baseline_tmax
        )

    def test_update_time_axis(self):
        waveform = np.random.rand(1000)
        self.analysis.update_time_axis(waveform)
        self.assertEqual(len(self.analysis.time_axis), len(waveform))

    def test_get_baseline(self):
        waveform_block = np.random.rand(10, 1000)
        self.analysis.update_time_axis(waveform_block[0])
        mean, error = self.analysis.get_baseline(waveform_block)
        self.assertIsInstance(mean, float)
        self.assertIsInstance(error, float)

    def test_get_pulse_shape(self):
        x = np.linspace(0, 1, 1000)
        y = np.zeros(x.size)
        y[250:750] = 1
        FWHM, RT, FT = self.analysis.get_pulse_shape(y, x)
        self.assertAlmostEqual(FWHM, 0.5, places=2)

    def test_get_pulse_shape_gaussian(self):
        width = 300
        x = np.linspace(-2000, 2000, 4000)
        from scipy.signal import gaussian

        y = gaussian(4000, std=width)
        FWHM, RT, FT = self.analysis.get_pulse_shape(y, x)

        # FWHM of a Gaussian should be approximately 2.3548 * std
        expected_FWHM = 2.3548 * width
        self.assertAlmostEqual(FWHM, expected_FWHM, delta=1)
        # RT and FT for a Gaussian should be approximately (80% - 20%) * FWHM

        max_y = np.max(y)
        y_20 = 0.2 * max_y
        y_80 = 0.8 * max_y

        x_20 = x[np.argmin(np.abs(y - y_20))]
        x_80 = x[np.argmin(np.abs(y - y_80))]

        expected_RT_FT = abs(x_80 - x_20)

        self.assertAlmostEqual(RT, expected_RT_FT, delta=1)
        self.assertAlmostEqual(FT, expected_RT_FT, delta=1)

    def test_get_simple_intensity_with_signal_mask(self):
        data = np.random.rand(10, 1000)
        self.analysis.update_time_axis(data[0])

        signal_mask = np.ones(data.shape[1], dtype=bool)
        signal_mask[:100] = False
        signal_mask[-100:] = False

        (mean, error), (baseline, baseline_error) = self.analysis.get_simple_intensity(
            data, signal_mask=signal_mask
        )
        self.assertIsInstance(mean, float)
        self.assertIsInstance(error, float)
        self.assertIsInstance(baseline, float)
        self.assertIsInstance(baseline_error, float)

    def test_get_maximum_index_and_coordinates(self):
        index = 250
        max_value = 12
        x = np.linspace(0, 1, 1000)
        y = np.zeros(x.size)
        y[index] = max_value
        max_index, y_at_max, x_at_max = self.analysis.get_maximum_index_and_coordinates(
            x, y
        )
        self.assertEqual(max_index, index)
        self.assertAlmostEqual(y_at_max, max_value)
        self.assertAlmostEqual(x_at_max, x[index])

    def test_extract_pulse_region(self):
        waveform = np.random.rand(1000)
        max_index = 500
        self.analysis.update_time_axis(waveform)
        pulse, pulse_time = self.analysis.extract_pulse_region(waveform, max_index)
        self.assertEqual(len(pulse), len(pulse_time))

    def test_integrate_waveform_in_Wb(self):
        time = np.linspace(0, 1e-9, 1000)
        amplitude = np.ones(1000)
        result = self.analysis.integrate_waveform_in_Wb(time, amplitude)
        self.assertAlmostEqual(result, 1e-9, places=4)

    def test_process_waveform(self):
        waveform = np.random.rand(1000)
        self.analysis.update_time_axis(waveform)
        result = self.analysis.process_waveform(waveform)
        self.assertEqual(len(result), 7)

    def test_get_simple_intensity(self):
        data = np.random.rand(10, 1000)
        self.analysis.update_time_axis(data[0])
        (mean, error), (baseline, baseline_error) = self.analysis.get_simple_intensity(
            data
        )
        self.assertIsInstance(mean, float)
        self.assertIsInstance(error, float)
        self.assertIsInstance(baseline, float)
        self.assertIsInstance(baseline_error, float)

    def test_init(self):
        analysis = PulseModeAnalysis(1e-9, 0, 10e-9)
        self.assertEqual(analysis.sampling_interval, 1e-9)
        self.assertEqual(analysis.baseline_tmin, 0)
        self.assertEqual(analysis.baseline_tmax, 10e-9)


if __name__ == "__main__":
    unittest.main()
