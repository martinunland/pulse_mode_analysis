
from dataclasses import dataclass
import logging
from typing import List, Tuple
from scipy.integrate import simps
import numpy as np

log = logging.getLogger(__name__)


class PulseModeAnalysis:
    """
    Data analysis implementation for pulse mode measurements.
    """

    def __init__(self, sampling_interval: float, baseline_tmin: float, baseline_tmax:float) -> None:
        self.sampling_interval = sampling_interval
        self.baseline_tmin = baseline_tmin
        self.baseline_tmax = baseline_tmax

    def get_simple_intensity(self, data, signal_mask=None, baseline_mask=None):

        if signal_mask == None:
            signal_mask = np.ones(
                data[0].size
            )  # The entire waveform will be integrated

        if baseline_mask == None:
            baseline = 0
            baseline_error = 0
        else:
            baseline, baseline_error = self.get_baseline(data, baseline_mask)

        charge = []
        for waveform in data:
            charge.append(
                simps(
                    waveform[signal_mask] - baseline,
                    self.time_axis[signal_mask],
                )
            )
        mean = np.mean(charge)
        error = np.std(charge) / np.sqrt(len(charge) - 1)
        return (mean, error), (baseline, baseline_error)

    def update_time_axis(self, waveform):
        """
        Make the time array based on the configured waveform length.
        Args:
            block: The input data block containing the waveforms.
        """
        log.debug("Updating/making time axis and baseline/signal masks...")
        self.time_axis = np.arange(0, waveform.size, 1) * self.sampling_interval

        self.baseline_mask = np.logical_and(
            self.time_axis > self.baseline_tmin,
            self.time_axis < self.baseline_tmax,
        )
        log.debug("Finished making time axis & masks!")


    def get_pulse_shape(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[float, float, float]:
        log.debug("Calculating pulse shape parameters")
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy array")
        if not isinstance(y, np.ndarray):
            raise TypeError("y must be a numpy array")

        max_index, max_val, x_at_max = self.get_max_index(x, y)
        first_part = x <= x_at_max
        second_part = x >= x_at_max

        x1_ar, x2_ar = [], []
        limits = np.array([0.8, 0.5, 0.2])
        for limit in limits:
            for idx, val in enumerate(y[first_part][::-1]):
                if val < limit * max_val:
                    x1_ar.append(
                        np.interp(
                            limit * max_val,
                            [val, y[first_part][::-1][idx - 1]],
                            [x[first_part][::-1][idx], x[first_part][::-1][idx - 1]],
                        )
                    )
                    break
            for idx, val in enumerate(y[second_part]):
                if val < limit * max_val:
                    x2_ar.append(
                        np.interp(
                            limit * max_val,
                            [y[second_part][idx - 1], val],
                            [x[second_part][::-1][idx - 1], x[second_part][::-1][idx]],
                        )
                    )
                    break

        FWHM = x2_ar[1] - x1_ar[1]
        RT = x1_ar[0] - x1_ar[2]
        FT = x2_ar[2] - x2_ar[0]

        return FWHM, RT, FT

    def get_baseline(
        self, waveformBlock: np.ndarray, mask: np.ndarray
    ) -> Tuple[float, float]:
        log.debug("Calculating mean baseline level of data block...")
        baselines = []
        for waveform in waveformBlock:
            baselines.append(waveform[mask])
        return np.average(baselines), np.std(baselines) / np.sqrt(len(baselines) - 1)

    def get_max_index(self, x, y):
        log.debug("Getting index max")
        max_index = np.argmax(y)
        max_val = y[max_index]
        x_at_max = x[max_index]
        return max_index, max_val, x_at_max

    def extract_pulse_region(self, waveform, max_index):
        log.debug("Selecting region of interest")
        start_index = max_index - int(15e-9 / self.sampling_interval)
        end_index = max_index + int(15e-9 / self.sampling_interval)
        start_index = max(start_index, 0)
        end_index = min(end_index, len(waveform) - 1)
        pulse = waveform[start_index:end_index]
        pulse_time = self.time_axis[start_index:end_index]
        return pulse, pulse_time

    def process_waveform(self, waveform) -> List:
        log.debug("Processing waveform")
        max_index, amplitude, transit_time = self.get_max_index(
            self.time_axis, waveform
        )
        pulse, pulse_time = self.extract_pulse_region(waveform, max_index)

        try:
            FWHM, RT, FT = self.get_pulse_shape(pulse_time, pulse)
        except Exception as _:
            FWHM, RT, FT = [-1, -1, -1]
            log.debug(
                "Calculating pulse shape parameters failed, passing default values"
            )

        log.debug("Calculating charges")
        charge = simps(pulse * 1e-3, pulse_time * 1e-9)
        pedestal_charge = simps(
            waveform[self.baseline_mask] * 1e-3,
            self.time_axis[self.baseline_mask] * 1e-9,
        )
        log.debug("Finished processing waveform")
        return pedestal_charge, transit_time, charge, amplitude, FWHM, RT, FT
