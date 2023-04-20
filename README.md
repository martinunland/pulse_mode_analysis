# Pulse Mode Analysis
This repository contains a Python implementation for analyzing pulse mode measurements. The main class, PulseModeAnalysis, provides methods for processing waveform data and extracting various parameters, such as baseline, pulse shape, and charge.

To install the library use pip:
```bash
pip install git+https://github.com/martinunland/pulse_mode_analysis.git
```
Or download the wheel in repositories and pip:
```bash
pip install pulse_mode_analysis-0.1.0-py3-none-any.whl
```
## Usage
First, import the PulseModeAnalysis class:

```python
from pulse_mode_analysis import PulseModeAnalysis
```

Next, create an instance of PulseModeAnalysis with the desired sampling interval, baseline minimum time, and baseline maximum time:

```python
analysis = PulseModeAnalysis(sampling_interval=1e-9, baseline_tmin=1e-9, baseline_tmax=10e-9)
(...)
analysis.update_time_axis(waveform)
```

Now, you can use the various methods provided by the PulseModeAnalysis class to process and analyze the waveform data, for example:

```python
result = analysis.process_waveform(waveform)
```
