# Battery Impedance Analyzer API Reference

## Overview

The Battery Impedance Analyzer library provides a comprehensive set of classes and functions for performing electrochemical impedance spectroscopy (EIS) analysis on battery systems. This document outlines the core modules, classes, and methods available in the API.

## Core Modules

### `battery_analyzer`

Main package containing all analyzer functionality.

```python
import battery_analyzer
```

## Main Classes

### `ImpedanceAnalyzer`

Primary class for configuring hardware connections and performing impedance measurements.

```python
from battery_analyzer import ImpedanceAnalyzer
```

#### Constructor

```python
analyzer = ImpedanceAnalyzer(device_type='potentiostat', threads=4)
```

**Parameters:**
- `device_type` (str): Type of measurement hardware ('potentiostat', 'impedance_analyzer', or 'custom')
- `threads` (int): Number of processing threads to use for data analysis
- `calibration_file` (str, optional): Path to calibration file
- `log_level` (str, optional): Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')

#### Methods

##### `connect()`

Connect to the measurement hardware.

```python
analyzer.connect(port='/dev/ttyUSB0', baud_rate=115200)
```

**Parameters:**
- `port` (str): Serial port or connection string
- `baud_rate` (int, optional): Connection baud rate for serial devices
- `timeout` (float, optional): Connection timeout in seconds

**Returns:**
- `bool`: True if connection successful, False otherwise

##### `measure_impedance()`

Perform an impedance spectroscopy measurement.

```python
data = analyzer.measure_impedance(
    freq_start=10000, 
    freq_end=0.01, 
    points=50,
    amplitude=10
)
```

**Parameters:**
- `freq_start` (float): Starting frequency in Hz
- `freq_end` (float): Ending frequency in Hz
- `points` (int): Number of measurement points
- `amplitude` (float): AC signal amplitude in mV
- `integration_time` (float, optional): Integration time per point in seconds
- `dc_bias` (float, optional): DC bias voltage in mV
- `temperature` (float, optional): Sample temperature in °C

**Returns:**
- `ImpedanceData`: Object containing measurement results

##### `plot_nyquist()`

Generate a Nyquist plot from impedance data.

```python
analyzer.plot_nyquist(data, figure_size=(10, 8), save_path='nyquist.png')
```

**Parameters:**
- `data` (ImpedanceData): Impedance data to plot
- `figure_size` (tuple, optional): Size of the figure in inches
- `title` (str, optional): Plot title
- `show_grid` (bool, optional): Whether to show grid lines
- `save_path` (str, optional): Path to save the plot

**Returns:**
- `matplotlib.figure.Figure`: The generated figure object

##### `export_data()`

Export impedance data to a file.

```python
analyzer.export_data(data, "results.csv", format="csv")
```

**Parameters:**
- `data` (ImpedanceData): Data to export
- `filename` (str): Output filename
- `format` (str, optional): Export format ('csv', 'json', 'excel')
- `include_metadata` (bool, optional): Whether to include measurement metadata

**Returns:**
- `bool`: True if export successful, False otherwise

### `BatteryHealth`

Class for analyzing battery health and degradation.

```python
from battery_analyzer import BatteryHealth
```

#### Constructor

```python
health_analyzer = BatteryHealth(model='lithium_ion_18650')
```

**Parameters:**
- `model` (str): Battery model ('lithium_ion_18650', 'lithium_polymer', 'ni_mh', etc.)
- `temperature` (float, optional): Reference temperature in °C
- `aging_model` (str, optional): Aging model to use ('calendar', 'cycle', 'combined')

#### Methods

##### `load_data()`

Load impedance data from a file or measurement.

```python
health_analyzer.load_data("impedance_data.csv")
```

**Parameters:**
- `source` (str or ImpedanceData): Filename or data object
- `format` (str, optional): File format if source is a filename

**Returns:**
- `bool`: True if loading successful, False otherwise

##### `fit_circuit_model()`

Fit impedance data to an equivalent circuit model.

```python
params = health_analyzer.fit_circuit_model('randles')
```

**Parameters:**
- `model` (str): Model name ('randles', 'modified_randles', 'transmission_line', etc.)
- `initial_params` (dict, optional): Initial parameter guesses
- `constrain_params` (dict, optional): Parameter constraints
- `max_iterations` (int, optional): Maximum fitting iterations

**Returns:**
- `dict`: Fitted parameters with confidence intervals

##### `calculate_soh()`

Calculate State of Health (SOH) from impedance data.

```python
soh = health_analyzer.calculate_soh(method='resistance')
```

**Parameters:**
- `method` (str, optional): Calculation method ('resistance', 'capacitance', 'diffusion', 'ml')
- `reference` (dict, optional): Reference values for new battery
- `temperature_compensate` (bool, optional): Whether to compensate for temperature effects

**Returns:**
- `float`: State of Health value (0-100%)

##### `predict_rul()`

Predict Remaining Useful Life (RUL).

```python
rul = health_analyzer.predict_rul(cycles=100, depth=0.8)
```

**Parameters:**
- `cycles` (int, optional): Number of previous cycles
- `depth` (float, optional): Typical depth of discharge
- `temperature` (float, optional): Operating temperature
- `current_rate` (float, optional): Typical C-rate

**Returns:**
- `dict`: Dictionary containing RUL estimate and confidence interval

##### `generate_report()`

Generate a comprehensive health report.

```python
health_analyzer.generate_report("battery_health_report.pdf")
```

**Parameters:**
- `filename` (str): Output filename
- `format` (str, optional): Report format ('pdf', 'html', 'json')
- `include_plots` (bool, optional): Whether to include plots
- `detail_level` (str, optional): Level of detail ('basic', 'standard', 'detailed')

**Returns:**
- `bool`: True if report generation successful, False otherwise

### `CircuitModels`

Class containing equivalent circuit models for fitting.

```python
from battery_analyzer import CircuitModels
```

#### Static Methods

##### `get_model()`

Get an equivalent circuit model function.

```python
model_func = CircuitModels.get_model('randles')
```

**Parameters:**
- `name` (str): Model name

**Returns:**
- `callable`: Function that calculates impedance for the model

##### `list_models()`

List available circuit models.

```python
models = CircuitModels.list_models()
```

**Returns:**
- `list`: List of available model names

## Data Structures

### `ImpedanceData`

Container for impedance measurement data.

**Attributes:**
- `frequency` (numpy.ndarray): Frequency points
- `impedance` (numpy.ndarray): Complex impedance values
- `phase` (numpy.ndarray): Phase angle values
- `magnitude` (numpy.ndarray): Impedance magnitude values
- `real` (numpy.ndarray): Real part of impedance
- `imag` (numpy.ndarray): Imaginary part of impedance
- `metadata` (dict): Measurement metadata

### `BatteryModel`

Representation of a battery model including parameters.

**Attributes:**
- `name` (str): Model name
- `parameters` (dict): Model parameters
- `circuit` (str): Circuit description
- `constraints` (dict): Parameter constraints

## Utility Functions

### `data_processing`

Module containing data processing utilities.

```python
from battery_analyzer import data_processing
```

#### Functions

##### `smooth_data()`

Apply smoothing to noisy data.

```python
smooth_impedance = data_processing.smooth_data(raw_data, window_size=5)
```

##### `remove_outliers()`

Remove outlier points from impedance data.

```python
clean_data = data_processing.remove_outliers(data, threshold=3.0)
```

##### `drift_correct()`

Correct for measurement drift.

```python
corrected_data = data_processing.drift_correct(data, reference_data)
```

### `visualization`

Module containing visualization utilities.

```python
from battery_analyzer import visualization
```

#### Functions

##### `plot_bode()`

Generate a Bode plot.

```python
fig = visualization.plot_bode(data)
```

##### `plot_cole_cole()`

Generate a Cole-Cole plot.

```python
fig = visualization.plot_cole_cole(data)
```

## Error Handling

The Battery Impedance Analyzer library uses exception classes derived from `BatteryAnalyzerError`.

### Common Exceptions

- `ConnectionError`: Hardware connection error
- `MeasurementError`: Error during measurement
- `FittingError`: Error during model fitting
- `DataProcessingError`: Error processing data
- `ValidationError`: Input validation error

Example:

```python
try:
    analyzer.connect(port='/dev/ttyUSB0')
except battery_analyzer.ConnectionError as e:
    print(f"Connection failed: {e}")
```

## Constants

The library provides a set of useful constants in the `constants` module.

```python
from battery_analyzer import constants
```

Examples:
- `constants.ROOM_TEMP`: Standard room temperature (25°C)
- `constants.DEFAULT_FREQUENCIES`: Default frequency range
- `constants.BATTERY_MODELS`: Dictionary of common battery models and specifications

---

*Battery Impedance Analyzer - Advancing battery analysis through integrated thermal-electrical impedance spectroscopy*