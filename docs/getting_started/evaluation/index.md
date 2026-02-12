# Model evaluation

GEB includes evaluation tools to assess model performance against observations. After running a simulation, you can compare your results with measured data to validate and improve your model.

## Overview

The evaluation module provides methods to:

- Compare simulated discharge with gauging station observations
- Visualize spatial patterns of model outputs
- Calculate performance metrics (KGE, NSE, R)
- Generate water balance reports
- Evaluate flood extents against satellite observations

## Available evaluations

GEB supports evaluation of:

- **Hydrology**: Discharge at gauging stations, water balance components, spatial patterns
- **Floods**: Flood extent and depth against satellite observations
- **Meteorological forecasts**: Forecast skill for precipitation and temperature

## Basic usage

Run evaluation from Python:

```python
from geb.model import GEBModel

model = GEBModel("model.yml")
model.evaluate.run(
    methods=["hydrology.evaluate_discharge"],
    run_name="default",
    spinup_name="spinup"
)
```

Or evaluate specific components:

```python
# Evaluate discharge only
model.evaluate.hydrology.evaluate_discharge(run_name="default")

# Create water balance plots
model.evaluate.hydrology.water_balance(run_name="default")

# Evaluate flood extents
model.evaluate.hydrology.evaluate_hydrodynamics(run_name="default")
```

## Output location

Evaluation results are saved to `output/evaluate/` in your model folder. This includes:

- Performance metrics (Excel and GeoParquet files)
- Plots and maps (PNG files)
- Interactive dashboards (HTML files)
- Processed data (Zarr files)

## Available methods

Common evaluation methods:

| Method | Description |
| --- | --- |
| `hydrology.plot_discharge` | Creates spatial map of mean discharge |
| `hydrology.evaluate_discharge` | Compares simulated vs observed discharge at stations |
| `hydrology.skill_score_graphs` | Creates boxplots of performance metrics |
| `hydrology.water_circle` | Visualizes water balance components |
| `hydrology.water_balance` | Calculates detailed water balance |
| `hydrology.evaluate_hydrodynamics` | Evaluates flood extents against observations |

## Next steps

Learn more about specific evaluation types:

- [Hydrology evaluation](hydrology.md): Discharge comparison and metrics
- [Flood evaluation](flood.md): Flood extent validation

## Code

::: geb.evaluate

