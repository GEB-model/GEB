# Model evaluation

GEB includes evaluation tools to assess model performance against observations. After running a simulation, you can compare your results with measured data to validate and improve your model.

## Overview

The evaluation module provides methods for both the hydrological and hydrodynamic outputs of GEB. You can:

- Compare simulated discharge with gauging station observations (from the global GRDC dataset or custom stations)
- Visualize spatial patterns of model outputs (e.g., discharge grids)
- Generate water balance reports
- Evaluate flood extents against satellite observations
- Calculate performance metrics (KGE, NSE, R for discharge; CSI for flood extent) 

## Available evaluations

GEB supports evaluation of:

- **Hydrology**: Discharge at gauging stations, water balance components, spatial patterns
- **Floods**: Flood extent and depth against satellite observations
- **Meteorological forecasts**: Forecast skill for precipitation and temperature

## Basic usage

Run all evaluation methods:

```bash
geb evaluate
```

This runs the default methods: `plot_discharge`, `evaluate_discharge`, `evaluate_hydrodynamics`, `water_balance`, `water_circle`

Run specific methods only:

```bash
geb evaluate --methods "plot_discharge,evaluate_discharge"
```

## All available methods
Below, you can find an overview of all the methods currently supported by the evaluation module:

| Method | Description |
| --- | --- |
| `hydrology.plot_discharge` | Creates spatial map of mean discharge |
| `hydrology.evaluate_discharge` | Compares simulated vs observed discharge at stations |
| `hydrology.skill_score_graphs` | Creates boxplots of performance metrics |
| `hydrology.water_circle` | Visualizes water balance components |
| `hydrology.water_balance` | Calculates detailed water balance |
| `hydrology.evaluate_hydrodynamics` | Evaluates flood extents against observations |

## Command options
Different options can be appended to the geb evaluate command, such as whether or not to include the hydrological spinup period, or to automatically correct the discharg e
```bash
geb evaluate --run-name default --spinup-name spinup --include-spinup --include-yearly-plots
```

| Option | Description | Default |
| --- | --- | --- |
| `--methods` | Comma-separated list of methods to run | All methods |
| `--run-name` | Name of the simulation run to evaluate | `default` |
| `--spinup-name` | Name of the spinup run | `spinup` |
| `--include-spinup` | Include spinup period in evaluation | `False` |
| `--include-yearly-plots` | Create plots for each year | `False` |
| `--correct-q-obs` | Correct observed discharge for area differences | `False` |

## Output location

Evaluation results are saved to `output/evaluate/` in your model folder. This includes:

- Performance metrics (Excel and GeoParquet files)
- Plots and maps (PNG files)
- Interactive dashboards (HTML files)
- Processed data (Zarr files)

## Next steps

Learn more about specific evaluation types:

- [Hydrology evaluation](hydrology.md): Discharge comparison and metrics
- [Flood evaluation](flood.md): Flood extent validation

## Code

::: geb.evaluate

