# Hydrology evaluation

Evaluate hydrological model performance by comparing simulated discharge with gauging station observations and analyzing the water balance.

## Overview

The hydrology evaluation module provides comprehensive tools to assess model performance:

| Method | Purpose | Output |
| --- | --- | --- |
| `evaluate_discharge` | Compare simulated vs observed discharge at gauging stations | Performance metrics (KGE, NSE, R), timeseries plots, interactive maps |
| `plot_discharge` | Visualize spatial patterns of mean discharge | Spatial maps showing discharge distribution |
| `skill_score_graphs` | Summarize performance across all stations in the model domain | Boxplots of KGE, NSE, R distributions |
| `water_circle` | Visualize water balance as flow diagram | Interactive Sankey diagram of water fluxes |
| `water_balance` | Analyze detailed water balance components | Yearly water balance tables and plots |

To use these for the evaluation, you can run them using geb evaluate. Below, you see an example for the evaluate_discharge methodology: 

```bash
geb evaluate --method hydrology.evaluate_discharge --run-name default
```

For more control, use additional options:

| Option | Description | Default |
| --- | --- | --- |
| `--run-name` | Name of the simulation run to evaluate | `default` |
| `--spinup-name` | Name of the spinup run | `spinup` |
| `--include-spinup` | Include spinup period in evaluation | `False` |
| `--include-yearly-plots` | Create plots for each year | `False` |
| `--correct-q-obs` | Correct observed discharge for upstream area differences | `False` |

## Discharge evaluation

The discharge evaluation compares simulated discharge from GEB with observed discharge data from gauging stations (GRDC global dataset or custom stations). Performance metrics are calculated for each station and visualized in plots and interactive maps.

### What it does

The evaluation process:

1. Loads observed discharge from gauging stations
2. Extracts simulated discharge at station locations
3. Calculates performance metrics for each station
4. Creates timeseries and scatter plots comparing observed vs simulated
5. Generates an interactive map showing station performance
6. Saves evaluation metrics to Excel and GeoParquet files

### Performance metrics

Three metrics are calculated for each station:

- **KGE** (Kling-Gupta Efficiency): Overall model performance (-∞ to 1, perfect = 1)
- **NSE** (Nash-Sutcliffe Efficiency): How well model predicts observations (-∞ to 1, perfect = 1)
- **R** (Correlation): Linear relationship between simulated and observed (0 to 1, perfect = 1)

### Outputs

The discharge evaluation results are saved to `output/evaluate/discharge/`:

**Overall evaluation results** (`evaluation_results/`):
- `evaluation_metrics.xlsx`: Performance metrics (KGE, NSE, R) for all stations with coordinates
- `evaluation_metrics.geoparquet`: Same metrics in geospatial format for GIS analysis
- `discharge_evaluation_metrics.png`: Map showing spatial distribution of metrics
- `discharge_evaluation_map.html`: Interactive Folium map to explore station performance

**Station specific plots** (`plots/`):
- `timeseries_plot_{station_id}.png`: Time series comparing observed vs simulated discharge
- `scatter_plot_{station_id}.png`: Scatter plots showing correlation between observed and simulated
- `return_period_plot_{station_id}.png`: GPD-POT return-period comparison (observed vs simulated)
- `shape_metrics_plot_{station_id}.png`: Skewness and kurtosis comparison (observed vs simulated)
- Yearly plots are created when `--include-yearly-plots` is enabled

**Outflow-only plots** (`plots/outflow/`):
- `river_outflow_hourly_m3_per_s_{river_id}.png`: Line plot of simulated river outflow discharge (m3/s) for each exported outflow location
- `river_outflow_hourly_m3_per_s_{river_id}_return_period.png`: GPD-POT return-period plot for each exported outflow location

The evaluation creates an interactive dashboard showing performance metrics across all stations (INSERT IMAGE). 

### Required input data

For discharge evaluation, your model must have been build and run, in which the following files are made in your model input folder: 

- Observed discharge data in the data catalog (`discharge/Q_obs`)
- Gauging station locations snapped to river network (`discharge/discharge_snapped_locations`)
- Simulated discharge output from model run (`output/report/{run_name}/hydrology.routing/discharge_daily.zarr`)

### Interpreting results

**Good performance:**

- KGE > 0.5: Model captures main discharge patterns
- NSE > 0.5: Predictions are better than using mean observed value
- R > 0.7: Strong correlation between simulated and observed

**Common issues:**

- **Low KGE but high R**: Model timing is correct but magnitude is off (check calibration parameters)
- **Negative NSE**: Model performs worse than using mean (check model setup and forcing data)
- **High variation between stations**: Some areas may need region-specific parameters or better forcing data

## Water balance

The water balance evaluation analyzes inflows, outflows, and storage changes across the model domain to verify water conservation and understand hydrological processes.

### Water circle

Visualize water balance components as a Sankey diagram:

```bash
geb evaluate --method hydrology.water_circle --run-name default
```

Shows flows between precipitation, evaporation, runoff, and storage components.

### Detailed water balance

Calculate and plot all water balance components:

```bash
geb evaluate --method hydrology.water_balance --run-name default 
```
Analyzes inflows, outflows, and storage changes across the model domain to verify water conservation.

## Code reference

::: geb.evaluate.hydrology.Hydrology

