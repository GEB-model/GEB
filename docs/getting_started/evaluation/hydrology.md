# Hydrology evaluation

Evaluate your hydrological model results by comparing simulated discharge with gauging station observations and analyzing water balance components.

## Overview

The hydrology evaluation tools help you:

- Compare simulated vs observed discharge at gauging stations
- Calculate performance metrics (KGE, NSE, R)
- Visualize spatial discharge patterns
- Analyze water balance components
- Identify model strengths and weaknesses

## Discharge evaluation

### Basic usage

Evaluate discharge against observations:

```python
model.evaluate.hydrology.evaluate_discharge(
    run_name="default",
    spinup_name="spinup",
    include_spinup=False,
    include_yearly_plots=True,
    correct_Q_obs=False
)
```

### Parameters

| Parameter | Description | Default |
| --- | --- | --- |
| `run_name` | Name of the simulation run to evaluate | `"default"` |
| `spinup_name` | Name of the spinup run | `"spinup"` |
| `include_spinup` | Include spinup period in evaluation | `False` |
| `include_yearly_plots` | Create plots for each year | `True` |
| `correct_Q_obs` | Correct observed discharge for upstream area differences | `False` |

### What it does

The evaluation process:

1. Loads observed discharge from gauging stations
2. Extracts simulated discharge at station locations
3. Calculates performance metrics for each station
4. Creates timeseries plots comparing observed vs simulated
5. Generates an interactive map showing station performance
6. Saves evaluation metrics to Excel and GeoParquet files

### Performance metrics

Three metrics are calculated for each station:

- **KGE** (Kling-Gupta Efficiency): Overall model performance (-∞ to 1, perfect = 1)
- **NSE** (Nash-Sutcliffe Efficiency): How well model predicts observations (-∞ to 1, perfect = 1)
- **R** (Correlation): Linear relationship between simulated and observed (0 to 1, perfect = 1)

### Outputs

Results are saved to `output/evaluate/discharge/`:

- `evaluation_results/evaluation_metrics.xlsx`: Performance metrics for all stations
- `evaluation_results/evaluation_metrics.geoparquet`: Spatial file with metrics
- `plots/`: Timeseries plots for each station (overall and yearly if requested)
- `discharge_evaluation_map.html`: Interactive map with station performance

## Discharge visualization

Create a spatial map of mean discharge:

```python
model.evaluate.hydrology.plot_discharge(run_name="default")
```

Outputs:
- `mean_discharge_m3_per_s.zarr`: Mean discharge data (m³/s)
- `mean_discharge_m3_per_s.png`: Spatial visualization

## Skill score graphs

Create boxplots summarizing performance across all stations:

```python
model.evaluate.hydrology.skill_score_graphs(export=True)
```

Shows the distribution of KGE, NSE, and R values across all evaluated stations.

## Water balance

### Water circle

Visualize water balance components as a Sankey diagram:

```python
model.evaluate.hydrology.water_circle(
    run_name="default",
    spinup_name="spinup",
    include_spinup=False,
    export=True
)
```

Shows flows between precipitation, evaporation, runoff, and storage components.

### Detailed water balance

Calculate and plot all water balance components:

```python
model.evaluate.hydrology.water_balance(
    run_name="default",
    spinup_name="spinup",
    include_spinup=False,
    export=True
)
```

Analyzes inflows, outflows, and storage changes across the model domain.

## Required input data

For discharge evaluation, your model must have:

- Observed discharge data in the data catalog (`discharge/Q_obs`)
- Gauging station locations snapped to river network (`discharge/discharge_snapped_locations`)
- Simulated discharge output from a model run (`output/report/{run_name}/hydrology.routing/discharge_daily.zarr`)

## Interpreting results

### Good performance

- KGE > 0.5: Model captures main discharge patterns
- NSE > 0.5: Predictions are better than using mean observed value
- R > 0.7: Strong correlation between simulated and observed

### Common issues

- **Low KGE but high R**: Model timing is correct but magnitude is off (check calibration)
- **Negative NSE**: Model performs worse than using mean (check model setup)
- **High variation between stations**: Some areas may need region-specific parameters

## Code reference

::: geb.evaluate.hydrology.Hydrology

