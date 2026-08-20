# Hydrology evaluation

Evaluate hydrological model performance by comparing simulated discharge with gauging station observations and analyzing the water balance.

## Overview

The hydrology evaluation module provides comprehensive tools to assess model performance:

| Method | Purpose | Output |
| --- | --- | --- |
| `evaluate_discharge` | Compare simulated vs observed discharge at gauging stations | Performance metrics (KGE, NSE, R), timeseries plots, interactive maps |
| `export_discharge_publication_data` | Collect observation-free station simulations for publication | Raw station Parquet files, station catalogue, evaluation spreadsheet, README |
| `plot_discharge_characteristics` | Relate KGE and its components to catchment characteristics from GRDC-Caravan | Combined heatmap–scatterplot figure, 32-characteristic scatterplot figure, and association table |
| `plot_discharge` | Visualize spatial patterns of mean discharge | Spatial maps showing discharge distribution |
| `skill_score_graphs` | Summarize performance across all stations in the model domain | Boxplots of KGE, NSE, R distributions |
| `water_circle` | Visualize water balance as flow diagram | Interactive Sankey diagram of water fluxes |
| `water_balance` | Analyze detailed water balance components | Yearly water balance tables and plots |

To use these for the evaluation, you can run them using geb evaluate. Below, you see an example for the evaluate_discharge methodology: 

```bash
geb evaluate hydrology.evaluate_discharge --run-name default
```

For more control, use additional options:

| Option | Description | Default |
| --- | --- | --- |
| `--run-name` | Name of the simulation run to evaluate | `default` |
| `--include-yearly-plots` | Create plots for each year | `True` |
| `--correct-discharge-observations` | Correct simulated discharge for upstream-area differences | `False` |
| `--create-plots` | Create station, dashboard, and skill-score plots | `True` |
| `--include-return-period-plots` | Create detailed return-period plots | `False` |

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

The main metrics calculated for each station are:

- **KGE** (Kling-Gupta Efficiency): Overall model performance (-∞ to 1, perfect = 1)
- **NSE** (Nash-Sutcliffe Efficiency): How well model predicts observations (-∞ to 1, perfect = 1)
- **KGE components**: Correlation, mean-flow bias, and variability ratios
- **R2**: Squared Pearson correlation (0 to 1, perfect = 1)
- **RMSE and RRMSE**: Absolute and variability-normalized errors

### Outputs

The discharge evaluation results are saved to
`output/<run_name>/evaluate/hydrology/evaluate_discharge/`:

**Overall evaluation results** (`evaluation_results/`):
- `evaluation_metrics.xlsx`: Performance metrics for all stations.
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

For discharge evaluation, your model must have been built and run. The following files must be available:

- Observed discharge data in the data catalog (`discharge/Q_obs`)
- Gauging station locations snapped to river network (`discharge/discharge_snapped_locations`)
- Per-station simulated discharge reports from the model run
  (`output/<run_name>/report/hydrology.routing/discharge_hourly_m3_per_s_<station_id>.parquet`)

### Mean-flow benchmark

Use the observed mean flow as a simple benchmark for discharge evaluation:

- NSE > 0: The simulation improves upon using the observed mean flow as the prediction.
- KGE > -0.41: The simulation improves upon the observed mean-flow benchmark.
- R describes correlation between simulated and observed flow, but is not itself a mean-flow benchmark score.

### Catchment-characteristic explanation

Run the GRDC-Caravan analysis after discharge evaluation:

```bash
geb evaluate hydrology.plot_discharge_characteristics --run-name default
```

The combined figure reports Spearman associations with correlation `r`,
mean-flow ratio `beta`, variability ratio `alpha`, and the original KGE.

### Publication-ready station simulations

After running discharge evaluation, create a self-contained folder for a later
Zenodo deposition:

```bash
geb evaluate hydrology.export_discharge_publication_data --run-name default
```

The resulting `evaluate_discharge/publication_data/` folder contains one raw
hourly reporter Parquet file per evaluated station, a CSV station catalogue
with the station identity and source plus original and snapped coordinates, the
evaluation spreadsheet, and a README. Simulations are raw GEB reporter values
in m3/s: no observation-based upstream-area correction or daily resampling is
applied.

Observed discharge is intentionally excluded because source-specific licences
can restrict redistribution. For GRDC stations, users should obtain the
observations from the GRDC Data Portal.

### External skill-score comparisons

External daily-discharge comparisons are optional and use local files only. Put
either or both of the following fixed filenames in `external_evaluation_data/`
inside the top-level model folder. For merged multi-model evaluations, this
folder is alongside the `merged/` and individual cluster folders:

- `Utrecht_1KM_daily_discharge.csv`, downloaded from [Zenodo record 6390219](https://zenodo.org/records/6390219).
- `google_streamflow_metrics.tgz`, downloaded as `metrics.tgz` from [Zenodo record 10397664](https://zenodo.org/records/10397664) and renamed. GEB reads both Google Streamflow and GloFAS scores from this archive.

GEB reads these local files directly and does not download external evaluation data automatically. Other filenames in this folder are ignored.

## Water balance

The water balance evaluation analyzes inflows, outflows, and storage changes across the model domain to verify water conservation and understand hydrological processes.

### Water circle

Visualize water balance components as a Sankey diagram:

```bash
geb evaluate hydrology.plot_water_circle --run-name default
```

Shows flows between precipitation, evaporation, runoff, and storage components.

### Detailed water balance

Calculate and plot all water balance components:

```bash
geb evaluate hydrology.plot_water_balance --run-name default
```
Analyzes inflows, outflows, and storage changes across the model domain to verify water conservation.

## Code reference

::: geb.evaluate.hydrology
