# Flood evaluation

Evaluate simulated flood extents by comparing them with satellite observations or other flood mapping data.

## Overview
The hydrodynamics evaluation tools help you:

- Compare simulated vs observed flood extents
- Calculate spatial performance metrics (Hit Rate, False Alarm Ratio, CSI)
- Visualize flood extent accuracy 
- Analyze model performance across different flood events
- Generate diagnostic plots for each event and forecast initialization

## Basic usage
Evaluate flood extents against observations:

```bash
geb evaluate --methods "evaluate_hydrodynamics" --run-name default
```

## Parameters

| Parameter | Description | Default |
| --- | --- | --- |
| `run_name` | Name of the simulation run to evaluate | `"default"` |

## What it does
The evaluation process:

- Reads all flood events from your model config file
- Loads observed flood extents from the config file. The file should have the same name as your flood event and be saved as a zarr (i.e. start time - end time.zarr)
- Loads corresponding simulated flood maps from model output
- Resamples observations to match model resolution if needed
- Calculates performance metrics for each event
- Creates visualizations comparing observed vs simulated extents
- Saves all results to structured output folders

## Performance metrics
Three spatial metrics are calculated for each flood map:

**Hit Rate (HR)**: Percentage of observed flooded pixels correctly predicted (0 to 1, perfect = 1)

$$\text{Hit Rate} = \frac{\text{Hits}}{\text{Hits} + \text{Misses}}$$

**False Alarm Ratio (FAR)**: Percentage of predicted floods that did not occur (0 to 1, perfect = 0)

$$\text{FAR} = \frac{\text{False Alarms}}{\text{False Alarms} + \text{Hits}}$$

**Critical Success Index (CSI)**: Overall accuracy accounting for hits, misses, and false alarms (0 to 1, perfect = 1)

$$\text{CSI} = \frac{\text{Hits}}{\text{Hits} + \text{False Alarms} + \text{Misses}}$$

## Outputs
Results are saved to output/evaluate/hydrodynamics/{event_name}/

For each event two files are generated:

- **start time - end time_performance_metrics.txt**: gives an overview of Hit Rate, False Alarm Ratio, Critical Success Index, number of flooded pixels and flooded area
- **start time - end time_validation_floodextent_plot.png**: figure plotting the hits (green), false alarms (orange) and misses (red), together with the catchment outline and OSM map

## Required input data
For hydrodynamics evaluation, your model must have:

   1. **Flood event configuration** in your model config file with path to observation file(s) (see example below):

   ```yaml
   hazards:
     floods:
       events:
         - start_time: "2021-07-12 09:00:00"
           end_time: "2021-07-20 09:00:00"
       observation_files:
         - "path/to/20210712T090000 - 20210720T090000.zarr.zarr"
   ```
  2. **Simulated flood maps** located in output/evaluate/hydrodynamics/{event_name}/

## Interpreting results 
A CSI above 0.7 is considered good model performance [@bernhofen2018first]




