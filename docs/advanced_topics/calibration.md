# Calibration

GEB uses a Snakemake-based calibration workflow powered by the DEAP (Distributed Evolutionary Algorithms in Python) library. The workflow supports multi-objective optimization by allowing users to define multiple calibration targets (e.g., hydrology, socioeconomic) and multiple subtargets within each target.

## Configuration

Calibration is configured within the `calibration` section of your `model.yml`. You can define multiple calibration experiments (e.g. `hydrology` or `crops`) to separate different optimization goals.

### Calibration targets

Each target defines its own time period, evaluation methods, and optimization parameters. An example for calibration of hydrology is given here:

```yaml
calibration:
  hydrology:
    spinup_time: 1990-01-01
    start_time: 2000-01-01
    end_time: 2023-12-31
    
    # Define what metrics to optimize
    calibration_targets:
      KGE_discharge:
        method: hydrology.evaluate_discharge
        metric: KGE
        weight: 1.0
        
    # DEAP Evolutionary Algorithm settings
    DEAP:
      ngen: 10          # Number of generations
      mu: 20            # Population size
      lambda_: 10       # Offspring size per generation
      crossover_prob: 0.7
      mutation_prob: 0.3
      blend_alpha: 0.15
      gaussian_sigma: 0.3
      gaussian_indpb: 0.3
      
    # Parameter configuration
    parameters:
      mannings_n_multiplier:
          variable: parameters.mannings_n_multiplier # Path in model config
          min: 0.1
          max: 10.0
```

### 2. Integration with `geb evaluate`

The `calibration_targets` section directly maps optimization goals to the GEB evaluation subsystem:

- **method**: This corresponds to a command that can be run via `geb evaluate <method>`. Any evaluation method registered in the GEB CLI (e.g., `hydrology.evaluate_discharge`) AND which returns a dictionary with one ore multiple scores can be used as a calibration target.
- **metric**: Because `geb evaluate` returns a dictionary of metrics, this field specifies which key from that dictionary should be used for the optimization (e.g., `KGE`, `NSE`, `R`, or any custom metric returned by the method).
- **weight**: Used for multi-objective weighting. 

## Running Calibration

Calibration is executed via the `geb workflow` command. You must specify the calibration target name (e.g., `hydrology`).

```bash
geb workflow calibrate hydrology --cores 8
```

The workflow will:
1. Initialize and build a base model.
2. Generate an initial population of individuals.
3. For each generation:
    - Create individual model directories in `calibration/{target}/{gen}_{ind}/`.
    - Apply parameters and run simulations.
    - Call `geb evaluate <method>` for each target, capturing the results.
    - Calculate weighted fitness by combining the targets and their weights.
    - Select the best individuals and breed the next generation using DEAP.
4. Summary results and a Pareto front will be generated upon completion.

## Outputs

All results are stored in the relative workspace path defined by the workflow:

- **`calibration_{track}_complete.done`**: A marker file indicating completion.
- **`calibration_{track}_pareto_front.yml`**: Contains the best individuals (the Pareto front) found across all generations.
- **`generation_{gen}_{track}_summary.yml`**: Statistics (mean, max, min fitness) for each generation.
- **`calibration/{track}/{gen}_{ind}/fitness.yml`**: Detailed fitness breakdown for a specific individual run.
