"""Common rules and functions shared across GEB workflows."""

import subprocess
import os
import sys
from pathlib import Path
import yaml
import pandas as pd
from geb.runner import parse_config

# Common configuration - can be customized by workflows
RUNS_DIR = config.get("RUNS_DIR", "runs")

def run_command(cmd: str, log_path: str, error_msg: str = "GEB command failed") -> None:
    """Helper function to run a GEB command and log output.

    Args:
        cmd: The shell command to run.
        log_path: Path to the log file.
        error_msg: Error message to raise on failure.

    Raises:
        RuntimeError: If the command returns a non-zero exit code.
    """
    with open(log_path, "a") as log_file:
        log_file.write(f"\nRunning: {cmd}\n")
        print(f"\n[GEB] Running: {cmd}")
        
        # Override progress bar interval for cleaner logs in Snakemake
        env = os.environ.copy()
        env["GEB_OVERRIDE_PROGRESSBAR_DT"] = "10"
        
        with open(log_path, "a") as log_file:
            result = subprocess.run(
                cmd, 
                shell=True, 
                stdout=log_file, 
                stderr=subprocess.STDOUT, 
                env=env,
                text=True
            )
            
        if result.returncode != 0:
            raise RuntimeError(f"{error_msg}. Check logs at: {log_path}")

# Initialize the base model (done once)
rule init_base:
    output:
        touch("base_init.done")
    log:
        "logs/base_init.log"
    run:
        run_command("geb init --overwrite", log[0], "Failed to initialize base model")

# Build the base model
rule build_base:
    input:
        "base_init.done"
    output:
        touch("base_build.done")
    log:
        "logs/base_build.log"
    run:
        run_command("geb build --continue", log[0], "Failed to build base model")

# Initialize individual run directory
rule init_individual:
    input:
        params=RUNS_DIR + "/{gen}_{ind}/parameters.yml",
        base_build="base_build.done"
    output:
        touch(RUNS_DIR + "/{gen}_{ind}/init.done")
    log:
        RUNS_DIR + "/{gen}_{ind}/logs/init.log"
    run:
        # Create run directory and logs
        run_dir = Path(RUNS_DIR) / f"{wildcards.gen}_{wildcards.ind}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        
        # Create empty build.yml and update.yml
        with open(run_dir / "build.yml", "w") as f:
            f.write("# Empty build configuration for individual run\n")
        
        with open(run_dir / "update.yml", "w") as f:
            f.write("# Empty update configuration for individual run\n")
        
        # Log completion
        with open(log[0], "w") as f:
            f.write(f"Initialized individual {wildcards.gen}_{wildcards.ind}\n")

# Alter individual model based on base model
rule alter_individual:
    input:
        RUNS_DIR + "/{gen}_{ind}/init.done"
    output:
        touch(RUNS_DIR + "/{gen}_{ind}/altered.done")
    log:
        RUNS_DIR + "/{gen}_{ind}/logs/alter.log"
    run:
        run_dir = Path(RUNS_DIR) / f"{wildcards.gen}_{wildcards.ind}"
        cmd = f"geb alter --from-model ../../. -wd {run_dir}"
        run_command(cmd, log[0], f"Failed to alter model {wildcards.gen}_{wildcards.ind}")

# Apply parameters to individual config
rule set_individual_parameters:
    input:
        altered_done=RUNS_DIR + "/{gen}_{ind}/altered.done",
        parameters=RUNS_DIR + "/{gen}_{ind}/parameters.yml",
    output:
        touch(RUNS_DIR + "/{gen}_{ind}/params_set.done")
    log:
        RUNS_DIR + "/{gen}_{ind}/logs/set_params.log"
    run:
        # Load parameters
        with open(input.parameters, "r") as f:
            params_data = yaml.safe_load(f)
        
        actual_params = params_data["parameters"]
        
        # Get the directory for this individual
        run_dir = Path(RUNS_DIR) / f"{wildcards.gen}_{wildcards.ind}"
        
        # Build single geb set command with all parameters
        param_args = " ".join(f"{param_name}={param_value}" for param_name, param_value in actual_params.items())

        config = parse_config(run_dir / "model.yml")

        spinup_time = config["calibration"]["spinup_time"]
        start_time = config["calibration"]["start_time"]
        end_time = config["calibration"]["end_time"]
        
        datetime_args = f"general.spinup_time={spinup_time} general.start_time={start_time} general.end_time={end_time}"
        
        # We ensure discharge stations are reported for evaluation
        cmd = f"geb set -c model.yml --working-directory {run_dir} {param_args} {datetime_args} report=null report._discharge_stations+=true"
        
        run_command(cmd, log[0], f"Failed to set parameters for {wildcards.gen}_{wildcards.ind}")

# Run spinup for an individual
rule spinup_individual:
    input:
        RUNS_DIR + "/{gen}_{ind}/params_set.done"
    output:
        touch(RUNS_DIR + "/{gen}_{ind}/spinup.done")
    log:
        RUNS_DIR + "/{gen}_{ind}/logs/spinup.log"
    run:
        run_dir = Path(RUNS_DIR) / f"{wildcards.gen}_{wildcards.ind}"
        run_command(f"geb spinup -wd {run_dir}", log[0], f"Spinup failed for {wildcards.gen}_{wildcards.ind}")

# Run main simulation for an individual
rule run_individual:
    input:
        RUNS_DIR + "/{gen}_{ind}/spinup.done"
    output:
        touch(RUNS_DIR + "/{gen}_{ind}/run.done")
    log:
        RUNS_DIR + "/{gen}_{ind}/logs/run.log"
    run:
        run_dir = Path(RUNS_DIR) / f"{wildcards.gen}_{wildcards.ind}"
        run_command(f"geb run -wd {run_dir}", log[0], f"Main run failed for {wildcards.gen}_{wildcards.ind}")

# Evaluate an individual
rule evaluate_individual:
    input:
        RUNS_DIR + "/{gen}_{ind}/run.done"
    output:
        RUNS_DIR + "/{gen}_{ind}/fitness.yml"
    log:
        RUNS_DIR + "/{gen}_{ind}/logs/evaluate.log"
    run:
        run_dir = Path(RUNS_DIR) / f"{wildcards.gen}_{wildcards.ind}"
        
        # Run geb evaluate using helper
        cmd = f"geb evaluate --methods hydrology.evaluate_discharge -wd {run_dir}"
        run_command(cmd, log[0], f"Evaluation failed for {wildcards.gen}_{wildcards.ind}")

        # Extract mean KGE from the generated geoparquet
        eval_file = run_dir / "evaluate/discharge/evaluation_results/evaluation_metrics.geoparquet"
        
        if not eval_file.exists():
            raise FileNotFoundError(f"Evaluation metrics file not found: {eval_file}. Check the evaluation logs.")
            
        df = pd.read_parquet(eval_file)
        if df.empty:
            raise ValueError(f"Evaluation metrics file is empty: {eval_file}. No stations were evaluated.")
            
        # Use mean KGE as fitness. DEAP uses a list for multi-objective (even single value)
        fitness = [float(df["KGE"].mean())]
        metrics = {"KGE": fitness[0]}

        # Save fitness file
        with open(output[0], "w") as f:
            yaml.dump({"fitness": fitness, "metrics": metrics}, f)
