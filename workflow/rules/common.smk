"""Common rules and functions shared across GEB workflows.

This module contains reusable rules for:
- Base model initialization and building
- Parameter file generation
- Individual model execution (init, spinup, run, evaluate)
"""

import os
from pathlib import Path

# Common configuration
REGION = config.get("REGION", "geul")
BASE_DIR = config.get("BASE_DIR", f"tests/tmp/snakemake/{REGION}")

# Handle empty BASE_DIR (current directory case)
if BASE_DIR:
    RUNS_DIR = f"{BASE_DIR}/runs"
else:
    RUNS_DIR = "runs"

# Initialize the base model (done once)
rule init_base:
    output:
        touch(f"{BASE_DIR}/base_init.done" if BASE_DIR else "base_init.done")
    log:
        f"{BASE_DIR}/logs/base_init.log" if BASE_DIR else "logs/base_init.log"
    shell:
        """
        mkdir -p $(dirname {log})
        uv run geb init --overwrite 2>&1 | tee {log}
        """

# Build the base model
rule build_base:
    input:
        f"{BASE_DIR}/base_init.done" if BASE_DIR else "base_init.done"
    output:
        touch(f"{BASE_DIR}/base_build.done" if BASE_DIR else "base_build.done")
    log:
        f"{BASE_DIR}/logs/base_build.log" if BASE_DIR else "logs/base_build.log"
    shell:
        """
        uv run geb build 2>&1 | tee {log}
        """

# Initialize individual run directory
rule init_individual:
    input:
        params=f"{RUNS_DIR}/{{gen}}_{{ind}}/parameters.yml",
        base_build=f"{BASE_DIR}/base_build.done" if BASE_DIR else "base_build.done"
    output:
        touch(f"{RUNS_DIR}/{{gen}}_{{ind}}/init.done")
    log:
        f"{RUNS_DIR}/{{gen}}_{{ind}}/logs/init.log"
    run:
        import os
        import yaml
        
        # Create run directory and logs
        run_dir = f"{RUNS_DIR}/{wildcards.gen}_{wildcards.ind}"
        os.makedirs(f"{run_dir}/logs", exist_ok=True)
        
        # Create empty build.yml and update.yml
        with open(f"{run_dir}/build.yml", "w") as f:
            f.write("# Empty build configuration for individual run\n")
        
        with open(f"{run_dir}/update.yml", "w") as f:
            f.write("# Empty update configuration for individual run\n")
        
        # Log completion
        with open(log[0], "w") as f:
            f.write(f"Initialized individual {wildcards.gen}_{wildcards.ind}\n")

# Alter individual model based on base model
rule alter_individual:
    input:
        f"{RUNS_DIR}/{{gen}}_{{ind}}/init.done"
    output:
        touch(f"{RUNS_DIR}/{{gen}}_{{ind}}/altered.done")
    log:
        f"{RUNS_DIR}/{{gen}}_{{ind}}/logs/alter.log"
    shell:
        """
        uv run geb alter --from-model ../../. -wd {RUNS_DIR}/{wildcards.gen}_{wildcards.ind} 2>&1 | tee {log}
        """

# Apply parameters to individual config
rule set_individual_parameters:
    input:
        altered_done=f"{RUNS_DIR}/{{gen}}_{{ind}}/altered.done",
        params=f"{RUNS_DIR}/{{gen}}_{{ind}}/parameters.yml"
    output:
        touch(f"{RUNS_DIR}/{{gen}}_{{ind}}/params_set.done")
    log:
        f"{RUNS_DIR}/{{gen}}_{{ind}}/logs/set_params.log"
    run:
        import yaml
        import os
        
        # Load parameters
        with open(input.params, "r") as f:
            params_data = yaml.safe_load(f)
        
        actual_params = params_data["parameters"]
        
        # Get the directory for this individual
        run_dir = os.path.join(RUNS_DIR, f"{wildcards.gen}_{wildcards.ind}")
        
        # Build geb set command for each parameter using working directory
        commands = []
        for param_name, param_value in actual_params.items():
            commands.append(f"uv run geb set -c model.yml --working-directory {run_dir} {param_name}={param_value}")

        # Set reporting to null
        commands.append(f"uv run geb set -c model.yml --working-directory {run_dir} report=null")

        # Overwrite reporting but now with specific parameters
        commands.append(f"uv run geb set -c model.yml --working-directory {run_dir} report._discharge_stations=true")
        
        # Execute all set commands
        import subprocess
        with open(log[0], "w") as log_file:
            for cmd in commands:
                log_file.write(f"Running: {cmd}\n")
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                log_file.write(result.stdout)
                log_file.write(result.stderr)
                if result.returncode != 0:
                    raise RuntimeError(f"Failed to set parameter: {cmd}")

# Run spinup for an individual
rule spinup_individual:
    input:
        f"{RUNS_DIR}/{{gen}}_{{ind}}/params_set.done"
    output:
        touch(f"{RUNS_DIR}/{{gen}}_{{ind}}/spinup.done")
    log:
        f"{RUNS_DIR}/{{gen}}_{{ind}}/logs/spinup.log"
    shell:
        """
        uv run geb spinup -wd {RUNS_DIR}/{wildcards.gen}_{wildcards.ind} 2>&1 | tee {log}
        """

# Run main simulation for an individual
rule run_individual:
    input:
        f"{RUNS_DIR}/{{gen}}_{{ind}}/spinup.done"
    output:
        touch(f"{RUNS_DIR}/{{gen}}_{{ind}}/run.done")
    log:
        f"{RUNS_DIR}/{{gen}}_{{ind}}/logs/run.log"
    shell:
        """
        uv run geb run -wd {RUNS_DIR}/{wildcards.gen}_{wildcards.ind} 2>&1 | tee {log}
        """

# Evaluate an individual
rule evaluate_individual:
    input:
        f"{RUNS_DIR}/{{gen}}_{{ind}}/run.done"
    output:
        f"{RUNS_DIR}/{{gen}}_{{ind}}/fitness.yml"
    log:
        f"{RUNS_DIR}/{{gen}}_{{ind}}/logs/evaluate.log"
    shell:
        """
        # Use geb evaluate to compute metrics
        uv run geb evaluate -wd {RUNS_DIR}/{wildcards.gen}_{wildcards.ind} 2>&1 | tee {log}
        
        # TODO: Extract fitness score from evaluation results
        # For now, create a dummy fitness file
        python -c "import yaml; yaml.dump({{'fitness': [0.5], 'metrics': {{'KGE': 0.5}}}}, open('{output}', 'w'))"
        
        # Clean up simulation_root directory to save disk space
        rm -rf {RUNS_DIR}/{wildcards.gen}_{wildcards.ind}/simulation_root
        """
