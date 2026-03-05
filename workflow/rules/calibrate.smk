"""Calibration-specific rules for GEB evolutionary algorithm optimization.

This module contains rules specific to calibration workflows using DEAP:
- Population generation and management
- Fitness aggregation and selection
- Offspring creation
- Pareto front computation

The calibration workflow organizes individual model runs in the format: 
{CALIBRATION_DIR}/{target}/{gen}_{ind}/
"""

import subprocess
import os
import sys
import random
import yaml
import numpy as np
import pandas as pd
import json
from deap import base, creator, tools, algorithms
import array
from pathlib import Path
from geb.runner import parse_config

# --- Configuration & Setup ---

def load_model_config():
    """Load the model.yml configuration file."""
    model_file = Path("model.yml")
    if not model_file.exists():
        raise FileNotFoundError(f"Model configuration file not found: {model_file}")
    
    with open(model_file, "r") as f:
        return yaml.safe_load(f)

MODEL_CONFIG = load_model_config()
CALIBRATION_TARGET = config.get("TARGET", None)

if CALIBRATION_TARGET is None:
    print("Error: calibration target must be specified for calibration workflows. For example 'geb workflow calibrate hydrology'")
    sys.exit(1)

# Ensure the calibration and target section exist
if "calibration" not in MODEL_CONFIG:
    raise ValueError("No 'calibration' section found in model.yml")
if CALIBRATION_TARGET not in MODEL_CONFIG["calibration"]:
    available = list(MODEL_CONFIG["calibration"].keys())
    raise KeyError(
        f"Calibration target '{CALIBRATION_TARGET}' not found in model.yml. "
        f"Available targets/sections: {available}"
    )

CALIBRATION_CONFIG = MODEL_CONFIG["calibration"][CALIBRATION_TARGET]
RUNS_DIR = config.get("RUNS_DIR", f"calibration/{CALIBRATION_TARGET}")

# DEAP Settings
NGEN = CALIBRATION_CONFIG["DEAP"]["ngen"]
MU = CALIBRATION_CONFIG["DEAP"]["mu"]
LAMBDA = CALIBRATION_CONFIG["DEAP"]["lambda_"]

# Parameters
PARAMETERS = {}
for param_name, param_config in CALIBRATION_CONFIG["parameters"].items():
    PARAMETERS[param_name] = {
        "min": param_config["min"],
        "max": param_config["max"],
        "variable": param_config["variable"]
    }
N_PARAMETERS = len(PARAMETERS)

# Target Weights
CALIBRATION_TARGETS = CALIBRATION_CONFIG["calibration_targets"]
weights = [float(target_info["weight"]) for target_info in CALIBRATION_TARGETS.values()]

# --- Helpers ---

def get_calibration_config(model_config: dict) -> dict:
    """Helper to get the calibration section for the current target."""
    return model_config.get("calibration", {}).get(CALIBRATION_TARGET, {})

def run_command(cmd: str, log_path: str, error_msg: str = "GEB command failed") -> str | None:
    """Helper function to run a GEB command, log output, and return captured stdout."""
    env = os.environ.copy()
    env["GEB_OVERRIDE_PROGRESSBAR_DT"] = "10"

    print(f"\n[GEB] Running: {cmd}")
    
    process = subprocess.run(
        cmd, 
        shell=True, 
        capture_output=True,
        env=env,
        text=True
    )
    
    with open(log_path, "a") as log_file:
        log_file.write(f"\n[{pd.Timestamp.now()}] Running: {cmd}\n")
        if process.stdout:
            log_file.write("--- STDOUT ---\n")
            log_file.write(process.stdout)
        if process.stderr:
            log_file.write("--- STDERR ---\n")
            log_file.write(process.stderr)
        log_file.write(f"Return code: {process.returncode}\n")
        
    if process.returncode != 0:
        raise RuntimeError(f"{error_msg}. Check logs at: {log_path}")

    return process.stdout

# --- DEAP Initialization ---

creator.create("FitnessMulti", base.Fitness, weights=weights)
creator.create("Individual", array.array, typecode="d", fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, N_PARAMETERS)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxBlend, alpha=CALIBRATION_CONFIG["DEAP"]["blend_alpha"])
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=CALIBRATION_CONFIG["DEAP"]["gaussian_sigma"], indpb=CALIBRATION_CONFIG["DEAP"]["gaussian_indpb"])
toolbox.register("select", tools.selNSGA2)

random.seed(42)

def generate_initial_population():
    population = toolbox.population(n=MU)
    individuals = []
    for i, ind in enumerate(population):
        label = "0_{:03d}".format(i)
        individuals.append({
            "label": label,
            "generation": 0,
            "individual_id": "{:03d}".format(i),
            "values": [float(x) for x in ind]
        })
    return individuals

INITIAL_POPULATION = generate_initial_population()

# --- Rules ---

rule init_base:
    output: touch("base_init.done")
    log: "logs/base_init.log"
    run: run_command("geb init --overwrite", log[0], "Failed to initialize base model")

rule build_base:
    input: "base_init.done"
    output: touch("base_build.done")
    log: "logs/base_build.log"
    run: run_command("geb build --continue", log[0], "Failed to build base model")

rule generate_initial_parameters:
    input: "base_build.done"
    output: RUNS_DIR + f"/generation_0_population.yml"
    run:
        with open(output[0], "w") as f:
            yaml.dump({"individuals": INITIAL_POPULATION}, f, default_flow_style=False)

rule generate_individual_parameters:
    input:
        pop_file=lambda wildcards: RUNS_DIR + f"/generation_0_population.yml" if int(wildcards.gen) == 0 else RUNS_DIR + f"/generation_{int(wildcards.gen) - 1}_next.yml",
        checkpoint_done=lambda wildcards: checkpoints.create_next_generation.get(gen=int(wildcards.gen) - 1).output if int(wildcards.gen) > 0 else []
    output: params=RUNS_DIR + "/{gen}_{ind}/parameters.yml"
    run:
        with open(input.pop_file, "r") as f:
            pop_data = yaml.safe_load(f)
        
        label = f"{wildcards.gen}_{wildcards.ind}"
        individual_data = next((ind for ind in pop_data["individuals"] if ind["label"] == label), None)
        if individual_data is None:
            raise ValueError(f"Individual {label} not found in population file")
        
        actual_params = {
            param_config["variable"]: float(param_config["min"] + individual_data["values"][i] * (param_config["max"] - param_config["min"]))
            for i, (param_name, param_config) in enumerate(PARAMETERS.items())
        }
        
        Path(output.params).parent.mkdir(parents=True, exist_ok=True)
        with open(output.params, "w") as f:
            yaml.dump({
                "label": label,
                "generation": int(wildcards.gen),
                "individual_id": wildcards.ind,
                "normalized_values": individual_data["values"],
                "parameters": actual_params
            }, f, default_flow_style=False, sort_keys=False)

rule init_individual:
    input:
        params=RUNS_DIR + "/{gen}_{ind}/parameters.yml",
        base_build="base_build.done"
    output: touch(RUNS_DIR + "/{gen}_{ind}/init.done")
    log: RUNS_DIR + "/{gen}_{ind}/logs/init.log"
    run:
        run_dir = Path(RUNS_DIR) / f"{wildcards.gen}_{wildcards.ind}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        with open(run_dir / "build.yml", "w") as f: f.write("# Empty build config\n")
        with open(run_dir / "update.yml", "w") as f: f.write("# Empty update config\n")

rule alter_individual:
    input: RUNS_DIR + "/{gen}_{ind}/init.done"
    output: touch(RUNS_DIR + "/{gen}_{ind}/altered.done")
    log: RUNS_DIR + "/{gen}_{ind}/logs/alter.log"
    run:
        run_dir = Path(RUNS_DIR) / f"{wildcards.gen}_{wildcards.ind}"
        cmd = f"geb alter --from-model ../../../. -wd {run_dir}"
        run_command(cmd, log[0], f"Failed to alter model {wildcards.gen}_{wildcards.ind}")

rule set_individual_parameters:
    input:
        altered_done=RUNS_DIR + "/{gen}_{ind}/altered.done",
        parameters=RUNS_DIR + "/{gen}_{ind}/parameters.yml",
    output: touch(RUNS_DIR + "/{gen}_{ind}/params_set.done")
    log: RUNS_DIR + "/{gen}_{ind}/logs/set_params.log"
    run:
        with open(input.parameters, "r") as f: params_data = yaml.safe_load(f)
        actual_params = params_data["parameters"]
        run_dir = Path(RUNS_DIR) / f"{wildcards.gen}_{wildcards.ind}"
        
        param_args = " ".join(f"{k}={v}" for k, v in actual_params.items())
        model_config = parse_config(run_dir / "model.yml")
        calib_config = get_calibration_config(model_config)
        
        datetime_args = "general.spinup_time={0} general.start_time={1} general.end_time={2}".format(
            calib_config["spinup_time"], calib_config["start_time"], calib_config["end_time"]
        )
        
        cmd = "geb set -c model.yml --working-directory {0} {1} {2} report=null report._discharge_stations+=true".format(
            run_dir, param_args, datetime_args
        )
        run_command(cmd, log[0], f"Failed to set parameters for {wildcards.gen}_{wildcards.ind}")

rule spinup_individual:
    input: RUNS_DIR + "/{gen}_{ind}/params_set.done"
    output: touch(RUNS_DIR + "/{gen}_{ind}/spinup.done")
    log: RUNS_DIR + "/{gen}_{ind}/logs/spinup.log"
    run:
        run_dir = Path(RUNS_DIR) / f"{wildcards.gen}_{wildcards.ind}"
        run_command(f"geb spinup -wd {run_dir}", log[0], f"Spinup failed for {wildcards.gen}_{wildcards.ind}")

rule run_individual:
    input: RUNS_DIR + "/{gen}_{ind}/spinup.done"
    output: touch(RUNS_DIR + "/{gen}_{ind}/run.done")
    log: RUNS_DIR + "/{gen}_{ind}/logs/run.log"
    run:
        run_dir = Path(RUNS_DIR) / f"{wildcards.gen}_{wildcards.ind}"
        run_command(f"geb run -wd {run_dir}", log[0], f"Main run failed for {wildcards.gen}_{wildcards.ind}")

rule evaluate_individual:
    input: RUNS_DIR + "/{gen}_{ind}/run.done"
    output: RUNS_DIR + "/{gen}_{ind}/fitness.yml"
    log: RUNS_DIR + "/{gen}_{ind}/logs/evaluate.log"
    run:
        run_dir = Path(RUNS_DIR) / f"{wildcards.gen}_{wildcards.ind}"
        model_config = parse_config(run_dir / "model.yml")
        calib_config = get_calibration_config(model_config)
        targets = calib_config.get("calibration_targets", {})
        if not targets:
            raise ValueError(f"No targets found for {CALIBRATION_TARGET}")

        all_metrics = {}
        weighted_fitness_sum = 0.0
        total_weight = 0.0

        for target_name, target_info in targets.items():
            method, metric_key, weight = target_info["method"], target_info["metric"], float(target_info["weight"])
            cmd = f"geb evaluate {method} -wd {run_dir} --create_plots false"
            stdout = run_command(cmd, log[0], f"Eval failed for {target_name}")

            try:
                metrics_dict = json.loads(stdout.strip().split("\n")[-1])
                metric_value = float(metrics_dict[metric_key])
                all_metrics[target_name] = {"value": metric_value, "metric": metric_key, "method": method, "weight": weight}
                weighted_fitness_sum += metric_value * weight
                total_weight += weight
            except Exception as e:
                raise RuntimeError(f"Failed to parse metrics for {target_name}: {e}")

        if total_weight <= 0: raise ValueError("Total weight <= 0")
        with open(output[0], "w") as f:
            yaml.dump({"fitness": weighted_fitness_sum/total_weight, "targets": all_metrics}, f)

checkpoint create_next_generation:
    input:
        fitness_files=lambda wildcards: [RUNS_DIR + "/{gen}_{i:03d}/fitness.yml".format(gen=wildcards.gen, i=i) for i in range(MU if int(wildcards.gen) == 0 else LAMBDA)],
        prev_pop=lambda wildcards: RUNS_DIR + f"/generation_0_population.yml" if int(wildcards.gen) == 0 else RUNS_DIR + f"/generation_{int(wildcards.gen) - 1}_next.yml"
    output:
        selected_pop=RUNS_DIR + f"/generation_{{gen}}_selected.yml",
        summary=RUNS_DIR + f"/generation_{{gen}}_summary.yml",
        next_pop=RUNS_DIR + f"/generation_{{gen}}_next.yml"
    run:
        gen = int(wildcards.gen)
        with open(input.prev_pop, "r") as f: pop_data = yaml.safe_load(f)
        
        individuals_with_fitness = []
        for fitness_file in input.fitness_files:
            path_parts = Path(fitness_file).parent.name.split("_")
            curr_gen, curr_ind = int(path_parts[0]), path_parts[1]
            with open(fitness_file, "r") as f: fitness_data = yaml.safe_load(f)
            label = f"{curr_gen}_{curr_ind}"
            ind_data = next((ind.copy() for ind in pop_data["individuals"] if ind["label"] == label), None)
            if ind_data:
                fit_val = fitness_data["fitness"]
                ind_data["fitness"] = fit_val if isinstance(fit_val, list) else [fit_val]
                individuals_with_fitness.append(ind_data)
        
        with open(output.selected_pop, "w") as f: yaml.dump({"individuals": individuals_with_fitness}, f, default_flow_style=False)
        
        if gen < NGEN - 1:
            deap_individuals = []
            for ind_data in individuals_with_fitness:
                ind = creator.Individual(ind_data["values"])
                ind.fitness.values = tuple(ind_data["fitness"])
                deap_individuals.append(ind)
            
            selected = toolbox.select(deap_individuals, MU)
            offspring = algorithms.varOr(selected, toolbox, LAMBDA, cxpb=CALIBRATION_CONFIG["DEAP"]["crossover_prob"], mutpb=CALIBRATION_CONFIG["DEAP"]["mutation_prob"])
            
            for child in offspring:
                for j in range(len(child)): child[j] = max(0.0, min(1.0, child[j]))
            
            offspring_data = [{"label": f"{gen + 1}_{i:03d}", "generation": gen + 1, "individual_id": f"{i:03d}", "values": [float(x) for x in child]} for i, child in enumerate(offspring)]
            with open(output.next_pop, "w") as f: yaml.dump({"individuals": offspring_data}, f, default_flow_style=False)
        else:
            with open(output.next_pop, "w") as f: yaml.dump({"individuals": []}, f, default_flow_style=False)

        summary_data = {
            "generation": gen,
            "num_individuals": len(individuals_with_fitness),
            "fitnesses": [ind["fitness"] for ind in individuals_with_fitness],
            "mean_fitness": [float(np.mean([ind["fitness"][i] for ind in individuals_with_fitness])) for i in range(len(individuals_with_fitness[0]["fitness"]))],
            "max_fitness": [float(np.max([ind["fitness"][i] for ind in individuals_with_fitness])) for i in range(len(individuals_with_fitness[0]["fitness"]))],
            "min_fitness": [float(np.min([ind["fitness"][i] for ind in individuals_with_fitness])) for i in range(len(individuals_with_fitness[0]["fitness"]))],
        }
        with open(output.summary, "w") as f: yaml.dump(summary_data, f, default_flow_style=False)

def aggregate_checkpoint_outputs(wildcards):
    checkpoints.create_next_generation.get(gen=NGEN - 1)
    return [RUNS_DIR + f"/generation_{gen}_selected.yml" for gen in range(NGEN)]

rule complete_calibration:
    input: aggregate_checkpoint_outputs
    output: touch(f"calibration_{CALIBRATION_TARGET}_complete.done")
    run:
        all_individuals = []
        for gen in range(NGEN):
            with open(RUNS_DIR + f"/generation_{gen}_selected.yml", "r") as f:
                all_individuals.extend(yaml.safe_load(f)["individuals"])
        
        deap_all = []
        for ind_data in all_individuals:
            ind = creator.Individual(ind_data["values"])
            ind.fitness.values = tuple(ind_data["fitness"])
            ind.label = ind_data["label"]
            deap_all.append(ind)
        
        pf = tools.ParetoFront()
        pf.update(deap_all)
        
        pareto_data = [{"label": ind.label, "values": [float(x) for x in ind], "fitness": list(ind.fitness.values)} for ind in pf]
        with open(f"calibration_{CALIBRATION_TARGET}_pareto_front.yml", "w") as f:
            yaml.dump({"target": CALIBRATION_TARGET, "pareto_front": pareto_data}, f, default_flow_style=False)
