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
import shutil
import sys
import random
import yaml
import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime
from deap import base, creator, tools, algorithms
import array
from pathlib import Path
from geb.runner import parse_config
from snakemake.logging import logger as snakemake_logger

# silence snakemake logging, in favour of our own progress messages and logging within each step
snakemake_logger.setLevel(logging.WARNING)

# --- Configuration & Setup ---

def load_model_config():
    """Load the model.yml configuration file."""
    model_file = Path("model.yml")
    if not model_file.exists():
        raise FileNotFoundError(f"Model configuration file not found: {model_file}")
    
    with open(model_file, "r") as f:
        return yaml.safe_load(f)

MODEL_CONFIG = load_model_config()
CALIBRATION_TRACK = config.get("TRACK", None)

if CALIBRATION_TRACK is None:
    print("Error: calibration track must be specified for calibration workflows. For example 'geb workflow calibrate hydrology'")
    sys.exit(1)

# Ensure the calibration and target section exist
if "calibration" not in MODEL_CONFIG:
    raise ValueError("No 'calibration' section found in model.yml")
if CALIBRATION_TRACK not in MODEL_CONFIG["calibration"]:
    available = list(MODEL_CONFIG["calibration"].keys())
    raise KeyError(
        f"Calibration track '{CALIBRATION_TRACK}' not found in model.yml. "
        f"Available tracks/sections: {available}"
    )

CALIBRATION_CONFIG = MODEL_CONFIG["calibration"][CALIBRATION_TRACK]
#RUNS_DIR = f"calibration/{CALIBRATION_TRACK}"
RUNS_DIR = Path("calibration") / CALIBRATION_TRACK

#NEW CODE
#Path accces to all cluster folders
CLUSTER_BASES = {
    Path(base).parent.name: Path(base).resolve()
    for base in CALIBRATION_CONFIG["clusters"]
}
CLUSTERS = sorted(CLUSTER_BASES)

if not CLUSTERS:
    raise ValueError("No clusters configured for this calibration track.")
#END NEWCODE

# DEAP Settings
NGEN = CALIBRATION_CONFIG["DEAP"]["ngen"]
MU = CALIBRATION_CONFIG["DEAP"]["mu"]
LAMBDA = CALIBRATION_CONFIG["DEAP"]["lambda_"]

# Parameters
PARAMETERS = {}
for param_name, param_config in CALIBRATION_CONFIG["parameters"].items():
#START NEWCODE
    minimum, maximum = param_config["min"], param_config["max"]
    if minimum >= maximum:
        raise ValueError(
            f"Invalid range for parameter {param_name!r}: {minimum}..{maximum}"
        )
#END NEWCODE
    PARAMETERS[param_name] = {
        "min": param_config["min"],
        "max": param_config["max"],
        "variable": param_config["variable"]
    }
N_PARAMETERS = len(PARAMETERS)
#START NEWCODE
if not PARAMETERS:
    raise ValueError(f"No calibration parameters configured for {CALIBRATION_TRACK!r}")
#END NEWCODE

# Target Weights
CALIBRATION_TARGETS = CALIBRATION_CONFIG["calibration_targets"]
weights = [target_info["weight"] for target_info in CALIBRATION_TARGETS.values()]
for weight in weights:
    if not isinstance(weight, (int, float)):
        raise ValueError(f"Invalid weight value: {weight}. Weights must be numeric.")

# --- DEAP Initialization ---

#NEWCODE
#Derive individual/cluster run path consistently
def individual_dir(gen, ind):
    return RUNS_DIR / f"{gen}_{ind}"

def cluster_run_dir(gen, ind, cluster):
    return individual_dir(gen, ind) / cluster

def map_source(cluster, hazard):
    return CLUSTER_BASES[cluster] / "output"/ f"{hazard}_maps"

def cluster_files(wildcards, suffix):
    return [str(cluster_run_dir(wildcards.gen, wildcards.ind, c) / suffix)
            for c in CLUSTERS]
#END NEWCODE            

def get_progress_message(wildcards, action):
    """Helper to generate a detailed progress message for individual steps."""
    gen = wildcards.gen
    ind = wildcards.ind
    total = MU if int(gen) == 0 else LAMBDA
    
    # Count done files for the current generation
    path = Path(RUNS_DIR)
    n_params = len(list(path.glob(f"{gen}_*/parameters.yml")))
    n_init = len(list(path.glob(f"{gen}_*/init.done")))
    n_altered = len(list(path.glob(f"{gen}_*/input/build_complete.txt")))
    n_params_set = len(list(path.glob(f"{gen}_*/params_set.done")))
    n_spinup = len(list(path.glob(f"{gen}_*/spinup.done")))
    n_run = len(list(path.glob(f"{gen}_*/run.done")))
    n_eval = len(list(path.glob(f"{gen}_*/fitness.yml")))
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return (
        f"[{timestamp}] {action} for gen {gen}, ind {ind}. "
        f"Progress: {n_params}/{total} create, {n_init}/{total} init, {n_altered}/{total} alter, "
        f"{n_params_set}/{total} set, {n_spinup}/{total} spinup, {n_run}/{total} run, {n_eval}/{total} evaluate."
    )

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

def run_command(cmd: str, log_path: str, error_msg: str = "GEB command failed") -> str | None:
    """Helper function to run a GEB command, log output, and return captured stdout."""
    #START NEWCODE
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    #END NEWCODE
    env = os.environ.copy()
    env["GEB_OVERRIDE_PROGRESSBAR_DT"] = "10"

    # print(f"[GEB] Running: {cmd}")
    
    process = subprocess.run(
        cmd, 
        shell=True, 
        capture_output=True,
        env=env,
        text=True
    )
    
    #with open(log_path, "a") as log_file:
    #    log_file.write(f"\n[{pd.Timestamp.now()}] Running: {cmd}\n")
    #    if process.stdout:
    #        log_file.write("--- STDOUT ---\n")
    #        log_file.write(process.stdout)
    #    if process.stderr:
    #        log_file.write("--- STDERR ---\n")
    #        log_file.write(process.stderr)
    #    log_file.write(f"Return code: {process.returncode}\n")

    #START NEWCODE
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n[{pd.Timestamp.now()}] Running: {cmd}\n")
        f.write(process.stdout or "")
        f.write(process.stderr or "")
        f.write(f"\nReturn code: {process.returncode}\n")  
    if process.returncode:
        raise RuntimeError(f"{error_msg}. See {log_path}")

    #END NEWCODE

    #if process.returncode != 0:
    #    raise RuntimeError(f"{error_msg}. Check logs at: {log_path}")

    return process.stdout

rule init_base:
    output: touch("base_init.done")
    log: "logs/snakemake_base_init.log"
    message: "Initializing base GEB model..."
    run: 
        if not Path("model.yml").exists():
            run_command("geb init --overwrite", log[0], "Failed to initialize base model")
        else:
            with open(log[0], "a") as log_file:
                log_file.write(f"model.yml already exists, skipping 'geb init'\n")

#START NEWCODE
#GEB selects the configured return period from these named map directories
#rule generate_flood_maps:
#    input: "base_build.done"
#    output: touch(RUNS_DIR + "/flood_maps.done")
#    log: RUNS_DIR + "/logs/generate_flood_maps.log"
#    run:
#        def _find_flood_maps_folder() -> tuple[Path, list[Path]]
#            candidates: list[Path] = []
#            try:
#                f = Path(__file__).resolve()
#                for i in range(1. 6):
#                    if len(f.parameters) >= i:
#                        candidates.append(f.parents[i - 1])
#            except Exception:
#                params_set
#
#            cwd = Path.cwd().resolve()
#            candidates.append(cwd)
#            for i in range(1, 6):
#                if len(cwd.parents) >= i:
#                    candidates.append(cwd.parents[i - 1])
#
#            seen = set()
#            uniq: list[Path] = []
#            for c in candidates:
#                if c not in seen:
#                    seen.add(c)
#                    uniq.append(c)
#
#            for root in uniq:
#                p = root / "models" / "large_scale" / "large_scale" / "clusterxxx" / "base" / "output" / "flood_maps" 
#                if p.exists():
#                    return p, uniq
#
#            return Path("models") / "large_scale" / "large_scale" / "clusterxxx" / "base" / "output" / "flood_maps", uniq
#
#        src, probed_candidates = _find_flood_maps_folder
#        try:
#            with open(log[0], "a") as fh:
#                fh.write(f"Probed candidates for flood_maps (in order):\n")
#                for c in probed_candidates:
#                    fh.write(str(c) + "\n")
#                fh.write(f"Resolved src path: {src}\n")
#                fh.write(f"src.exists: {src.exists()}\n")
#                if src.exists():
#                    try:
#                        fh.write("Contents of src:\n")
#                        for p in sorted(src.iterdir()):
#                            fh.write(" - " + p.name + ("/" if p.is_dir() else "") + "\n")
#                    except Exception as e:
#                        fh.write("Failed to list src contents: " + str(e) + "\n")
#        except Exception:
#            pass
#
#        dst = Path(RUNS_DIR) / "output" / "flood_maps"
#
#        if not src.exists():
#            with open(log[0], "a") as fh:
#                fh.write(f"Source flood maps not found at {src}\n")
#            raise RuntimeError(
#                f"Source flood maps not found at {src}. Run 'geb exec estimate_return_periods -wd models/geul' first."
#            )
#
#        # Ensure destination parent exists
#        dst.parent.mkdir(parents=True, exist_ok=True)
#        # Remove any previous copy then copy the entire folder
#        try:
#            if dst.exists():
#                shutil.rmtree(dst)
#            shutil.copytree(src, dst)
#        except Exception as e:
#            with open(log[0], "a") as fh:
#                fh.write(f"Failed copying flood_maps: {e}\n")
#            raise RuntimeError(f"Failed to copy flood_maps: {e}")
#
#        Path(output[0]).touch()


rule build_base:
    input: "base_init.done"
    output: "input/build_complete.txt"
    log: "logs/snakemake_base_build.log"
    message: "Building base GEB model..."
    run: 
        run_command("geb build --continue", log[0], "Failed to build base model")

rule generate_initial_parameters:
    input: 
        base_init="base_init.done",
        cluster_builds=[
            str(base / "input" / "build_complete.txt")
            for base in CLUSTER_BASES.values()
        ]    
    output: str(RUNS_DIR / f"generation_0_{CALIBRATION_TRACK}_population.yml")
    message: "Generating initial DEAP population for {CALIBRATION_TRACK}..."
    run:
        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
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
        with open(output[0], "w") as f:
            yaml.dump({"individuals": individuals}, f, default_flow_style=False)

rule generate_individual_parameters:
    input:
        pop_file=lambda wildcards: (
            str(RUNS_DIR / f"generation_0_{CALIBRATION_TRACK}_population.yml")
            if int(wildcards.gen) == 0
            else f"calibration/generation_{int(wildcards.gen) - 1}_{CALIBRATION_TRACK}_next.yml"
        ),
        #pop_file=lambda wildcards: "calibration/generation_0_{0}_population.yml".format(TRACK=CALIBRATION_TRACK) if int(wildcards.gen) == 0 else "calibration/generation_{0}_{1}_next.yml".format(int(wildcards.gen) - 1, CALIBRATION_TRACK),
        checkpoint_done=lambda wildcards: checkpoints.create_next_generation.get(gen=int(wildcards.gen) - 1).output if int(wildcards.gen) > 0 else []
    #output: params=RUNS_DIR + "/{gen}_{ind}/parameters.yml"
    output: params=str(RUNS_DIR / "{gen}_{ind}" / "parameters.yml")
    run:
        print(get_progress_message(wildcards, "Defining parameters"))
        with open(input.pop_file, "r") as f:
            pop_data = yaml.safe_load(f)
        
        label = f"{wildcards.gen}_{wildcards.ind}"
        individual_data = next((ind for ind in pop_data["individuals"] if ind["label"] == label), None)
        if individual_data is None:
            raise ValueError(f"Individual {label} not found in population file")
        
        #actual_params = {
        #    param_config["variable"]: float(param_config["min"] + individual_data["values"][i] * (param_config["max"] - param_config["min"]))
        #    for i, (param_name, param_config) in enumerate(PARAMETERS.items())
        #}
        #START NEWCODE
        actual_params = {}
        for i, (pram_name, param_config) in enumerate(PARAMETERS.items()):
            raw = float(param_config["min"]) + float(individual_data["values"][i]) * (
                float(param_config["max"]) - float(param_config["min"])
            )
            value = int(round(raw)) if param_config.get("is_integer", False) else raw 
            actual_params[param_config["variable"]] = value
        #END NEWCODE

        
        Path(output.params).parent.mkdir(parents=True, exist_ok=True)
        with open(output.params, "w") as f:
            yaml.dump({
                "label": label,
                "generation": int(wildcards.gen),
                "individual_id": wildcards.ind,
                "normalized_values": individual_data["values"],
                "parameters": actual_params
            }, f, default_flow_style=False, sort_keys=False)

#NEW RULES
rule init_individual:
    input:
        params=str(RUNS_DIR / "{gen}_{ind}" / "parameters.yml")
    output:
        done=touch(str(RUNS_DIR / "{gen}_{ind}" / "{cluster}" / "prepared.done"))
    log:
        str(RUNS_DIR / "{gen}_{ind}" / "{cluster}" / "logs" / "prepare.log")
    wildcard_constraints:
        cluster="|".join(CLUSTERS)
    run:
        base = CLUSTER_BASES[wildcards.cluster]
        work = cluster_run_dir(wildcards.gen, wildcards.ind, wildcards.cluster)
        work.mkdir(parents=True, exist_ok=True)
        (work / "logs").mkdir(parents=True, exist_ok=True)

        # CHANGED: use this cluster's base model as the source.
        # Confirm this CLI form with `geb alter --help` on your GEB version.
        cmd = f"geb alter --from-model {base} -wd {work}"
        run_command(cmd, log[0], f"Could not prepare {wildcards.cluster}")

        model_file = work / "model.yml"
        with model_file.open("r", encoding="utf-8") as f:
            candidate_config = yaml.safe_load(f) or {}

        report_config = candidate_config.get("report")
        if not isinstance(report_config, dict):
            report_config = {}
            candidate_config["report"] = report_config
        internal_config = report_config.get("_config")
        if not isinstance(internal_config, dict):
            internal_config = {}

        report_config["_config"] = {
            "compression_level": (
                internal_config.get("compression_level")
                if internal_config.get("compression_level") is not None
                else 9
            ),
            "chunk_target_size_bytes": (
                internal_config.get("chunk_target_size_bytes")
                if internal_config.get("chunk_target_size_bytes") is not None
                else 100_000_000
            ),
        }

        adaptation_eval = (
            candidate_config
            .setdefault("agent_settings", {})
            .setdefault("households", {})
            .setdefault("adaptation_evaluation", {})
        )
        #adaptation_eval["cluster_run_dirs"] = [
        #    str(cluster_run_dir(wildcards.gen, wildcards.ind, cluster).resolve())
        #    for cluster in CLUSTERS
        #]
        #adaptation_eval["cluster_base_dirs"] = [
        #    str(Path(CLUSTER_BASES[cluster]).resolve())
        #    for cluster in CLUSTERS
        #]

        base_dirs = [
            str(Path(CLUSTER_BASES[cluster]).resolve())
            for cluster in CLUSTERS
        ]
        adaptation_eval["clusters"] = base_dirs
        adaptation_eval["cluster_base_dirs"] = base_dirs
        adaptation_eval["cluster_run_dirs"] = [
            str(cluster_run_dir(wildcards.gen, wildcards.ind, cluster).resolve())
            for cluster in CLUSTERS
        ]

        with model_file.open("w", encoding="utf-8") as f:
            yaml.safe_dump(candidate_config, f, sort_keys=False)

        # NEW: make both map sets available to this cluster run.
        output_dir = work / "output" 
        output_dir.mkdir(parents=True, exist_ok=True)
        for hazard in ("flood", "wind"):
            src = map_source(wildcards.cluster, hazard)
            dst = output_dir / f"{hazard}_maps"
            if not src.is_dir():
                raise FileNotFoundError(f"Missing {hazard} maps: {src}")
            if dst.is_symlink() or dst.is_file():
                dst.unlink()
            elif dst.exists():
                shutil.rmtree(dst)
            try:
                dst.symlink_to(src, target_is_directory=True)
            except OSError:
                shutil.copytree(src, dst)

        spinup_output_dir = output_dir / "spinup"
        spinup_output_dir.mkdir(parents=True, exist_ok=True)

        for hazard in ("flood", "wind"):
            dst = spinup_output_dir / f"{hazard}_maps"
            if dst.is_symlink() or dst.is_file():
                dst.unlink()
            elif dst.exists():
                shutil.rmtree(dst)
            dst.symlink_to(
                Path("..") / f"{hazard}_maps",
                target_is_directory=True,
            )

rule set_cluster_parameters:
    input:
        prepared=str(RUNS_DIR / "{gen}_{ind}" / "{cluster}" / "prepared.done"),
        params=str(RUNS_DIR / "{gen}_{ind}" / "parameters.yml")
    output:
        done=touch(str(RUNS_DIR / "{gen}_{ind}" / "{cluster}" / "params_set.done"))
    log:
        str(RUNS_DIR / "{gen}_{ind}" / "{cluster}" / "logs" / "set_params.log")
    wildcard_constraints:
        cluster="|".join(CLUSTERS)
    run:
        work = cluster_run_dir(wildcards.gen, wildcards.ind, wildcards.cluster)
        params = yaml.safe_load(Path(input.params).read_text())["parameters"]
        param_args = " ".join(f"{k}={v}" for k, v in params.items())
        date_args = (
            f"general.spinup_time={CALIBRATION_CONFIG['spinup_time']} "
            f"general.start_time={CALIBRATION_CONFIG['start_time']} "
            f"general.end_time={CALIBRATION_CONFIG['end_time']}"
        )
        cmd = (f"geb set -c model.yml --working-directory {work} "
               f"{param_args} {date_args} "
               "report._discharge_stations+=true")
        run_command(cmd, log[0], "Could not set cluster parameters")
        
rule spinup_cluster:
    input:
        str(RUNS_DIR / "{gen}_{ind}" / "{cluster}" / "params_set.done")
    output:
        touch(str(RUNS_DIR / "{gen}_{ind}" / "{cluster}" / "spinup.done"))
    log:
        str(RUNS_DIR / "{gen}_{ind}" / "{cluster}" / "logs" / "spinup.log")
    wildcard_constraints:
        cluster="|".join(CLUSTERS)
    run:
        work = cluster_run_dir(wildcards.gen, wildcards.ind, wildcards.cluster)
        run_command(f"geb spinup -wd {work}", log[0], "Cluster spinup failed")

rule run_cluster:
    input:
        str(RUNS_DIR / "{gen}_{ind}" / "{cluster}" / "spinup.done")
    output:
        touch(str(RUNS_DIR / "{gen}_{ind}" / "{cluster}" / "run.done"))
    log:
        str(RUNS_DIR / "{gen}_{ind}" / "{cluster}" / "logs" / "run.log")
    wildcard_constraints:
        cluster="|".join(CLUSTERS)
    run:
        work = cluster_run_dir(wildcards.gen, wildcards.ind, wildcards.cluster)
        run_command(f"geb exec run_yearly -wd {work}", log[0], "Cluster model run failed")
        sim_root = work / "simulation_root"
        if sim_root.exists():
            shutil.rmtree(sim_root)

# END OF NEW RULES


#NEWCODE/NEWRULE
rule evaluate_cluster:
    input:
        str(RUNS_DIR / "{gen}_{ind}" / "{cluster}" / "run.done")
    output:
        metrics=str(RUNS_DIR / "{gen}_{ind}" / "{cluster}" / "metrics.json")
    log:
        str(RUNS_DIR / "{gen}_{ind}" / "{cluster}" / "logs" / "evaluate.log")
    wildcard_constraints:
        cluster="|".join(CLUSTERS)
    run:
        work = cluster_run_dir(wildcards.gen, wildcards.ind, wildcards.cluster)
        all_metrics = {}
        for name, target in CALIBRATION_TARGETS.items():
            cmd = f"geb evaluate {target['method']} -wd {work}"
            stdout = run_command(cmd, log[0], f"Evaluation failed for {name}")
            lines = [line.strip() for line in stdout.splitlines() if line.strip()]
            if not lines:
                raise RuntimeError(f"No JSON result returned for target {name}")
            all_metrics[name] = json.loads(lines[-1])
        Path(output.metrics).write_text(json.dumps(all_metrics, indent=2))

rule evaluate_individual:
# National fitness file per individual, after all clusters finish
    input:
        #cluster_metrics=lambda wc: [
        #    str(cluster_run_dir(wc.gen, wc.ind, cluster) / "metrics.json")
        #    for cluster in CLUSTERS
        #]
        runs=lambda wc: [
            str(cluster_run_dir(wc.gen, wc.ind, c) / "run.done")
            for c in CLUSTERS
        ]
    output:
        #fitness=str(RUNS_DIR / "{gen}_{ind}" / "fitness.yml")
        fitness=str(RUNS_DIR / "{gen}_{ind}" / "fitness.yml")
    log: 
        str(RUNS_DIR / "{gen}_{ind}"/ "logs"/ "evaluate.log")
    run:
        entry = cluster_run_dir(wildcards.gen, wildcards.ind, CLUSTERS[0])
        cmd = f"geb evaluate agents.evaluate_household_adaptation -wd {entry}"
        stdout = run_command(cmd, log[0], "National household adaptation failed")
        lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        metrics = json.loads(lines[-1])

        fitness = {}
        for name, target in CALIBRATION_TARGETS.items():
            metric = target["metric"]
            if metric not in metrics:
                raise KeyError(
                    f"Evaluator did not return{metric}. "
                    f"Return keys: {sorted(metrics)}"
                )

            fitness[name] = {
                "value": float(metrics[metric]),
                "metric": metric,
                "method": target["method"],
                "weight": float(target["weight"]),
            }
        
        Path(output.fitness).parent.mkdir(parents=True, exist_ok=True)
        Path(output.fitness).write_text(yaml.safe_dump({"targets": fitness}, sort_keys=False))


checkpoint create_next_generation:
    input:
        #fitness_files=lambda wildcards: [RUNS_DIR + "/{gen}_{i:03d}/fitness.yml".format(gen=wildcards.gen, i=i) for i in range(MU if int(wildcards.gen) == 0 else LAMBDA)],
        fitness_files=lambda wildcards: [
            str(RUNS_DIR / f"{wildcards.gen}_{i:03d}" / "fitness.yml")
            for i in range(MU if int(wildcards.gen) == 0 else LAMBDA)
        ],
        #prev_pop=lambda wildcards: "calibration/generation_0_{0}_population.yml".format(TRACK=CALIBRATION_TRACK) if int(wildcards.gen) == 0 else "calibration/generation_{gen_prev}_{target}_next.yml".format(gen_prev=int(wildcards.gen) - 1, target=CALIBRATION_TRACK)
        prev_pop=lambda wildcards: (
            str(RUNS_DIR / f"generation_0_{CALIBRATION_TRACK}_population.yml")
            if int(wildcards.gen) == 0
            else f"calibration/generation_{int(wildcards.gen) - 1}_{CALIBRATION_TRACK}_next.yml"
        )
    output:
        selected_pop="calibration/generation_{gen}_{target}_selected.yml".format(target=CALIBRATION_TRACK, gen="{gen}"),
        summary="calibration/generation_{gen}_{target}_summary.yml".format(target=CALIBRATION_TRACK, gen="{gen}"),
        next_pop="calibration/generation_{gen}_{target}_next.yml".format(target=CALIBRATION_TRACK, gen="{gen}")
    message: "Aggregating fitness and creating next generation for {wildcards.gen}..."
    run:
        gen = int(wildcards.gen)
        with open(input.prev_pop, "r") as f: pop_data = yaml.safe_load(f)
        
        individuals_with_fitness = []
        for fitness_file in input.fitness_files:
            path_parts = Path(fitness_file).parent.name.split("_")
            curr_gen, curr_ind = int(path_parts[0]), path_parts[1]
            with open(fitness_file, "r") as f: fitness_data = yaml.safe_load(f)
            label = "{0}_{1}".format(curr_gen, curr_ind)
            ind_data = next((ind.copy() for ind in pop_data["individuals"] if ind["label"] == label), None)
            if ind_data:
                # We extract individual metric values into a tuple for DEAP multi-objective handling
                # DEAP 'weights' already handle maximization (+1) vs minimization (-1)
                #fit_tuple = [t["value"] for t in fitness_data["targets"].values()]
                #START NEWCODE
                fit_tuple = [
                    float(fitness_data["targets"][target_name]["value"])
                    for target_name in CALIBRATION_TARGETS
                ]
                #END NEWCODE
                ind_data["fitness"] = fit_tuple
                individuals_with_fitness.append(ind_data)
        
        with open(output.selected_pop, "w") as f: yaml.dump({"individuals": individuals_with_fitness}, f, default_flow_style=False)
        
        if gen < NGEN - 1:
            deap_individuals = []
            for ind_data in individuals_with_fitness:
                ind = creator.Individual(ind_data["values"])
                ind.fitness.values = ind_data["fitness"]
                deap_individuals.append(ind)
            
            selected = toolbox.select(deap_individuals, MU)
            offspring = algorithms.varOr(selected, toolbox, LAMBDA, cxpb=CALIBRATION_CONFIG["DEAP"]["crossover_prob"], mutpb=CALIBRATION_CONFIG["DEAP"]["mutation_prob"])
            
            for child in offspring:
                for j in range(len(child)): child[j] = max(0.0, min(1.0, child[j]))
            
            offspring_data = [{"label": "{0}_{1:03d}".format(gen + 1, i), "generation": gen + 1, "individual_id": "{0:03d}".format(i), "values": [float(x) for x in child]} for i, child in enumerate(offspring)]
            with open(output.next_pop, "w") as f: yaml.dump({"individuals": offspring_data}, f, default_flow_style=False)
        else:
            with open(output.next_pop, "w") as f: yaml.dump({"individuals": []}, f, default_flow_style=False)

        summary_data = {
            "generation": gen,
            "num_individuals": len(individuals_with_fitness),
            "fitness_tuples": [list(ind["fitness"]) for ind in individuals_with_fitness],
            # Mean/Max/Min of the first objective as a proxy for summary (or full breakdown)
            "mean_fitness": [float(np.mean([ind["fitness"][i] for ind in individuals_with_fitness])) for i in range(len(individuals_with_fitness[0]["fitness"]))],
        }
        with open(output.summary, "w") as f: yaml.dump(summary_data, f, default_flow_style=False)

def aggregate_checkpoint_outputs(wildcards):
    checkpoints.create_next_generation.get(gen=NGEN - 1)
    return ["calibration/generation_{gen}_{target}_selected.yml".format(gen=gen, target=CALIBRATION_TRACK) for gen in range(NGEN)]

rule complete_calibration:
    input: aggregate_checkpoint_outputs
    output: touch("calibration/calibration_{target}_complete.done".format(target=CALIBRATION_TRACK))
    message: "Finalizing calibration for {CALIBRATION_TRACK} and computing Pareto front..."
    run:
        all_individuals = []
        for gen in range(NGEN):
            with open("calibration/generation_{gen}_{target}_selected.yml".format(gen=gen, target=CALIBRATION_TRACK), "r") as f:
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
        with open("calibration/calibration_{target}_pareto_front.yml".format(target=CALIBRATION_TRACK), "w") as f:
            yaml.dump({"target": CALIBRATION_TRACK, "pareto_front": pareto_data}, f, default_flow_style=False)
