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
RUNS_DIR = f"calibration/{CALIBRATION_TRACK}"

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
weights = [target_info["weight"] for target_info in CALIBRATION_TARGETS.values()]
for weight in weights:
    if not isinstance(weight, (int, float)):
        raise ValueError(f"Invalid weight value: {weight}. Weights must be numeric.")

# --- DEAP Initialization ---

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

def get_calibration_config(model_config: dict) -> dict:
    """Helper to get the calibration section for the current target."""
    return model_config.get("calibration", {}).get(CALIBRATION_TARGET, {})

def run_command(cmd: str, log_path: str, error_msg: str = "GEB command failed") -> str | None:
    """Helper function to run a GEB command, log output, and return captured stdout."""
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

rule build_base:
    input: "base_init.done"
    output: "input/build_complete.txt"
    log: "logs/snakemake_base_build.log"
    message: "Building base GEB model..."
    run: 
        run_command("geb build --continue", log[0], "Failed to build base model")

rule generate_initial_parameters:
    input: "input/build_complete.txt"
    output: "calibration/generation_0_{0}_population.yml".format(CALIBRATION_TRACK)
    message: "Generating initial DEAP population for {CALIBRATION_TRACK}..."
    run:
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

rule generate_flood_maps:
    input: "base_build.done"
    output: touch(RUNS_DIR + "/flood_maps.done")
    log: RUNS_DIR + "/logs/generate_flood_maps.log"
    run:
        # Copy only the configured return-period maps (e.g. 2.zarr, 25.zarr)
        # from canonical model output into the calibration run directory.
        from pathlib import Path
        import shutil
        from geb.runner import parse_config

        # Locate the canonical flood_maps folder by probing likely repo roots
        def _find_flood_maps_folder() -> tuple[Path, list[Path]]:
            candidates: list[Path] = []
            try:
                f = Path(__file__).resolve()
                # add a few parents of the rule file
                for i in range(1, 6):
                    if len(f.parents) >= i:
                        candidates.append(f.parents[i - 1])
            except Exception:
                pass

            # add current working directory and its parents
            cwd = Path.cwd().resolve()
            candidates.append(cwd)
            for i in range(1, 6):
                if len(cwd.parents) >= i:
                    candidates.append(cwd.parents[i - 1])

            # dedupe while preserving order
            seen = set()
            uniq: list[Path] = []
            for c in candidates:
                if c not in seen:
                    seen.add(c)
                    uniq.append(c)

            for root in uniq:
                p = root / "models" / "geul" / "output" / "flood_maps"
                if p.exists():
                    return p, uniq

            # fallback to relative path (may still fail)
            return Path("models") / "geul" / "output" / "flood_maps", uniq

        src, probed_candidates = _find_flood_maps_folder()
        # Diagnostic: write out probed candidates and resolved src
        try:
            with open(log[0], "a") as fh:
                fh.write(f"Probed candidates for flood_maps (in order):\n")
                for c in probed_candidates:
                    fh.write(str(c) + "\n")
                fh.write(f"Resolved src path: {src}\n")
                fh.write(f"src.exists: {src.exists()}\n")
                # list src contents if present
                if src.exists():
                    try:
                        fh.write("Contents of src:\n")
                        for p in sorted(src.iterdir()):
                            fh.write(" - " + p.name + ("/" if p.is_dir() else "") + "\n")
                    except Exception as e:
                        fh.write("Failed to list src contents: " + str(e) + "\n")
        except Exception:
            pass
        dst = Path(RUNS_DIR) / "output" / "flood_maps"

        if not src.exists():
            with open(log[0], "a") as fh:
                fh.write(f"Source flood maps not found at {src}\n")
            raise RuntimeError(
                f"Source flood maps not found at {src}. Run 'geb exec estimate_return_periods -wd models/geul' first."
            )

        # read return periods using parse_config so 'inherits' is resolved
        try:
            # Ensure any `{GEB_PACKAGE_DIR}` inherits can be resolved
            import os
            from geb import GEB_PACKAGE_DIR
            os.environ.setdefault("GEB_PACKAGE_DIR", str(GEB_PACKAGE_DIR))

            # Prefer any of the previously probed candidate roots to find the
            # repository `models/geul/model.yml`. When Snakemake loads rules
            # from site-packages, `__file__` can point into the virtualenv, so
            # using fixed parents[] is brittle. Use `probed_candidates` (from
            # _find_flood_maps_folder) to locate the real repo root.
            model_cfg_path = None
            try:
                # If `src` (the canonical flood_maps folder) was resolved above,
                # derive the repository model.yml from it. `src` should be at:
                # <repo_root>/models/geul/output/flood_maps, so `src.parents[2]`
                # points to the `geul` folder.
                if src.exists():
                    candidate = src.parents[2] / "model.yml"
                    if candidate.exists():
                        model_cfg_path = candidate
                if model_cfg_path is None:
                    for root in probed_candidates:
                        candidate = Path(root) / "models" / "geul" / "model.yml"
                        if candidate.exists():
                            model_cfg_path = candidate
                            break
            except Exception:
                model_cfg_path = None

            if model_cfg_path is None:
                # fallback: relative path from the repository working dir
                model_cfg_path = Path("models") / "geul" / "model.yml"

            # Diagnostic: record model_cfg_path and existence
            try:
                with open(log[0], "a") as fh:
                    fh.write(f"Attempting parse_config on: {model_cfg_path}\n")
                    fh.write(f"model_cfg_path.exists: {model_cfg_path.exists()}\n")
                    if model_cfg_path.exists():
                        try:
                            fh.write("--- model.yml (safe load) ---\n")
                            import yaml as _yaml
                            fh.write(_yaml.safe_load(model_cfg_path.read_text()).__repr__() + "\n")
                        except Exception as e:
                            fh.write("safe_load of model.yml failed: " + str(e) + "\n")
            except Exception:
                pass

            model_cfg = parse_config(model_cfg_path, current_directory=model_cfg_path.parent)
            # write parse_config result for debugging
            try:
                with open(log[0], "a") as fh:
                    fh.write("--- parse_config result keys ---\n")
                    if isinstance(model_cfg, dict):
                        for k in sorted(model_cfg.keys()):
                            fh.write(str(k) + "\n")
                    else:
                        fh.write(f"parse_config returned non-dict: {type(model_cfg)}\n")
            except Exception:
                pass

            return_periods = model_cfg.get("hazards", {}).get("floods", {}).get("return_periods", [])
        except Exception as e:
            try:
                with open(log[0], "a") as fh:
                    fh.write("parse_config raised exception:\n")
                    import traceback as _tb
                    fh.write(_tb.format_exc())
            except Exception:
                pass
            return_periods = []

        if not return_periods:
            with open(log[0], "a") as fh:
                fh.write("No return periods configured in models/geul/model.yml; nothing to copy.\n")
            raise RuntimeError("No return periods found in models/geul/model.yml")

        missing = []
        dst.parent.mkdir(parents=True, exist_ok=True)
        for rp in return_periods:
            src_rp = src / f"{rp}.zarr"
            dst_rp = dst / f"{rp}.zarr"
            if not src_rp.exists():
                missing.append(str(src_rp))
                continue
            if dst_rp.exists():
                shutil.rmtree(dst_rp)
            shutil.copytree(src_rp, dst_rp)

        if missing:
            with open(log[0], "a") as fh:
                fh.write("Missing return-period maps:\n")
                for m in missing:
                    fh.write(m + "\n")
            raise RuntimeError(
                "Not all expected return-period maps were found in models/geul/output/flood_maps."
            )

        Path(output[0]).touch()

rule generate_individual_parameters:
    input:
        pop_file=lambda wildcards: "calibration/generation_0_{0}_population.yml".format(CALIBRATION_TRACK) if int(wildcards.gen) == 0 else "calibration/generation_{0}_{1}_next.yml".format(int(wildcards.gen) - 1, CALIBRATION_TRACK),
        checkpoint_done=lambda wildcards: checkpoints.create_next_generation.get(gen=int(wildcards.gen) - 1).output if int(wildcards.gen) > 0 else []
    output: params=RUNS_DIR + "/{gen}_{ind}/parameters.yml"
    run:
        print(get_progress_message(wildcards, "Defining parameters"))
        with open(input.pop_file, "r") as f:
            pop_data = yaml.safe_load(f)
        
        label = f"{wildcards.gen}_{wildcards.ind}"
        individual_data = next((ind for ind in pop_data["individuals"] if ind["label"] == label), None)
        if individual_data is None:
            raise ValueError(f"Individual {label} not found in population file")
        
        actual_params = {}
        for i, (param_name, param_config) in enumerate(PARAMETERS.items()):
            # compute raw value in the parameter's range
            raw_value = param_config["min"] + individual_data["values"][i] * (param_config["max"] - param_config["min"])
            # preserve float parameters if min or max is a float, otherwise use integer
            try:
                is_float = isinstance(param_config["min"], float) or isinstance(param_config["max"], float)
            except Exception:
                is_float = False

            if is_float:
                actual = float(raw_value)
            else:
                # round to nearest integer for integer parameters
                actual = int(round(raw_value))

            actual_params[param_config["variable"]] = actual
        
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
        base_build="input/build_complete.txt"
    output: touch(RUNS_DIR + "/{gen}_{ind}/init.done")
    log: RUNS_DIR + "/{gen}_{ind}/logs/snakemake_init.log"
    run:
        print(get_progress_message(wildcards, "Initializing folder"))
        run_dir = Path(RUNS_DIR) / f"{wildcards.gen}_{wildcards.ind}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        with open(run_dir / "build.yml", "w") as f: f.write("# Empty build config\n")
        with open(run_dir / "update.yml", "w") as f: f.write("# Empty update config\n")

        # Ensure the run has an `output/flood_maps` path that points to the
        # centralized calibration flood maps. Some environments (e.g. HPC
        # workers) may not allow symlinks; in that case we fall back to copying
        # the maps into the run dir so `agents.households` can read them at
        # `run_dir/output/flood_maps/{rp}.zarr`.
        try:
            out_dir = run_dir / "output"
            out_dir.mkdir(parents=True, exist_ok=True)
            link = out_dir / "flood_maps"
            target = Path(RUNS_DIR) / "output" / "flood_maps"
            if link.exists() or link.is_symlink():
                try:
                    if link.is_dir() and not link.is_symlink():
                        import shutil
                        shutil.rmtree(link)
                    else:
                        link.unlink()
                except Exception:
                    pass

            try:
                # create relative symlink when possible
                rel = os.path.relpath(str(target), start=str(out_dir))
                link.symlink_to(rel)
            except Exception:
                # fallback: copy the folder contents if target exists
                import shutil
                if target.exists():
                    try:
                        shutil.copytree(target, link)
                    except Exception:
                        pass
        except Exception:
            pass

rule alter_individual:
    input: RUNS_DIR + "/{gen}_{ind}/init.done"
    output: RUNS_DIR + "/{gen}_{ind}/input/build_complete.txt"
    log: RUNS_DIR + "/{gen}_{ind}/logs/snakemake_alter.log"
    run:
        print(get_progress_message(wildcards, "Altering model configuration"))
        run_dir = Path(RUNS_DIR) / f"{wildcards.gen}_{wildcards.ind}"
        cmd = f"geb alter --from-model ../../../. -wd {run_dir}"
        run_command(cmd, log[0], f"Failed to alter model {wildcards.gen}_{wildcards.ind}")

rule set_individual_parameters:
    input:
        altered_done=RUNS_DIR + "/{gen}_{ind}/input/build_complete.txt",
        parameters=RUNS_DIR + "/{gen}_{ind}/parameters.yml",
    output: touch(RUNS_DIR + "/{gen}_{ind}/params_set.done")
    log: RUNS_DIR + "/{gen}_{ind}/logs/snakemake_set_params.log"
    run:
        print(get_progress_message(wildcards, "Setting param values"))
        with open(input.parameters, "r") as f: params_data = yaml.safe_load(f)
        actual_params = params_data["parameters"]
        run_dir = Path(RUNS_DIR) / f"{wildcards.gen}_{wildcards.ind}"
        
        param_args = " ".join(f"{k}={v}" for k, v in actual_params.items())
        model_config = parse_config(run_dir / "model.yml")
        
        datetime_args = "general.spinup_time={0} general.start_time={1} general.end_time={2}".format(
            CALIBRATION_CONFIG["spinup_time"], CALIBRATION_CONFIG["start_time"], CALIBRATION_CONFIG["end_time"]
        )
        cmd = "geb set -c model.yml --working-directory {0} {1} {2} report=null report._config.chunk_target_size_bytes+=100000000 report._config.compression_level+=9 report._discharge_stations+=true report._calibration+=true".format(run_dir, param_args, datetime_args)
        run_command(cmd, log[0], f"Failed to set parameters for {wildcards.gen}_{wildcards.ind}")

rule spinup_individual:
    input:
        params=RUNS_DIR + "/{gen}_{ind}/params_set.done",
        flood_maps=RUNS_DIR + "/flood_maps.done"
    output: touch(RUNS_DIR + "/{gen}_{ind}/spinup.done")
    log: RUNS_DIR + "/{gen}_{ind}/logs/snakemake_spinup.log"
    run:
        print(get_progress_message(wildcards, "Performing spinup"))
        run_dir = Path(RUNS_DIR) / f"{wildcards.gen}_{wildcards.ind}"
        # Diagnostic: record whether run_dir/output/flood_maps exists and list contents
        try:
            with open(log[0], "a") as fh:
                fh.write("\n--- Diagnostic: flood_maps visibility before spinup ---\n")
                fh.write(f"run_dir: {run_dir}\n")
                out_fm = run_dir / "output" / "flood_maps"
                fh.write(f"run_dir/output exists: {(run_dir / 'output').exists()}\n")
                fh.write(f"run_dir/output/flood_maps exists: {out_fm.exists()}\n")
                fh.write(f"run_dir/output/flood_maps is_symlink: {out_fm.is_symlink()}\n")
                try:
                    if out_fm.exists():
                        fh.write("Contents:\n")
                        for p in sorted(out_fm.iterdir()):
                            fh.write(" - " + p.name + ("/" if p.is_dir() else "") + "\n")
                    else:
                        fh.write("run_dir/output/flood_maps not present\n")
                except Exception as e:
                    fh.write(f"Listing run_dir/output/flood_maps failed: {e}\n")

                # also list the centralized calibration flood_maps
                try:
                    central = Path(RUNS_DIR) / "output" / "flood_maps"
                    fh.write(f"central flood_maps: {central}\n")
                    fh.write(f"central exists: {central.exists()}\n")
                    if central.exists():
                        fh.write("Central contents:\n")
                        for p in sorted(central.iterdir()):
                            fh.write(" - " + p.name + ("/" if p.is_dir() else "") + "\n")
                except Exception as e:
                    fh.write(f"Listing central flood_maps failed: {e}\n")
                fh.write("--- end diagnostic ---\n\n")
        except Exception:
            pass

        run_command(f"geb spinup -wd {run_dir}", log[0], f"Spinup failed for {wildcards.gen}_{wildcards.ind}")

rule run_individual:
    input: RUNS_DIR + "/{gen}_{ind}/spinup.done"
    output: touch(RUNS_DIR + "/{gen}_{ind}/run.done")
    log: RUNS_DIR + "/{gen}_{ind}/logs/snakemake_run.log"
    run:
        print(get_progress_message(wildcards, "Performing run"))
        run_dir = Path(RUNS_DIR) / f"{wildcards.gen}_{wildcards.ind}"
        run_command(f"geb run -wd {run_dir}", log[0], f"Main run failed for {wildcards.gen}_{wildcards.ind}")
        shutil.rmtree(run_dir / "simulation_root")

rule evaluate_individual:
    input: RUNS_DIR + "/{gen}_{ind}/run.done"
    output: RUNS_DIR + "/{gen}_{ind}/fitness.yml"
    log: RUNS_DIR + "/{gen}_{ind}/logs/snakemake_evaluate.log"
    run:
        print(get_progress_message(wildcards, "Evaluating fitness"))
        run_dir = Path(RUNS_DIR) / f"{wildcards.gen}_{wildcards.ind}"
        model_config = parse_config(run_dir / "model.yml")
        targets = CALIBRATION_CONFIG.get("calibration_targets", {})
        if not targets:
            raise ValueError(f"No targets found for {CALIBRATION_TRACK} in model.yml")

        all_metrics = {}

        for target_name, target_info in targets.items():
            method, metric_key, weight = target_info["method"], target_info["metric"], float(target_info["weight"])
            cmd = "geb evaluate {0} -wd {1}".format(method, run_dir)
            stdout = run_command(cmd, log[0], "Eval failed for {0}".format(target_name))

            try:
                # The output should be a JSON string on the last line
                lines = [l.strip() for l in stdout.strip().split("\n") if l.strip()]
                metrics_dict = json.loads(lines[-1])
                metric_value = float(metrics_dict[metric_key])
                all_metrics[target_name] = {"value": metric_value, "metric": metric_key, "method": method, "weight": weight}
            except Exception as e:
                raise RuntimeError("Failed to parse metrics for {0}: {1}. Raw output: {2}".format(target_name, e, stdout))

        with open(output[0], "w") as f:
            yaml.dump({"targets": all_metrics}, f)

        shutil.rmtree(run_dir / "input")

checkpoint create_next_generation:
    input:
        fitness_files=lambda wildcards: [RUNS_DIR + "/{gen}_{i:03d}/fitness.yml".format(gen=wildcards.gen, i=i) for i in range(MU if int(wildcards.gen) == 0 else LAMBDA)],
        prev_pop=lambda wildcards: "calibration/generation_0_{0}_population.yml".format(CALIBRATION_TRACK) if int(wildcards.gen) == 0 else "calibration/generation_{gen_prev}_{target}_next.yml".format(gen_prev=int(wildcards.gen) - 1, target=CALIBRATION_TRACK)
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
                fit_tuple = [t["value"] for t in fitness_data["targets"].values()]
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
