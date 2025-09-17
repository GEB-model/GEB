"""Sensitivity analysis for the GEB model."""

import multiprocessing
import os
import shutil
import signal
from functools import wraps
from subprocess import Popen
import threading
import sys
import time
import datetime
import traceback
from .workflows.io import open_zarr

import numpy as np
import yaml
from pathlib import Path
import zarr
import xarray as xr

SCENARIO_FILES = [
    "well_no_insurance.yml",
    # "well_combined_insurance.yml",
    # "well_index_insurance.yml",
    "well_precipitation_insurance.yml",
    "well_personal_insurance.yml",
]
SCENARIO_NAMES = [os.path.splitext(f)[0] for f in SCENARIO_FILES]

SCENARIO_DIR = "configs_insurance_multirun"


def _tail_file(path, n=80):
    try:
        with open(path, "r", errors="ignore") as f:
            lines = f.readlines()
        return "".join(lines[-n:])
    except Exception as e:
        return f"<could not read {path}: {e}>"


def handle_ctrl_c(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        global ctrl_c_entered
        if not ctrl_c_entered:
            signal.signal(signal.SIGINT, default_sigint_handler)  # the default
            try:
                return func(*args, **kwargs)
            except KeyboardInterrupt:
                ctrl_c_entered = True
                return KeyboardInterrupt
            finally:
                signal.signal(signal.SIGINT, pool_ctrl_c_handler)
        else:
            return KeyboardInterrupt

    return wrapper


def pool_ctrl_c_handler(*args, **kwargs):
    global ctrl_c_entered
    ctrl_c_entered = True


def multi_set(dict_obj, value, *attrs):
    d = dict_obj
    for attr in attrs[:-1]:
        d = d[attr]
    if attrs[-1] not in d:
        raise KeyError(f"Key {attrs} does not exist in config file.")

    # Check if the value is a numpy scalar and convert it if necessary
    if isinstance(value, np.generic):
        value = value.item()

    d[attrs[-1]] = value


def summarize_multirun(
    config,
    replicates,
):
    # load in data per parameter, summarize, write and do next
    # First check which variables & nr of runs
    folder = config["multirun"]["folder"]
    base_directory = Path(folder) / "runs"

    def list_params_from_any_run(base_directory: Path):
        candidates = [p.name for p in base_directory.iterdir() if p.is_dir()]
        if not candidates:
            raise FileNotFoundError(f"No subfolders found in {base_directory}")
        chosen = sorted(candidates)[0]
        target = base_directory / chosen / "report" / chosen / "agents.crop_farmers"
        if not target.is_dir():
            raise FileNotFoundError(f"Expected directory not found: {target}")
        names = []
        for p in target.iterdir():
            if p.is_dir():
                n = p.name[:-5] if p.name.endswith(".zarr") else p.name
                names.append(n)
        names.sort()
        return chosen, names

    chosen_name, params = list_params_from_any_run(base_directory / "0")
    STD_DDOF = 0
    module_name = "agents.crop_farmers"

    for key in SCENARIO_NAMES:
        boolean = False
        for param in params:
            print("Scenario:", key, ", Parameter:", param)
            # --- collect + aggregate runs (same as before) ---
            da_runs = []
            param_file = param + ".zarr"
            for run in range(replicates):
                run_dir = (
                    base_directory
                    / str(run)
                    / key
                    / "report"
                    / key
                    / module_name
                    / param_file
                )
                da = open_zarr(run_dir)  # DataArray (time, agents)
                da_runs.append(da.transpose("time", "agents"))

            da_all = xr.concat(da_runs, dim="run")

            is_pm1 = da_all.isin([-1, 1]).fillna(True).all()
            # Make this a plain bool even if it's a dask-backed array
            is_pm1 = bool(getattr(is_pm1.data, "compute", lambda: is_pm1.data)())

            # Compute mean/std across runs (works for both numeric and ±1 data)
            mean_da = da_all.astype(np.float64).mean("run")
            std_da = da_all.astype(np.float64).std("run", ddof=STD_DDOF)

            # Get numpy arrays (handles eager or dask-backed data)
            mean_np = np.asarray(
                getattr(mean_da.data, "compute", lambda: mean_da.data)()
            ).astype(np.float32, copy=False)
            std_np = np.asarray(
                getattr(std_da.data, "compute", lambda: std_da.data)()
            ).astype(np.float32, copy=False)

            # If this parameter is strictly ±1-valued data, binarize the mean to {-1, 1}
            # Tie-breaking: mean == 0 -> 1 (adjust to >0 if you prefer ties to go to -1 or stay 0)
            if is_pm1:
                mean_np = np.where(mean_np >= 0, 1, -1).astype(np.int8, copy=False)
                if param == "well_adaptation":
                    pass

            T, A = mean_np.shape

            # --- TEMPLATE: read from run 0's param.zarr ---
            template_path = (
                base_directory
                / "0"
                / key
                / "report"
                / key
                / module_name
                / f"{param}.zarr"
            )
            template_store = zarr.storage.LocalStore(template_path, read_only=True)
            template_group = zarr.open_group(template_store, mode="r")

            # reuse time from template (slice if your aggregated time is shorter)
            template_time = np.asarray(template_group["time"][:], dtype=np.int64)
            # if needed, rebuild time from DA to match the slice exactly
            out_time = mean_da["time"].astype("datetime64[s]").astype(np.int64).values
            if out_time.shape != template_time.shape or not np.array_equal(
                out_time, template_time[: out_time.shape[0]]
            ):
                # prefer DA-derived time so it matches the aggregated data slice
                time_to_write = out_time
            else:
                time_to_write = template_time[:T]

            # get chunking from first non-time array in the template
            # (keeps the "structure" closer to your originals)
            non_time_keys = [k for k in template_group.array_keys() if k != "time"]
            template_chunks = None
            if non_time_keys:
                template_chunks = template_group[
                    non_time_keys[0]
                ].chunks  # e.g., (time_chunk, agent_chunk)

            # --- OUTPUT: open in append mode, reuse what exists, only add/overwrite what's needed ---
            filepath = Path(folder) / "report" / key / module_name / (param + ".zarr")
            filepath.parent.mkdir(parents=True, exist_ok=True)
            out_store = zarr.storage.LocalStore(filepath, read_only=False)
            z = zarr.open_group(out_store, mode="a")

            # TIME: create only if missing; else verify length
            if "time" not in z:
                time_arr = z.create_array(
                    "time",
                    shape=time_to_write.shape,
                    dtype=time_to_write.dtype,
                    dimension_names=["time"],
                )
                time_arr[:] = time_to_write
                time_arr.attrs.update(
                    {
                        "standard_name": "time",
                        "units": "seconds since 1970-01-01T00:00:00",
                        "calendar": "gregorian",
                    }
                )
            else:
                # optional: sanity check
                if z["time"].shape[0] != time_to_write.shape[0]:
                    # overwrite to keep it consistent with this slice
                    del z["time"]
                    time_arr = z.create_array(
                        "time",
                        shape=time_to_write.shape,
                        dtype=time_to_write.dtype,
                        dimension_names=["time"],
                    )
                    time_arr[:] = time_to_write
                    time_arr.attrs.update(
                        {
                            "standard_name": "time",
                            "units": "seconds since 1970-01-01T00:00:00",
                            "calendar": "gregorian",
                        }
                    )

            # MEAN: (delete if exists, then recreate with same chunking style as template)
            if "mean" in z:
                del z["mean"]
            mean_arr = z.create_array(
                "mean",
                shape=(T, A),
                dtype=mean_np.dtype,
                chunks=template_chunks,  # <-- reuse template chunking if available
                dimension_names=["time", "agents"],
            )
            mean_arr[:] = mean_np
            mean_arr.attrs.update(
                {
                    "statistic": "mean",
                    **(
                        {"units": da_all.attrs["units"]}
                        if "units" in da_all.attrs
                        else {}
                    ),
                }
            )

            # STDEV:
            if "stdev" in z:
                del z["stdev"]
            std_arr = z.create_array(
                "stdev",
                shape=(T, A),
                dtype=std_np.dtype,
                chunks=template_chunks,
                dimension_names=["time", "agents"],
            )
            std_arr[:] = std_np
            std_arr.attrs.update(
                {
                    "statistic": "stdev",
                    "ddof": STD_DDOF,
                    **(
                        {"units": da_all.attrs["units"]}
                        if "units" in da_all.attrs
                        else {}
                    ),
                }
            )


@handle_ctrl_c
def run_model(args):
    """
    Runs ONE replicate (run_id) that launches ALL scenarios concurrently.
    Results: <multirun.folder>/runs/<run_id>/<scenario>/
    Logs:    <multirun.folder>/logs/<run_id>/<scenario>/
    """
    config, run_id, scenario_files, cpus_per_scenario = args

    base_folder = config["multirun"]["folder"]
    os.makedirs(base_folder, exist_ok=True)

    runs_root = os.path.join(base_folder, "runs")
    os.makedirs(runs_root, exist_ok=True)

    logs_root = os.path.join(base_folder, "logs")
    os.makedirs(logs_root, exist_ok=True)
    logs_run_path = os.path.join(logs_root, str(run_id))
    os.makedirs(logs_run_path, exist_ok=True)

    # Per-run results folder (e.g., .../runs/0)
    run_directory = os.path.join(runs_root, str(run_id))
    os.makedirs(run_directory, exist_ok=True)

    def _tail_file(path, n=80):
        try:
            with open(path, "r", errors="ignore") as f:
                lines = f.readlines()
            return "".join(lines[-n:])
        except Exception as e:
            return f"<could not read {path}: {e}>"

    def run_model_scenario(
        config_path_for_cli: str, scenario_label: str, scenario_logs_dir: str
    ):
        env = os.environ.copy()

        # NEW: cap threads so the scenario respects cpus_per_scenario
        thread_cap = str(max(1, int(cpus_per_scenario)))
        for k in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "BLIS_NUM_THREADS",
        ):
            env[k] = thread_cap
        env.setdefault("OMP_PROC_BIND", "close")
        env.setdefault("OMP_PLACES", "cores")

        cli_py_path = os.path.join(os.environ.get("GEB_PACKAGE_DIR"), "cli.py")
        venv_activate = "/scistor/ivm/mka483/GEB_p3/GEB/.venvs/geb_p3_new/bin/activate"

        # Keep sourcing the venv, and use your current interpreter
        inner_shell = (
            f"source {venv_activate} && "
            f"{sys.executable} {cli_py_path} run --config {config_path_for_cli}"
        )

        # NEW: if under SLURM, give this scenario its own CPUs via srun; else plain bash -lc
        if "SLURM_JOB_ID" in os.environ:
            command = (
                "srun --ntasks=1 "
                f"--cpus-per-task={cpus_per_scenario} "
                "--exclusive --cpu-bind=cores "
                f"bash -lc '{inner_shell}'"
            )
        else:
            command = f"bash -lc '{inner_shell}'"

        print(f"[rep {run_id} | {scenario_label}] Executing: {command}", flush=True)

        max_retries = 10000
        retries = 0

        while retries <= max_retries:
            attempt = retries + 1

            out_file_path = os.path.join(
                scenario_logs_dir,
                f"model_out_run_{run_id}_{scenario_label}_attempt{attempt}.txt",
            )
            err_file_path = os.path.join(
                scenario_logs_dir,
                f"model_err_run_{run_id}_{scenario_label}_attempt{attempt}.txt",
            )
            retry_log_path = os.path.join(
                scenario_logs_dir, f"retry_log_run_{run_id}_{scenario_label}.txt"
            )

            with (
                open(out_file_path, "w") as out_file,
                open(err_file_path, "w") as err_file,
            ):
                p = Popen(
                    command,
                    stdout=out_file,
                    stderr=err_file,
                    shell=True,
                    executable="/bin/bash",
                    env=env,
                )
                p.wait()

            if p.returncode == 0:
                return 0
            elif p.returncode == 1:
                err_tail = _tail_file(err_file_path, 80)
                print(
                    f"[rep {run_id} | {scenario_label}] RC=1 (no retry). "
                    f"Last stderr lines:\n{err_tail}\n--- end stderr ---",
                    flush=True,
                )
                return 1
            elif p.returncode in (2, 66):
                err_tail = _tail_file(err_file_path, 80)
                msg = (
                    f"[rep {run_id} | {scenario_label}] RC={p.returncode} "
                    f"(attempt {attempt}/{max_retries}). Last stderr lines:\n"
                    f"{err_tail}\n--- end stderr ---"
                )
                print(msg, flush=True)
                try:
                    with open(retry_log_path, "a") as rf:
                        rf.write(
                            f"\n==== Attempt {attempt} @ {datetime.datetime.now().isoformat()} ====\n{msg}\n"
                        )
                except Exception:
                    pass

                retries += 1
                if retries > max_retries:
                    break
                time.sleep(1)
                continue
            else:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_filename = os.path.join(
                    scenario_logs_dir, f"log_error_run_{run_id}_{scenario_label}.txt"
                )
                with open(log_filename, "w") as f:
                    content = (
                        f"Timestamp: {timestamp}\n"
                        f"Process ID: {os.getpid()}\n"
                        f"Command: {command}\n\n"
                        f"Return code: {p.returncode}\n"
                        f"Traceback:\n{traceback.format_exc()}"
                    )
                    f.write(content)
                raise ValueError(
                    f"[rep {run_id} | {scenario_label}] RC={p.returncode}. "
                    f"See {log_filename}."
                )

        raise ValueError(
            f"[rep {run_id} | {scenario_label}] RC 2/66 received {max_retries} times. See logs."
        )

    # --- Launch all scenarios concurrently for this replicate ---
    threads = []
    for fname in scenario_files:
        scenario_label = os.path.splitext(fname)[0]
        scenario_src_path = os.path.join(SCENARIO_DIR, fname)

        # results folder for this scenario
        scenario_dir = os.path.join(run_directory, scenario_label)
        scenario_done_path = os.path.join(scenario_dir, "done.txt")

        scenario_logs_dir = os.path.join(logs_run_path, scenario_label)
        os.makedirs(scenario_logs_dir, exist_ok=True)

        # prepare results folder
        if os.path.isdir(scenario_dir):
            if os.path.exists(scenario_done_path):
                print(f"[rep {run_id} | {scenario_label}] already done. Skipping.")
                continue
            else:
                shutil.rmtree(scenario_dir)
        os.makedirs(scenario_dir, exist_ok=True)

        # write scenario config with output_folder pointing to scenario_dir
        with open(scenario_src_path, "r") as f:
            template = yaml.safe_load(f) or {}
        if "general" not in template or not isinstance(template["general"], dict):
            template["general"] = {}
        template["general"]["output_folder"] = scenario_dir

        scenario_config_path = os.path.join(scenario_dir, "config.yml")
        with open(scenario_config_path, "w") as f:
            yaml.safe_dump(template, f, sort_keys=False)

        def _runner(
            config_path=scenario_config_path,
            label=scenario_label,
            logs_dir=scenario_logs_dir,
            done_path=scenario_done_path,
        ):
            rc = run_model_scenario(config_path, label, logs_dir)
            if rc == 0:
                with open(done_path, "w") as f:
                    f.write("done")
                print(f"[rep {run_id} | {label}] done.")
            else:
                print(f"[rep {run_id} | {label}] failed with RC={rc}.")

        t = threading.Thread(target=_runner, daemon=False)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()


def init_pool(manager_current_gpu_use_count, manager_lock, gpus, models_per_gpu):
    # set global variable for each process in the pool:
    global ctrl_c_entered
    global default_sigint_handler
    ctrl_c_entered = False
    default_sigint_handler = signal.signal(signal.SIGINT, pool_ctrl_c_handler)

    global lock
    global current_gpu_use_count
    global n_gpu_spots
    n_gpu_spots = gpus * models_per_gpu
    lock = manager_lock
    current_gpu_use_count = manager_current_gpu_use_count


def multi_run(config, working_directory):
    multi_run_config = config["multirun"]
    DEFAULT_CPUS_PER_SCENARIO = int(os.getenv("CPUS_PER_SCENARIO", "1"))
    pool_size = int(os.getenv("SLURM_CPUS_PER_TASK") or 1)
    num_scenarios = len(SCENARIO_FILES)
    only_summarize = multi_run_config["only_summarize"]
    # read from config first, fallback to env, then 1
    cpus_per_scenario = int(
        multi_run_config.get("cpus_per_scenario", DEFAULT_CPUS_PER_SCENARIO)
    )

    # avoid oversubscription: reps * scenarios * cpus_per_scenario <= pool_size
    replicates = max(1, pool_size // (num_scenarios * cpus_per_scenario))

    if replicates * num_scenarios * cpus_per_scenario > pool_size:
        print("[warn] computed replicates oversubscribe CPUs; throttling.", flush=True)

    if only_summarize:
        replicates = multi_run_config["nr_replicates"]
        summarize_multirun(config, replicates)
    else:
        print(f"Pool size (SLURM_CPUS_PER_TASK): {pool_size}")
        print(f"CPUs per scenario: {cpus_per_scenario}")
        print(f"Scenarios per replicate: {num_scenarios}")
        print(f"Replicates to run (folders 0..{replicates - 1}): {replicates}")

        signal.signal(signal.SIGINT, signal.SIG_IGN)

        # each arg = one replicate with all scenarios + cpus_per_scenario
        args = [
            (config, i, SCENARIO_FILES, cpus_per_scenario) for i in range(replicates)
        ]

        manager = multiprocessing.Manager()
        current_gpu_use_count = manager.Value("i", 0)
        manager_lock = manager.Lock()

        processes = min(pool_size, len(args)) if len(args) > 0 else 1

        with multiprocessing.Pool(
            processes=processes,
            initializer=init_pool,
            initargs=(
                current_gpu_use_count,
                manager_lock,
                multi_run_config.get("gpus", 0),
                multi_run_config.get("models_per_gpu", 1),
            ),
        ) as pool:
            pool.map(run_model, args)

        # no need to pool.close() inside 'with'

        global ctrl_c_entered, default_sigint_handler
        ctrl_c_entered = False
        default_sigint_handler = signal.signal(signal.SIGINT, pool_ctrl_c_handler)

        summarize_multirun(config, replicates)
