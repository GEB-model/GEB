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

import numpy as np
import yaml

SCENARIO_FILES = [
    "well_combined_insurance.yml",
    "well_index_insurance.yml",
    "well_precipitation_insurance.yml",
    "well_personal_insurance.yml",
    "well_no_insurance.yml",
]

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


@handle_ctrl_c
def run_model(args):
    """
    Runs ONE replicate (run_id) that launches ALL scenarios concurrently.
    Results: <multirun.folder>/runs/<run_id>/<scenario>/
    Logs:    <multirun.folder>/logs/<run_id>/<scenario>/
    """
    config, run_id, scenario_files = args

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
        """
        Runs one scenario with retries. Prints tail of stderr on RC=2/66/1.
        Uses current interpreter via sys.executable, and 'run' subcommand.
        """
        env = os.environ.copy()
        cli_py_path = os.path.join(os.environ.get("GEB_PACKAGE_DIR"), "cli.py")

        venv_activate = "/scistor/ivm/mka483/GEB_p3/GEB/.venvs/geb_p3_new/bin/activate"

        command = (
            f"source {venv_activate} && "
            f"{sys.executable} {cli_py_path} run --config {config_path_for_cli}"
        )

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

    pool_size = int(os.getenv("SLURM_CPUS_PER_TASK") or 10)
    num_scenarios = len(SCENARIO_FILES)

    replicates = max(1, pool_size // num_scenarios)

    print(f"Pool size: {pool_size}")
    print(f"Scenarios per replicate: {num_scenarios}")
    print(f"Replicates to run (folders 0..{replicates - 1}): {replicates}")
    # Ignore the interrupt signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Create a tuple of arguments for each run
    args = [(config, i, SCENARIO_FILES) for i in range(replicates)]

    # Create a manager for multiprocessing
    manager = multiprocessing.Manager()
    # Create a shared variable to keep track of the number of GPUs in use
    current_gpu_use_count = manager.Value("i", 0)
    # Create a lock for managing access to the shared variable
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

    pool.close()

    global ctrl_c_entered
    global default_sigint_handler
    ctrl_c_entered = False
    default_sigint_handler = signal.signal(signal.SIGINT, pool_ctrl_c_handler)
