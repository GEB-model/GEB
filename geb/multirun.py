#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sensitivity analysis for the GEB model
"""

import os
import shutil
import random
import string
import numpy as np
from copy import deepcopy
import signal
import yaml
from functools import wraps

import multiprocessing
from subprocess import Popen, PIPE


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
    This function takes an individual from the population and runs the model with the corresponding parameters.
    It first checks if the run directory already exists and whether the model was run before.
    If the directory exists and the model was run before, it skips running the model.
    Otherwise, it runs the model and saves the results to the run directory.
    """

    config, run_id = args
    os.makedirs(config["multirun"]["folder"], exist_ok=True)
    runs_path = os.path.join(config["multirun"]["folder"], "runs")
    os.makedirs(runs_path, exist_ok=True)
    logs_path = os.path.join(config["multirun"]["folder"], "logs")
    os.makedirs(logs_path, exist_ok=True)

    # Define the directory where the model run will be stored
    run_directory = os.path.join(runs_path, str(run_id))

    # Check if the run directory already exists
    if os.path.isdir(run_directory):
        if os.path.exists(os.path.join(run_directory, "done.txt")):
            runmodel = False
        else:
            runmodel = True
            shutil.rmtree(run_directory)
    else:
        runmodel = True

    if runmodel:
        # Create the configuration file for the model run
        config_path = os.path.join(run_directory, "config.yml")
        while True:
            os.mkdir(run_directory)
            template = deepcopy(config)
            template["general"]["report_folder"] = run_directory

            # write the template to the specified config file
            with open(config_path, "w") as f:
                yaml.dump(template, f)

            # acquire lock to check and set GPU usage
            lock.acquire()
            if current_gpu_use_count.value < n_gpu_spots:
                use_gpu = int(
                    current_gpu_use_count.value
                    / config["calibration"]["DEAP"]["models_per_gpu"]
                )
                current_gpu_use_count.value += 1
                print(
                    f"Using 1 GPU, current_counter: {current_gpu_use_count.value}/{n_gpu_spots}"
                )
            else:
                use_gpu = False
                print(
                    f"Not using GPU, current_counter: {current_gpu_use_count.value}/{n_gpu_spots}"
                )
            lock.release()

            def run_model_scenario(scenario):
                # build the command to run the script, including the use of a GPU if specified
                command = [
                    "geb",
                    "run",
                    "--config",
                    config_path,
                    "--scenario",
                    scenario,
                ]
                if use_gpu is not False:
                    command.extend(["--GPU", "--gpu_device", use_gpu])
                print(command, flush=True)

                # run the command and capture the output and errors
                p = Popen(command, stdout=PIPE, stderr=PIPE)
                output, errors = p.communicate()

                # check the return code of the command and handle accordingly
                if p.returncode == 0:  # model has run successfully
                    with open(
                        os.path.join(logs_path, f"log{str(run_id)}_{scenario}.txt"), "w"
                    ) as f:
                        content = (
                            "OUTPUT:\n"
                            + str(output.decode())
                            + "\nERRORS:\n"
                            + str(errors.decode())
                        )
                        f.write(content)
                    modflow_folder = os.path.join(
                        run_directory, "spinup", "modflow_model"
                    )
                    if os.path.exists(modflow_folder):
                        shutil.rmtree(modflow_folder)

                elif p.returncode == 1:  # model has failed
                    with open(
                        os.path.join(
                            logs_path,
                            f"log{str(run_id)}_{scenario}_{''.join((random.choice(string.ascii_lowercase) for x in range(10)))}.txt",
                        ),
                        "w",
                    ) as f:
                        content = (
                            "OUTPUT:\n"
                            + str(output.decode())
                            + "\nERRORS:\n"
                            + str(errors.decode())
                        )
                        f.write(content)
                    shutil.rmtree(run_directory)

                else:
                    raise ValueError(
                        "Return code of run.py was not 0 or 1, but instead "
                        + str(p.returncode)
                        + "."
                    )

                return p.returncode

            return_code = run_model_scenario(config["multirun"]["scenario"])
            if return_code == 0:
                # release the GPU if it was used
                if use_gpu is not False:
                    lock.acquire()
                    current_gpu_use_count.value -= 1
                    lock.release()
                    print(
                        f"Released 1 GPU, current_counter: {current_gpu_use_count.value}/{n_gpu_spots}"
                    )
                with open(os.path.join(run_directory, "done.txt"), "w") as f:
                    f.write("done")
                break


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
    nr_runs = multi_run_config["run_nrs"]

    pool_size = int(os.getenv("SLURM_CPUS_PER_TASK") or multi_run_config["pool"])
    print(f"Pool size: {pool_size}")
    # Ignore the interrupt signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Create a tuple of arguments for each run
    args = [(config, i) for i in range(nr_runs)]

    # Create a manager for multiprocessing
    manager = multiprocessing.Manager()
    # Create a shared variable to keep track of the number of GPUs in use
    current_gpu_use_count = manager.Value("i", 0)
    # Create a lock for managing access to the shared variable
    manager_lock = manager.Lock()

    with multiprocessing.Pool(
        processes=pool_size,
        initializer=init_pool,
        initargs=(
            current_gpu_use_count,
            manager_lock,
            multi_run_config["gpus"],
            (
                multi_run_config["models_per_gpu"]
                if "models_per_gpu" in multi_run_config
                else 1
            ),
        ),
    ) as pool:
        pool.map(run_model, args)

    pool.close()

    global ctrl_c_entered
    global default_sigint_handler
    ctrl_c_entered = False
    default_sigint_handler = signal.signal(signal.SIGINT, pool_ctrl_c_handler)
