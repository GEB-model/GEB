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
from SALib.sample import sobol as sobol_sample
from SALib.sample import latin as latin_sample

import multiprocessing
from subprocess import Popen, PIPE


def sensitivity_parameters(parameters, distinct_samples, type="saltelli"):
    parameters_list = list(parameters.keys())
    bounds = [
        [param_data["min"], param_data["max"]] for param_data in parameters.values()
    ]

    problem = {"num_vars": len(parameters), "names": parameters_list, "bounds": bounds}

    # Set the nr of samples
    distinct_samples = distinct_samples  # N results in N * (2D + 2) total samples
    if type == "saltelli":
        # Sample the values
        param_values = sobol_sample.sample(problem, distinct_samples, seed=42)
    elif type == "delta":
        param_values = latin_sample.sample(problem, distinct_samples, seed=42)

    return param_values


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

    config, parameters, param_adjustment_factor, run_id = args
    os.makedirs(config["sensitivity"]["path"], exist_ok=True)
    runs_path = os.path.join(config["sensitivity"]["path"], "runs")
    os.makedirs(runs_path, exist_ok=True)
    logs_path = os.path.join(config["sensitivity"]["path"], "logs")
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
        ## Change here to make the relevant parameters into individual_parameters. Use the right names.
        # Create a dictionary of the individual's parameters
        individual_parameters = {}
        for i, parameter_data in enumerate(parameters.values()):
            individual_parameters[parameter_data["variable"]] = param_adjustment_factor[
                i
            ]

        # Create the configuration file for the model run
        config_path = os.path.join(run_directory, "config.yml")
        while True:
            os.mkdir(run_directory)
            template = deepcopy(config)

            template["general"]["report_folder"] = run_directory

            template["general"]["sensitivity_analysis"] = True

            # Update the template configuration file with the individual's parameters
            template["general"]["start_time"] = config["sensitivity"]["start_time"]
            template["general"]["end_time"] = config["sensitivity"]["end_time"]

            template["report"] = {}
            template["report_hydrology"] = {}

            template.update(config["sensitivity"]["output_variables"])

            # loop through individual parameters and set them in the template
            for parameter, value in individual_parameters.items():
                multi_set(template, value, *parameter.split("."))

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

            return_code = run_model_scenario(config["sensitivity"]["scenario"])
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


def sensitivity_analysis(config, working_directory):
    sensitivity_config = config["sensitivity"]
    # use_multiprocessing = sensitivity_config['multiprocessing']['use_multiprocessing']
    config["sensitivity"]["scenario"] = sensitivity_config["scenario"]
    parameters = config["sensitivity"]["parameters"]
    distinct_samples = sensitivity_config["distinct_samples"]
    SA_type = sensitivity_config["SA_type"]

    param_adjustment_factor = sensitivity_parameters(
        parameters, distinct_samples, SA_type
    )

    pool_size = int(
        os.getenv("SLURM_CPUS_PER_TASK")
        or sensitivity_config["multiprocessing"]["pool"]
    )
    print(f"Pool size: {pool_size}")
    # Ignore the interrupt signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Create a tuple of arguments for each run
    args = [
        (config, parameters, param_adjustment_factor[i], i)
        for i in range(len(param_adjustment_factor))
    ]

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
            sensitivity_config["gpus"],
            (
                sensitivity_config["models_per_gpu"]
                if "models_per_gpu" in sensitivity_config
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
