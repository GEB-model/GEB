#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Calibration tool for Hydrological models
using a distributed evolutionary algorithms in python
DEAP library
https://github.com/DEAP/deap/blob/master/README.md

Félix-Antoine Fortin, François-Michel De Rainville, Marc-André Gardner, Marc Parizeau and Christian Gagné, "DEAP: Evolutionary Algorithms Made Easy", Journal of Machine Learning Research, vol. 13, pp. 2171-2175, jul 2012

The calibration tool was created by Hylke Beck 2014 (JRC, Princeton) hylkeb@princeton.edu
Thanks Hylke for making it available for use and modification
Modified by Peter Burek and Jens de Bruijn

The submodule Hydrostats was created 2011 by:
Sat Kumar Tomer (modified by Hylke Beck)
Please see his book "Python in Hydrology"   http://greenteapress.com/pythonhydro/pythonhydro.pdf

"""
import os
import shutil
import hydroStats
import array
import random
import string
import numpy as np
import signal
import pandas as pd
import yaml
from deap import creator, base, tools, algorithms
from datetime import datetime, timedelta
from functools import wraps

import multiprocessing
from subprocess import Popen, PIPE

import pickle
from calconfig import config, args

calibration_config = config['calibration']

OBJECTIVE = 'KGE'

dischargetss = os.path.join('spinup', 'var.discharge_daily.tss')

calibration_path = calibration_config['path']
os.makedirs(calibration_path, exist_ok=True)
runs_path = os.path.join(calibration_path, 'runs')
os.makedirs(runs_path, exist_ok=True)
logs_path = os.path.join(calibration_path, 'logs')
os.makedirs(logs_path, exist_ok=True)

use_multiprocessing = calibration_config['DEAP']['use_multiprocessing']

select_best_n_individuals = calibration_config['DEAP']['select_best']

ngen = calibration_config['DEAP']['ngen']
mu = calibration_config['DEAP']['mu']
lambda_ = calibration_config['DEAP']['lambda_']

# Load observed streamflow
if 'gauges' in config['general']:
	gauges = config['general']['gauges']
else:
	gauges = config['general']['poor_point']
streamflow_path = os.path.join(config['general']['original_data'], 'calibration', 'streamflow', f"{gauges['lon']} {gauges['lat']}.csv")
streamflow_data = pd.read_csv(streamflow_path, sep=",", parse_dates=True, index_col=0)
observed_streamflow = streamflow_data["flow"]
observed_streamflow.name = 'observed'
assert (observed_streamflow >= 0).all()

def handle_ctrl_c(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        global ctrl_c_entered
        if not ctrl_c_entered:
            signal.signal(signal.SIGINT, default_sigint_handler) # the default
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
	if not attrs[-1] in d:
		raise KeyError(f"Key {attrs} does not exist in config file.")
	d[attrs[-1]] = value

def get_irrigation_equipment_score(run_directory, individual):
	fp = os.path.join(run_directory, 'spinup', 'well_irrigated_per_district.csv')
	df = pd.read_csv(fp, index_col=0, parse_dates=True)
	df.at[pd.Timestamp(2010, 6, 1), '30']
	return random.random()

def get_discharge_score(run_directory, individual):
    # Get the path of the simulated streamflow file
	Qsim_tss = os.path.join(run_directory, dischargetss)
	
	# Check if the simulated streamflow file exists
	if not os.path.isfile(Qsim_tss):
		print("run_id: " + str(individual.label)+" File: "+ Qsim_tss)
		raise Exception("No simulated streamflow found. Is the data exported in the ini-file (e.g., 'OUT_TSS_Daily = var.discharge'). Probably the model failed to start? Check the log files of the run!")
	
	# Read the simulated streamflow data from the file
	simulated_streamflow = pd.read_csv(Qsim_tss, sep=r"\s+", index_col=0, skiprows=4, header=None, skipinitialspace=True)
	
	# Replace missing data with NaN
	simulated_streamflow[1][simulated_streamflow[1]==1e31] = np.nan

	# Generate a list of dates for the simulated streamflow data
	simulated_dates = [calibration_config['spinup_time']]
	for _ in range(len(simulated_streamflow) - 1):
		simulated_dates.append(simulated_dates[-1] + timedelta(days=1))
	
	# Set the index of the simulated streamflow data to the generated dates
	simulated_streamflow = simulated_streamflow[1]
	simulated_streamflow.index = [pd.Timestamp(date) for date in simulated_dates]
	simulated_streamflow.name = 'simulated'

	# Combine the simulated and observed streamflow data
	streamflows = pd.concat([simulated_streamflow, observed_streamflow], join='inner', axis=1)
	
	# Filter the streamflow data to the specified start and end times
	streamflows = streamflows[(streamflows.index > datetime.combine(calibration_config['start_time'], datetime.min.time())) & (streamflows.index < datetime.combine(calibration_config['end_time'], datetime.min.time()))]
	
	# Add a small value to the simulated streamflow to avoid division by zero
	streamflows['simulated'] += 0.0001

	# Check the specified objective function and calculate the score
	if OBJECTIVE == 'KGE':
		KGE = hydroStats.KGE(s=streamflows['simulated'],o=streamflows['observed'])
		print("run_id: " + str(individual.label)+", KGE: "+"{0:.3f}".format(KGE))
		with open(os.path.join(calibration_path,"runs_log.csv"), "a") as myfile:
			myfile.write(str(individual.label)+"," + str(KGE)+"\n")
		return KGE
	elif OBJECTIVE == 'COR':
		COR = hydroStats.correlation(s=streamflows['simulated'],o=streamflows['observed'])
		print("run_id: " + str(individual.label)+", COR "+"{0:.3f}".format(COR))
		with open(os.path.join(calibration_path,"runs_log.csv"), "a") as myfile:
			myfile.write(str(individual.label)+"," + str(COR)+"\n")
		return COR
	elif OBJECTIVE == 'NSE':
		NSE = hydroStats.NS(s=streamflows['simulated'], o=streamflows['observed'])
		print("run_id: " + str(individual.label) + ", NSE: " + "{0:.3f}".format(NSE))
		with open(os.path.join(calibration_path, "runs_log.csv"), "a") as myfile:
			myfile.write(str(individual.label) + "," + str(NSE) + "\n")
		return NSE
	else:
	raise ValueError


@handle_ctrl_c
def run_model(individual):
	"""
	This function takes an individual from the population and runs the model with the corresponding parameters.
	It first checks if the run directory already exists and whether the model was run before. 
	If the directory exists and the model was run before, it skips running the model. 
	Otherwise, it runs the model and saves the results to the run directory.
	"""
	# Define the directory where the model run will be stored
	run_directory = os.path.join(runs_path, individual.label)

	# Check if the run directory already exists
	if os.path.isdir(run_directory):
		# If the directory exists, check if the model was run before
		if os.path.exists(os.path.join(run_directory, dischargetss)):
			# If the model was run before, set runmodel to False
			runmodel = False
		else:
			# If the model was not run before, set runmodel to True and delete the directory
			runmodel = True
			shutil.rmtree(run_directory)
	else:
		# If the directory does not exist, set runmodel to True
		runmodel = True

	if runmodel:
		# Convert the individual's parameter ratios to the corresponding parameter values
		individual_parameter_ratio = individual.tolist()
		# Assert that all parameter ratios are between 0 and 1
		assert (np.array(individual_parameter_ratio) >= 0).all() and (np.array(individual_parameter_ratio) <= 1).all()
		calibration_parameters = calibration_config['parameters']
		
		# Create a dictionary of the individual's parameters
		individual_parameters = {}
		for i, parameter_data in enumerate(calibration_parameters.values()):
			individual_parameters[parameter_data['variable']] = \
				parameter_data['min'] + individual_parameter_ratio[i] * (parameter_data['max'] - parameter_data['min'])
		
		# Create the configuration file for the model run
		config_path = os.path.join(run_directory, 'config.yml')
		while True:
			os.mkdir(run_directory)
			with open(args.config, 'r') as f:
				template = yaml.load(f, Loader=yaml.FullLoader)

			# Update the template configuration file with the individual's parameters
			template['general']['spinup_time'] = calibration_config['spinup_time']
			template['general']['start_time'] = calibration_config['end_time']
			template['general']['export_inital_on_spinup'] = False
			template['report'] = {
				# "crops_per_district": {
				# 	"type": "farmers",
				# 	"function": "groupcount",
				# 	"varname": "crop
				# 	"save": "save",
				# 	"format": "csv",
				# 	"groupby": "tehsil"
				# },
				"well_irrigated_per_district": {
					"type": "farmers",
					"function": "mean",
					"varname": "has_well",
					"save": "save",
					"format": "csv",
					"groupby": "tehsil"
				},
			}
			template['report_cwatm'] = {}

			# loop through individual parameters and set them in the template
			for parameter, value in individual_parameters.items():
				multi_set(template, value, *parameter.split('.'))

			# set the report folder in the general section of the template
			template['general']['report_folder'] = run_directory

			# write the template to the specified config file
			with open(config_path, 'w') as f:
				yaml.dump(template, f)

			# acquire lock to check and set GPU usage
			lock.acquire()
			if current_gpu_use_count.value < n_gpus:
				use_gpu = current_gpu_use_count.value
				current_gpu_use_count.value += 1
				print(f'Using 1 GPU, current_counter: {current_gpu_use_count.value}/{n_gpus}')
			else:
				use_gpu = False
				print(f'Not using GPU, current_counter: {current_gpu_use_count.value}/{n_gpus}')
			lock.release()

			# build the command to run the script, including the use of a GPU if specified
			command = f"python run.py --config {config_path} --scenario spinup"
			if use_gpu is not False:
				command += f' --GPU --gpu_device {use_gpu}'
			print(command, flush=True)

			# run the command and capture the output and errors
			p = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
			output, errors = p.communicate()

			# release the GPU if it was used
			if use_gpu is not False:
				lock.acquire()
				current_gpu_use_count.value -= 1
				lock.release()
				print(f'Released 1 GPU, current_counter: {current_gpu_use_count.value}/{n_gpus}')

			# check the return code of the command and handle accordingly
			if p.returncode == 0:
				with open(os.path.join(logs_path, f"log{individual.label}.txt"), 'w') as f:
					content = "OUTPUT:\n" + str(output.decode())+"\nERRORS:\n" + str(errors.decode())
					f.write(content)
				modflow_folder = os.path.join(run_directory, 'spinup', 'modflow_model')
				if os.path.exists(modflow_folder):
					shutil.rmtree(modflow_folder)
				break
			elif p.returncode == 1:
				with open(os.path.join(logs_path, f"log{individual.label}_{''.join((random.choice(string.ascii_lowercase) for x in range(10)))}.txt"), 'w') as f:
					content = "OUTPUT:\n" + str(output.decode())+"\nERRORS:\n" + str(errors.decode())
					f.write(content)
				shutil.rmtree(run_directory)
			else:
				raise ValueError

	return (
		get_discharge_score(run_directory, individual),
		# get_irrigation_equipment_score(run_directory, individual),
	)

def export_front_history(calibration_values, effmax, effmin, effstd, effavg):
	# Save history of the change in objective function scores during calibration to csv file
	print(">> Saving optimization history (front_history.csv)")
	front_history = {}
	for i, calibration_value in enumerate(calibration_values):
		front_history.update({
			(calibration_value, 'effmax_R', ): effmax[:, i],
			(calibration_value, 'effmin_R'): effmin[:, i],
			(calibration_value, 'effstd_R'): effstd[:, i],
			(calibration_value, 'effavg_R'): effavg[:, i],
		})
	front_history = pd.DataFrame(
		front_history,
		index=list(range(ngen))
	)
	front_history.to_excel(os.path.join(calibration_path, "front_history.xlsx"))

def init_pool(manager_current_gpu_use_count, manager_lock, gpus):
	# set global variable for each process in the pool:
	global ctrl_c_entered
	global default_sigint_handler
	ctrl_c_entered = False
	default_sigint_handler = signal.signal(signal.SIGINT, pool_ctrl_c_handler)

	global lock
	global current_gpu_use_count
	global n_gpus
	n_gpus = gpus
	lock = manager_lock
	current_gpu_use_count = manager_current_gpu_use_count

if __name__ == "__main__":
    # Define the calibration values and weights for the fitness function
	calibration_values = ['KGE']
	weights = (1, )
	
    # Create the fitness class using DEAP's creator class
	creator.create("FitnessMulti", base.Fitness, weights=weights)
	
    # Create the individual class using DEAP's creator class
    # The individual class is an array of typecode 'd' with the FitnessMulti class as its fitness attribute
	creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMulti)

    # Create a toolbox object to register the functions used in the genetic algorithm
	toolbox = base.Toolbox()

	# Register the attribute generator function
    # This function generates a random float between 0 and 1
	toolbox.register("attr_float", random.uniform, 0, 1)
	
    # Register the selection method
    # This function selects the best individuals from a population
	toolbox.register("select", tools.selBest)
	
    # Register the population initializer
    # This function creates a population of individuals, each with len(calibration_config['parameters']) number of attributes
    # Each attribute is generated using the attr_float function
	toolbox.register("Individual", tools.initRepeat, creator.Individual, toolbox.attr_float, len(calibration_config['parameters']))
	toolbox.register("population", tools.initRepeat, list, toolbox.Individual)
    
    # Define a function to check if an individual's attribute values are within the bounds
	def checkBounds(min, max):
		def decorator(func):
			def wrappper(*args, **kargs):
				offspring = func(*args, **kargs)
                # Iterate through the offspring, and for each attribute, check if it is within bounds
                # If it is out of bounds, set it to the maximum or minimum value respectively
				for child in offspring:
					for i in range(len(child)):
						if child[i] > max:
							child[i] = max
						elif child[i] < min:
							child[i] = min
				return offspring
			return wrappper
		return decorator

    # Register the evaluation function
    # This function runs the model and returns the fitness value of an individual
	toolbox.register("evaluate", run_model)
    
    # Register the crossover function
    # This function mates two individuals using a blend crossover with an alpha value of 0.15
	toolbox.register("mate", tools.cxBlend, alpha=0.15)
    
    # Register the mutation function
    # This function mutates an individual using gaussian mutation with a mu of 0, sigma of 0.3, and indpb of 0.3
	toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.3, indpb=0.3)
    
    # Register the selection method
    # This function uses the NSGA-II algorithm to select individuals from a population
	toolbox.register("select", tools.selNSGA2)

    # Create a history object to keep track of the population statistics
	history = tools.History()

    # Decorate the crossover and mutation functions with the checkBounds function
    # This ensures that the attribute values of the offspring stay within the bounds
	toolbox.decorate("mate", checkBounds(0, 1))
	toolbox.decorate("mutate", checkBounds(0, 1))

    # Create a manager for multiprocessing
	manager = multiprocessing.Manager()
    # Create a shared variable to keep track of the number of GPUs in use
	current_gpu_use_count = manager.Value('i', 0)
    # Create a lock for managing access to the shared variable
	manager_lock = manager.Lock()

    # Check if multiprocessing should be used
	if use_multiprocessing is True:
        # Get the number of CPU cores available for the pool
		pool_size = int(os.getenv('SLURM_CPUS_PER_TASK') or 4)
		print(f'Pool size: {pool_size}')
        # Ignore the interrupt signal
		signal.signal(signal.SIGINT, signal.SIG_IGN)
        # Create a multiprocessing pool with the specified number of processes
        # Initialize the pool with the shared variable and lock, and the number of GPUs available
		pool = multiprocessing.Pool(processes=pool_size, initializer=init_pool, initargs=(current_gpu_use_count, manager_lock, calibration_config['gpus']))
        # Register the map function to use the multiprocessing pool
		toolbox.register("map", pool.map)
	else:
        # Initialize the pool without multiprocessing
		init_pool(current_gpu_use_count, manager_lock, calibration_config['gpus'])

    # Set the probabilities of mating and mutation
	cxpb = 0.7 # The probability of mating two individuals
	mutpb = 0.3 # The probability of mutating an individual
    # Ensure that the probabilities add up to 1
	assert cxpb + mutpb == 1, "cxpb + mutpb must be equal to 1"

    # Create arrays to hold statistics about the population
	effmax = np.full((ngen, len(calibration_values)), np.nan)
	effmin = np.full((ngen, len(calibration_values)), np.nan)
	effavg = np.full((ngen, len(calibration_values)), np.nan)
	effstd = np.full((ngen, len(calibration_values)), np.nan)

    # Define the checkpoint file path
	checkpoint = os.path.join(calibration_path, "checkpoint.pkl")
    # Check if the checkpoint file exists
	if os.path.exists(os.path.join(checkpoint)):
        # If the checkpoint file exists, load the population and other information from the checkpoint
		with open(checkpoint, "rb" ) as cp_file:
			cp = pickle.load(cp_file)
			population = cp["population"]
			start_gen = cp["generation"]
			random.setstate(cp["rndstate"])
			if start_gen > 0:
				offspring = cp["offspring"]
			pareto_front =  cp["pareto_front"]
	else:
        # If the checkpoint file does not exist, start from the first generation
		start_gen = 0
        # Create the initial population
		population = toolbox.population(n=mu)
        # Give each individual a label
		for i, individual in enumerate(population):
			individual.label = str(start_gen % 1000).zfill(2) + '_' + str(i % 1000).zfill(3)
		pareto_front = tools.ParetoFront()
		history.update(population)

    # Start the genetic algorithm loop
	for generation in range(start_gen, ngen):
        # If this is the first generation, save the initial population
		if generation == 0:
			cp = dict(population=population, generation=generation, rndstate=random.getstate(), pareto_front=pareto_front)
		else:
			# Vary the population using crossover and mutation
			offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)
			for i, child in enumerate(offspring):
				child.label = str(generation % 1000).zfill(2) + '_' + str(i % 1000).zfill(3)
			# Save the population and offspring
			cp = dict(population=population, generation=generation, rndstate=random.getstate(), offspring=offspring, pareto_front=pareto_front)

        # Save the checkpoint
		with open(checkpoint, "wb") as cp_file:
			pickle.dump(cp, cp_file)

		# Evaluate the individuals with an invalid fitness
		if generation == 0:
			individuals_to_evaluate = [ind for ind in population if not ind.fitness.valid]
		else:
			individuals_to_evaluate = [ind for ind in offspring if not ind.fitness.valid]
		fitnesses = list(toolbox.map(toolbox.evaluate, individuals_to_evaluate))
		if any(map(lambda x: isinstance(x, KeyboardInterrupt), fitnesses)):
			raise KeyboardInterrupt

		for ind, fit in zip(individuals_to_evaluate, fitnesses):
			ind.fitness.values = fit

		# Update the hall of fame with the generated individuals
		if generation == 0:
			pareto_front.update(population)
			population[:] = toolbox.select(population, lambda_)
		else:
			pareto_front.update(offspring)
			population[:] = toolbox.select(population + offspring, select_best_n_individuals)

		# Select the next generation population
		history.update(population)

		# Loop through the different objective functions and calculate some statistics
		# from the Pareto optimal population
		for ii in range(len(calibration_values)):
			effmax[generation, ii] = np.amax([pareto_front[x].fitness.values[ii] for x in range(len(pareto_front))])
			effmin[generation, ii] = np.amin([pareto_front[x].fitness.values[ii] for x in range(len(pareto_front))])
			effavg[generation, ii] = np.average([pareto_front[x].fitness.values[ii] for x in range(len(pareto_front))])
			effstd[generation, ii] = np.std([pareto_front[x].fitness.values[ii] for x in range(len(pareto_front))])
		
		print(">> gen: " + str(generation) + ", effmax_KGE: "+"{0:.3f}".format(effmax[generation, 0]))
		# print(">> gen: " + str(generation) + ", effmax_irrigation_equipment: "+"{0:.3f}".format(effmax[generation, 1]))

	# Closing the multiprocessing pool
	if use_multiprocessing is True:
		pool.close()
	
	global ctrl_c_entered
	global default_sigint_handler
	ctrl_c_entered = False
	default_sigint_handler = signal.signal(signal.SIGINT, pool_ctrl_c_handler)

	export_front_history(calibration_values, effmax, effmin, effstd, effavg)