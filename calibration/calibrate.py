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
import json
import shutil
import hydroStats
import array
import random
import string
import numpy as np
import signal
import pandas as pd
from datetime import timedelta
import yaml
import geopandas as gpd
from deap import creator, base, tools, algorithms
from functools import wraps

import multiprocessing
from subprocess import Popen, PIPE

import pickle
from calconfig import config, args

calibration_config = config['calibration']

CALIBRATION_VALUES = {
	'KGE': 1,
	'irrigation_wells': 1
}

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
SCENARIO = calibration_config['scenario']
dischargetss = os.path.join(SCENARIO, 'var.discharge_daily.tss')

target_variables = calibration_config['target_variables']

# Load observed streamflow
gauges = [tuple(gauge) for gauge in config['general']['gauges']]
observed_streamflow = {}
for gauge in gauges:
	streamflow_path = os.path.join(config['general']['original_data'], 'calibration', 'streamflow', f"{gauge[0]} {gauge[1]}.csv")
	streamflow_data = pd.read_csv(streamflow_path, sep=",", parse_dates=True, index_col=0)
	observed_streamflow[gauge] = streamflow_data["flow"]
	# drop all rows with NaN values
	observed_streamflow[gauge] = observed_streamflow[gauge].dropna()
	observed_streamflow[gauge].name = 'observed'
	assert (observed_streamflow[gauge] >= 0).all()

with open(os.path.join(config['general']['input_folder'], 'agents', 'attributes', 'irrigation_sources.json')) as f:
	irrigation_source_key = json.load(f)

def get_observed_well_ratio():
	observed_irrigation_sources = gpd.read_file(os.path.join(config['general']['original_data'], 'census', 'output', 'irrigation_source_2010-2011.geojson')).to_crs(3857)
	simulated_subdistricts = gpd.read_file(os.path.join(config['general']['input_folder'], 'areamaps', 'subdistricts.geojson'))
	# set index to unique ID combination of state, district and subdistrict
	observed_irrigation_sources.set_index(['state_code', 'district_c', 'sub_distri'], inplace=True)
	simulated_subdistricts.set_index(['state_code', 'district_c', 'sub_distri'], inplace=True)
	# select rows from observed_irrigation_sources where the index is in simulated_subdistricts
	observed_irrigation_sources = observed_irrigation_sources.loc[simulated_subdistricts.index]

	region_mask = gpd.read_file(os.path.join(config['general']['input_folder'], 'areamaps', 'mask.geojson')).to_crs(3857)
	assert len(region_mask) == 1
	# get overlapping areas of observed_irrigation_sources and region_mask
	observed_irrigation_sources['area_in_region_mask'] = (gpd.overlay(observed_irrigation_sources, region_mask, how='intersection').area / observed_irrigation_sources.area.values).values

	ANALYSIS_THRESHOLD = 0.5

	observed_irrigation_sources = observed_irrigation_sources[observed_irrigation_sources['area_in_region_mask'] > ANALYSIS_THRESHOLD]
	observed_irrigation_sources = observed_irrigation_sources.join(simulated_subdistricts['ID'])
	observed_irrigation_sources.set_index('ID', inplace=True)

	total_holdings_observed = observed_irrigation_sources[[c for c in observed_irrigation_sources.columns if c.endswith('total_holdings')]].sum(axis=1)
	total_holdings_with_well_observed = (
		observed_irrigation_sources[[c for c in observed_irrigation_sources.columns if c.endswith('well_holdings')]].sum(axis=1) +
		observed_irrigation_sources[[c for c in observed_irrigation_sources.columns if c.endswith('tubewell_holdings')]].sum(axis=1)
	)
	ratio_holdings_with_well_observed = total_holdings_with_well_observed / total_holdings_observed
	return ratio_holdings_with_well_observed

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

def get_irrigation_wells_score(run_directory, individual):
	tehsils = np.load(os.path.join(run_directory, SCENARIO, 'tehsil', '20110101.npy'))
	field_size = np.load(os.path.join(run_directory, SCENARIO, 'field_size', '20110101.npy'))
	irrigation_source = np.load(os.path.join(run_directory, SCENARIO, 'irrigation_source', '20110101.npy'))
	well_irrigated = np.isin(irrigation_source, [irrigation_source_key['well'], irrigation_source_key['tubewell']])
	# Calculate the ratio of farmers with a well per tehsil
	farmers_per_tehsil = np.bincount(tehsils)
	well_irrigated_per_tehsil = np.bincount(tehsils, weights=well_irrigated)
	ratio_well_irrigated = well_irrigated_per_tehsil / farmers_per_tehsil

	ratio_holdings_with_well_observed = get_observed_well_ratio()

	ratio_holdings_with_well_simulated = ratio_well_irrigated[ratio_holdings_with_well_observed.index]
	ratio_holdings_with_well_observed = ratio_holdings_with_well_observed.values

	irrigation_well_score = 1 - abs(((ratio_holdings_with_well_simulated - ratio_holdings_with_well_observed) / ratio_holdings_with_well_observed))
	irrigation_well_score = float(np.mean(irrigation_well_score))
	print("run_id: " + str(individual.label)+", IWS: "+"{0:.3f}".format(irrigation_well_score))
	with open(os.path.join(calibration_path,"IWS_log.csv"), "a") as myfile:
		myfile.write(str(individual.label)+"," + str(irrigation_well_score)+"\n")

	return irrigation_well_score

def get_KGE_score(run_directory, individual):
    # Get the path of the simulated streamflow file
	Qsim_tss = os.path.join(run_directory, dischargetss)
	
	# Check if the simulated streamflow file exists
	if not os.path.isfile(Qsim_tss):
		print("run_id: " + str(individual.label)+" File: "+ Qsim_tss)
		raise Exception("No simulated streamflow found. Is the data exported in the ini-file (e.g., 'OUT_TSS_Daily = var.discharge'). Probably the model failed to start? Check the log files of the run!")
	
	# Read the simulated streamflow data from the file
	simulated_streamflow = pd.read_csv(Qsim_tss, index_col=0, skiprows=8, delim_whitespace=True, names=gauges)
	# parse the dates in the index
	simulated_streamflow.index = pd.date_range(calibration_config['start_time'] + timedelta(days=1), calibration_config['end_time'])

	def get_streamflows(simulated_streamflow, observed_streamflow, gauge):
		simulated_streamflow_gauge = simulated_streamflow[gauge]
		simulated_streamflow_gauge.name = 'simulated'
		observed_streamflow_gauge = observed_streamflow[gauge]
		observed_streamflow_gauge.name = 'observed'

		# Combine the simulated and observed streamflow data
		streamflows = pd.concat([simulated_streamflow_gauge, observed_streamflow_gauge], join='inner', axis=1)
		
		# Add a small value to the simulated streamflow to avoid division by zero
		streamflows['simulated'] += 0.0001
		return streamflows
	
	streamflows = [get_streamflows(simulated_streamflow, observed_streamflow, gauge) for gauge in gauges]
	streamflows = [streamflow for streamflow in streamflows if not streamflow.empty]
	if config['calibration']['monthly'] is True:
		# Calculate the monthly mean of the streamflow data
		streamflows = [streamflows.resample('M').mean() for streamflows in streamflows]

	KGEs = [hydroStats.KGE(s=streamflows['simulated'], o=streamflows['observed']) for streamflows in streamflows]
	assert KGEs  # Check if KGEs is not empty
	KGE = np.mean(KGEs)
	
	print("run_id: " + str(individual.label)+", KGE: "+"{0:.3f}".format(KGE))
	with open(os.path.join(calibration_path,"KGE_log.csv"), "a") as myfile:
		myfile.write(str(individual.label)+"," + str(KGE)+"\n")

	return KGE


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
		if os.path.exists(os.path.join(run_directory, 'done.txt')):
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
			template['general']['start_time'] = calibration_config['start_time']
			template['general']['end_time'] = calibration_config['end_time']

			del template['report']
			del template['report_cwatm']
			
			template.update(target_variables)

			# loop through individual parameters and set them in the template
			for parameter, value in individual_parameters.items():
				multi_set(template, value, *parameter.split('.'))

			# set the report folder in the general section of the template
			template['general']['report_folder'] = run_directory
			template['general']['initial_conditions_folder'] = os.path.join(run_directory, 'initial_conditions')
			template['general']['export_inital_on_spinup'] = True

			# write the template to the specified config file
			with open(config_path, 'w') as f:
				yaml.dump(template, f)

			# acquire lock to check and set GPU usage
			lock.acquire()
			if current_gpu_use_count.value < n_gpu_spots:
				use_gpu = int(current_gpu_use_count.value / calibration_config['DEAP']['models_per_gpu'])
				current_gpu_use_count.value += 1
				print(f'Using 1 GPU, current_counter: {current_gpu_use_count.value}/{n_gpu_spots}')
			else:
				use_gpu = False
				print(f'Not using GPU, current_counter: {current_gpu_use_count.value}/{n_gpu_spots}')
			lock.release()

			def run_model(scenario):
				# build the command to run the script, including the use of a GPU if specified
				command = f"python run.py --config {config_path} --scenario {scenario}"
				if use_gpu is not False:
					command += f' --GPU --gpu_device {use_gpu}'
				print(command, flush=True)

				# run the command and capture the output and errors
				p = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
				output, errors = p.communicate()

				# check the return code of the command and handle accordingly
				if p.returncode == 0:  # model has run successfully
					with open(os.path.join(logs_path, f"log{individual.label}_{scenario}.txt"), 'w') as f:
						content = "OUTPUT:\n" + str(output.decode())+"\nERRORS:\n" + str(errors.decode())
						f.write(content)
					modflow_folder = os.path.join(run_directory, 'spinup', 'modflow_model')
					if os.path.exists(modflow_folder):
						shutil.rmtree(modflow_folder)
				
				elif p.returncode == 1:  # model has failed
					with open(os.path.join(logs_path, f"log{individual.label}_{scenario}_{''.join((random.choice(string.ascii_lowercase) for x in range(10)))}.txt"), 'w') as f:
						content = "OUTPUT:\n" + str(output.decode())+"\nERRORS:\n" + str(errors.decode())
						f.write(content)
					shutil.rmtree(run_directory)
				
				else:
					raise ValueError("Return code of run.py was not 0 or 1, but instead " + str(p.returncode) + ".")
				
				return p.returncode
				
			return_code = run_model('spinup')
			if return_code == 0:
				return_code = run_model(SCENARIO)
				if return_code == 0:
					# release the GPU if it was used
					if use_gpu is not False:
						lock.acquire()
						current_gpu_use_count.value -= 1
						lock.release()
						print(f'Released 1 GPU, current_counter: {current_gpu_use_count.value}/{n_gpu_spots}')
					with open(os.path.join(run_directory, 'done.txt'), 'w') as f:
						f.write('done')
					break

			# release the GPU if it was used
			if use_gpu is not False:
				lock.acquire()
				current_gpu_use_count.value -= 1
				lock.release()
				print(f'Released 1 GPU, current_counter: {current_gpu_use_count.value}/{n_gpu_spots}')

	scores = []
	for score in CALIBRATION_VALUES:
		if score == 'KGE':
			scores.append(get_KGE_score(run_directory, individual))
		if score == 'irrigation_wells':
			scores.append(get_irrigation_wells_score(run_directory, individual))
	return tuple(scores)

def export_front_history(CALIBRATION_VALUES, effmax, effmin, effstd, effavg):
	# Save history of the change in objective function scores during calibration to csv file
	print(">> Saving optimization history (front_history.csv)")
	front_history = {}
	for i, calibration_value in enumerate(CALIBRATION_VALUES):
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

if __name__ == "__main__":
    # Create the fitness class using DEAP's creator class
	creator.create("FitnessMulti", base.Fitness, weights=tuple(CALIBRATION_VALUES.values()))
	
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
		pool = multiprocessing.Pool(processes=pool_size, initializer=init_pool, initargs=(
			current_gpu_use_count, manager_lock, calibration_config['DEAP']['gpus'], calibration_config['DEAP']['models_per_gpu'])
		)
        # Register the map function to use the multiprocessing pool
		toolbox.register("map", pool.map)
	else:
        # Initialize the pool without multiprocessing
		init_pool(
			current_gpu_use_count,
			manager_lock, calibration_config['DEAP']['gpus'], calibration_config['DEAP']['models_per_gpu'])

    # Set the probabilities of mating and mutation
	cxpb = 0.7 # The probability of mating two individuals
	mutpb = 0.3 # The probability of mutating an individual
    # Ensure that the probabilities add up to 1
	assert cxpb + mutpb == 1, "cxpb + mutpb must be equal to 1"

    # Create arrays to hold statistics about the population
	effmax = np.full((ngen, len(CALIBRATION_VALUES)), np.nan)
	effmin = np.full((ngen, len(CALIBRATION_VALUES)), np.nan)
	effavg = np.full((ngen, len(CALIBRATION_VALUES)), np.nan)
	effstd = np.full((ngen, len(CALIBRATION_VALUES)), np.nan)

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
		for ii in range(len(CALIBRATION_VALUES)):
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

	export_front_history(CALIBRATION_VALUES, effmax, effmin, effstd, effavg)