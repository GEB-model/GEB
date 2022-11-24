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
from datetime import datetime, timedelta, date
import os
import shutil
import hydroStats
import array
import random
import string
import numpy as np
import signal
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import pandas as pd
import yaml
from functools import wraps

import multiprocessing
import time
from subprocess import Popen, PIPE

import pickle
from calconfig import config, args

config = config['calibration']

OBJECTIVE = 'KGE'

dischargetss = os.path.join('spinup', 'var.discharge_daily.tss')

calibration_path = config['path']
os.makedirs(calibration_path, exist_ok=True)
runs_path = os.path.join(calibration_path, 'runs')
os.makedirs(runs_path, exist_ok=True)
logs_path = os.path.join(calibration_path, 'logs')
os.makedirs(logs_path, exist_ok=True)

Qtss_csv = config['observations']['discharge']['path']
Qtss_col = config['observations']['discharge']['column']

use_multiprocessing = config['DEAP']['use_multiprocessing']

select_best = config['DEAP']['select_best']

ngen = config['DEAP']['ngen']
mu = config['DEAP']['mu']
lambda_ = config['DEAP']['lambda_']

# Load observed streamflow
streamflow_data = pd.read_csv(Qtss_csv, sep=",", parse_dates=True, index_col=0)
observed_streamflow = streamflow_data[Qtss_col]
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
                return KeyboardInterrupt()
            finally:
                signal.signal(signal.SIGINT, pool_ctrl_c_handler)
        else:
            return KeyboardInterrupt()
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
	Qsim_tss = os.path.join(run_directory, dischargetss)
	if not os.path.isfile(Qsim_tss):
		print("run_id: "+str(individual.label)+" File: "+ Qsim_tss)
		raise Exception("No simulated streamflow found. Is the data exported in the ini-file (e.g., 'OUT_TSS_Daily = var.discharge'). Probably the model failed to start? Check the log files of the run!")
	
	simulated_streamflow = pd.read_csv(Qsim_tss, sep=r"\s+", index_col=0, skiprows=4, header=None, skipinitialspace=True)
	simulated_streamflow[1][simulated_streamflow[1]==1e31] = np.nan

	simulated_dates = [config['spinup_start']]
	for _ in range(len(simulated_streamflow) - 1):
		simulated_dates.append(simulated_dates[-1] + timedelta(days=1))
	simulated_streamflow = simulated_streamflow[1]
	simulated_streamflow.index = [pd.Timestamp(date) for date in simulated_dates]
	simulated_streamflow.name = 'simulated'

	streamflows = pd.concat([simulated_streamflow, observed_streamflow], join='inner', axis=1)
	streamflows = streamflows[(streamflows.index > datetime.combine(config['start_date'], datetime.min.time())) & (streamflows.index < datetime.combine(config['end_date'], datetime.min.time()))]
	streamflows['simulated'] += 0.0001

	if config['monthly']:
		streamflows['date'] = streamflows.index
		streamflows = streamflows.resample('M', on='date').mean()

	if OBJECTIVE == 'KGE':
		# Compute objective function score
		KGE = hydroStats.KGE(s=streamflows['simulated'],o=streamflows['observed'])
		print("run_id: "+str(individual.label)+", KGE: "+"{0:.3f}".format(KGE))
		with open(os.path.join(calibration_path,"runs_log.csv"), "a") as myfile:
			myfile.write(str(individual.label)+","+str(KGE)+"\n")
		return KGE
	elif OBJECTIVE == 'COR':
		COR = hydroStats.correlation(s=streamflows['simulated'],o=streamflows['observed'])
		print("run_id: "+str(individual.label)+", COR "+"{0:.3f}".format(COR))
		with open(os.path.join(calibration_path,"runs_log.csv"), "a") as myfile:
			myfile.write(str(individual.label)+","+str(COR)+"\n")
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
	run_directory = os.path.join(runs_path, individual.label)

	if os.path.isdir(run_directory):
		if os.path.exists(os.path.join(run_directory, dischargetss)):
			runmodel = False
		else:
			runmodel = True
			shutil.rmtree(run_directory)
	else:
		runmodel = True

	if runmodel:
		individual_parameter_ratio = individual.tolist()
		assert (np.array(individual_parameter_ratio) >= 0).all() and (np.array(individual_parameter_ratio) <= 1).all()
		calibration_parameters = config['parameters']
		
		individual_parameters = {}
		for i, parameter_data in enumerate(calibration_parameters.values()):
			individual_parameters[parameter_data['variable']] = \
				parameter_data['min'] + individual_parameter_ratio[i] * (parameter_data['max'] - parameter_data['min'])
		
		config_path = os.path.join(run_directory, 'config.yml')
		while True:
			os.mkdir(run_directory)
			with open(args.config, 'r') as f:
				template = yaml.load(f, Loader=yaml.FullLoader)

			template['general']['spinup_start'] = config['spinup_start']
			template['general']['start_time'] = config['end_date']
			template['general']['export_inital_on_spinup'] = False
			template['report'] = {
				# "crops_per_district": {
				# 	"type": "farmers",
				# 	"function": "groupcount",
				# 	"varname": "crop",
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

			for parameter, value in individual_parameters.items():
				multi_set(template, value, *parameter.split('.'))

			template['general']['report_folder'] = run_directory

			with open(config_path, 'w') as f:
				yaml.dump(template, f)
			
			lock.acquire()
			if current_gpu_use_count.value < n_gpus:
				use_gpu = current_gpu_use_count.value
				current_gpu_use_count.value += 1
				print(f'Using 1 GPU, current_counter: {current_gpu_use_count.value}/{n_gpus}')
			else:
				use_gpu = False
				print(f'Not using GPU, current_counter: {current_gpu_use_count.value}/{n_gpus}')
			lock.release()
			
			command = f"python run.py --config {config_path} --headless --scenario spinup"
			if use_gpu is not False:
				command += f' --GPU --gpu_device {use_gpu}'
			print(command)

			p = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
			output, errors = p.communicate()

			if use_gpu is not False:
				lock.acquire()
				current_gpu_use_count.value -= 1
				lock.release()
				print(f'Released 1 GPU, current_counter: {current_gpu_use_count.value}/{n_gpus}')
			if p.returncode == 0:
				with open(os.path.join(logs_path, f"log{individual.label}.txt"), 'w') as f:
					content = "OUTPUT:\n"+str(output.decode())+"\nERRORS:\n"+str(errors.decode())
					f.write(content)
				modflow_folder = os.path.join(run_directory, 'spinup', 'modflow_model')
				if os.path.exists(modflow_folder):
					shutil.rmtree(modflow_folder)
				break
			elif p.returncode == 1:
				with open(os.path.join(logs_path, f"log{individual.label}_{''.join((random.choice(string.ascii_lowercase) for x in range(10)))}.txt"), 'w') as f:
					content = "OUTPUT:\n"+str(output.decode())+"\nERRORS:\n"+str(errors.decode())
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

if __name__ == "__main__":
	calibration_values = ['KGE']
	weights = (1, )
	
	creator.create("FitnessMulti", base.Fitness, weights=weights)
	creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMulti)

	toolbox = base.Toolbox()

	# Attribute generator
	toolbox.register("attr_float", random.uniform, 0, 1)
	toolbox.register("select", tools.selBest)
	# Structure initializers
	toolbox.register("Individual", tools.initRepeat, creator.Individual, toolbox.attr_float, len(config['parameters']))
	toolbox.register("population", tools.initRepeat, list, toolbox.Individual)

	def checkBounds(min, max):
		def decorator(func):
			def wrappper(*args, **kargs):
				offspring = func(*args, **kargs)
				for child in offspring:
					for i in range(len(child)):
						if child[i] > max:
							child[i] = max
						elif child[i] < min:
							child[i] = min
				return offspring
			return wrappper
		return decorator

	toolbox.register("evaluate", run_model)
	toolbox.register("mate", tools.cxBlend, alpha=0.15)
	toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.3, indpb=0.3)
	toolbox.register("select", tools.selNSGA2)

	history = tools.History()

	toolbox.decorate("mate", checkBounds(0, 1))
	toolbox.decorate("mutate", checkBounds(0, 1))

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

	manager = multiprocessing.Manager()
	current_gpu_use_count = manager.Value('i', 0)
	manager_lock = manager.Lock()

	if use_multiprocessing is True:
		pool_size = int(os.getenv('SLURM_CPUS_PER_TASK') or 4)
		print(f'Pool size: {pool_size}')
		signal.signal(signal.SIGINT, signal.SIG_IGN)
		pool = multiprocessing.Pool(processes=pool_size, initializer=init_pool, initargs=(current_gpu_use_count, manager_lock, config['gpus']))
		toolbox.register("map", pool.map)
	else:
		init_pool(current_gpu_use_count, manager_lock, config['gpus'])
	
	cxpb = 0.7 # The probability of mating two individuals
	mutpb = 0.3 # The probability of mutating an individual
	assert cxpb + mutpb == 1, "cxpb + mutpb must be equal to 1"

	effmax = np.full((ngen, len(calibration_values)), np.nan)
	effmin = np.full((ngen, len(calibration_values)), np.nan)
	effavg = np.full((ngen, len(calibration_values)), np.nan)
	effstd = np.full((ngen, len(calibration_values)), np.nan)

	checkpoint = os.path.join(calibration_path, "checkpoint.pkl")
	if os.path.exists(os.path.join(checkpoint)):
		with open(checkpoint, "rb" ) as cp_file:
			cp = pickle.load(cp_file)
			population = cp["population"]
			start_gen = cp["generation"]
			random.setstate(cp["rndstate"])
			if start_gen > 0:
				offspring = cp["offspring"]
			pareto_front =  cp["pareto_front"]
	else:
		start_gen = 0
		population = toolbox.population(n=mu)
		for i, individual in enumerate(population):
			individual.label = str(start_gen % 1000).zfill(2) + '_' + str(i % 1000).zfill(3)
		pareto_front = tools.ParetoFront()
		history.update(population)

	for generation in range(start_gen, ngen):
		if generation == 0:
			cp = dict(population=population, generation=generation, rndstate=random.getstate(), pareto_front=pareto_front)
		else:
			# Vary the population
			offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)
			for i, child in enumerate(offspring):
				child.label = str(generation % 1000).zfill(2) + '_' + str(i % 1000).zfill(3)
			cp = dict(population=population, generation=generation, rndstate=random.getstate(), offspring=offspring, pareto_front=pareto_front)

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
			population[:] = toolbox.select(population + offspring, select_best)

		# Select the next generation population
		history.update(population)

		# Loop through the different objective functions and calculate some statistics
		# from the Pareto optimal population
		for ii in range(len(calibration_values)):
			effmax[generation, ii] = np.amax([pareto_front[x].fitness.values[ii] for x in range(len(pareto_front))])
			effmin[generation, ii] = np.amin([pareto_front[x].fitness.values[ii] for x in range(len(pareto_front))])
			effavg[generation, ii] = np.average([pareto_front[x].fitness.values[ii] for x in range(len(pareto_front))])
			effstd[generation, ii] = np.std([pareto_front[x].fitness.values[ii] for x in range(len(pareto_front))])
		
		print(">> gen: "+str(generation) + ", effmax_KGE: "+"{0:.3f}".format(effmax[generation, 0]))
		# print(">> gen: "+str(generation) + ", effmax_irrigation_equipment: "+"{0:.3f}".format(effmax[generation, 1]))

	# Finito
	if use_multiprocessing is True:
		pool.close()
	
	global ctrl_c_entered
	global default_sigint_handler
	ctrl_c_entered = False
	default_sigint_handler = signal.signal(signal.SIGINT, pool_ctrl_c_handler)

	export_front_history(calibration_values, effmax, effmin, effstd, effavg)

	# Compute overall efficiency scores from the objective function scores for the
	# solutions in the Pareto optimal front
	# The overall efficiency reflects the proximity to R = 1, NSlog = 1, and B = 0 %
	# best = toolbox.select(pareto_front, 1)[0]

	# Convert the scaled parameter values of pareto_front ranging from 0 to 1 to unscaled parameter values
	
	# pareto_front_df = pd.DataFrame(columns=calibration_values + ParamRanges.index.tolist(), index=range(len(pareto_front)))
	# for i, individual in enumerate(pareto_front):
	# 	for j, calibration_value in enumerate(calibration_values):
	# 		pareto_front_df.iloc[i][calibration_value] = individual.fitness.values[j]
	# 	for j, (variable_name, values) in enumerate(ParamRanges.iterrows()):
	# 		pareto_front_df.iloc[i][variable_name] = individual[j] * (values['MaxValue'] - values['MinValue']) + values['MinValue']

	# pareto_front_df.to_excel(os.path.join(calibration_path, "pareto_front.xlsx"))