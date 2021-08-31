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
from datetime import datetime, timedelta
import os
import shutil
import hydroStats
import array
import random
import numpy as np
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import pandas as pd
import yaml

import multiprocessing
import time
from subprocess import Popen, PIPE

from sys import platform
import pickle

## Set global parameter
global gen
gen = 0
WarmupDays = 0

with open('calibration/config.yml', 'r') as f:
	config = yaml.load(f, Loader=yaml.FullLoader)

ROOT = 'DataDrive/GEB/calibration'
OBJECTIVE = 'KGE'

LOG_FOLDER = os.path.join(ROOT, 'logs')
if not os.path.exists(LOG_FOLDER):
	os.makedirs(LOG_FOLDER)

if config['timeperiod'] == "monthly":
	monthly = 1
	dischargetss = os.path.join(config['scenario'], 'var.discharge_monthavg.tss')
	frequen = 'MS'
elif config['timeperiod'] == "daily":
	monthly = 0
	dischargetss = os.path.join(config['scenario'], 'var.discharge_daily.tss')
	frequen = 'd'
else:
	raise ValueError("timeperiod must be 'monthly' or 'daily'")

ParamRangesPath = os.path.join(ROOT, config['parameter_ranges'])
SubCatchmentPath = os.path.join(ROOT, config['subcatchmentpath'])
if not os.path.exists(SubCatchmentPath):
	os.makedirs(SubCatchmentPath)

Qtss_csv = os.path.join(ROOT, config['observed_data']['path'])
Qtss_col = config['observed_data']['column']

use_multiprocessing = config['DEAP']['use_multiprocessing']

try:
    pool_limit = config['DEAP']['pool_limit']
except:
    pool_limit = 10000

ngen = config['DEAP']['ngen']
mu = config['DEAP']['mu']
lambda_ = config['DEAP']['lambda_']
maximize =  config['DEAP']['maximize']
if maximize:
	maxDeap = 1.0
else:
	maxDeap = -1.0


define_first_run = config['options']['define_first_run']
if define_first_run:
	raise NotImplementedError
redo_best_run = config['options']['redo_best_run']

########################################################################
#   Preparation for calibration
########################################################################

# Load parameter range file
ParamRanges = pd.read_csv(ParamRangesPath, sep=",", index_col=0)
ParamRanges = ParamRanges[ParamRanges['Use'] == True].drop('Use', axis=1)

# Load observed streamflow
streamflow_data = pd.read_csv(Qtss_csv, sep=",", parse_dates=True, index_col=0)
observed_streamflow = streamflow_data[Qtss_col]
observed_streamflow.name = 'observed'
assert (observed_streamflow >= 0).all()

# first standard parameter set
if define_first_run:
	raise NotImplementedError

ii = 1

########################################################################
#   Function for running the model, returns objective function scores
########################################################################

def RunModel(Individual):

	# Convert scaled parameter values ranging from 0 to 1 to usncaled parameter values
	Parameters = [None] * len(ParamRanges)
	for ii in range(0,len(ParamRanges-1)):
		Parameters[ii] = Individual[ii]*(float(ParamRanges.iloc[ii,1])-float(ParamRanges.iloc[ii,0]))+float(ParamRanges.iloc[ii,0])

	# Note: The following code must be identical to the code near the end where the model is run
	# using the "best" parameter set. This code:
	# 1) Modifies the settings file containing the unscaled parameter values amongst other things
	# 2) Makes a .bat file to run the model
	# 3) Run the model and loads the simulated streamflow

	# Random number is appended to settings and .bat files to avoid simultaneous editing
	id =int(Individual[-1])
	run_id = str(id//1000).zfill(2) + "_" + str(id%1000).zfill(3)
	print('working on:', run_id)

	directory_run = os.path.join(SubCatchmentPath, run_id)

	if os.path.isdir(directory_run):
		if os.path.exists(os.path.join(directory_run, dischargetss)):
			runmodel = False
		else:
			runmodel = True
			shutil.rmtree(directory_run)
	else:
		runmodel = True

	config_path = os.path.join(directory_run, 'config.yml')
	if runmodel:
		os.mkdir(directory_run)

		with open('GEB.yml', 'r') as f:
			template = yaml.load(f, Loader=yaml.FullLoader)

		template['general']['start_time'] = config['start_date']
		template['general']['end_time'] = config['end_date']

		template['report'] = {}  # no other reporting than discharge required.
		template['report_cwatm'] = {}  # no other reporting than discharge required.

		for i, name in enumerate(ParamRanges.index):
			template['parameters'][name] = Parameters[i]

		template['general']['report_folder'] = directory_run

		with open(config_path, 'w') as f:
			yaml.dump(template, f)

		command = f"python run.py --config {config_path} --headless --scenario {config['scenario']}"

		p = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
		output, errors = p.communicate()
		with open(os.path.join(LOG_FOLDER, f"log{run_id}.txt"), 'w') as f:
			content = "OUTPUT:\n"+str(output.decode())+"\nERRORS:\n"+str(errors.decode())
			f.write(content)

		modflow_folder = os.path.join(directory_run, config['scenario'], 'modflow_model')
		if os.path.exists(modflow_folder):
			shutil.rmtree(modflow_folder)
	
	else:
		with open(config_path, 'r') as f:
			template = yaml.load(f, Loader=yaml.FullLoader)

	Qsim_tss = os.path.join(directory_run, dischargetss)
	
	if not os.path.isfile(Qsim_tss):
		print("run_id: "+str(run_id)+" File: "+ Qsim_tss)
		raise Exception("No simulated streamflow found. Is the data exported in the ini-file (e.g., 'OUT_TSS_Daily = var.discharge'). Probably the model failed to start? Check the log files of the run!")
	
	simulated_streamflow = pd.read_csv(Qsim_tss,sep=r"\s+", index_col=0, skiprows=4, header=None, skipinitialspace=True)
	simulated_streamflow[1][simulated_streamflow[1]==1e31] = np.nan

	simulated_dates = [template['general']['start_time']]
	for _ in range(len(simulated_streamflow) - 1):
		simulated_dates.append(simulated_dates[-1] + timedelta(days=1))
	simulated_streamflow = simulated_streamflow[1]
	simulated_streamflow.index = [pd.Timestamp(date) for date in simulated_dates]
	simulated_streamflow.name = 'simulated'

	streamflows = pd.concat([simulated_streamflow, observed_streamflow], join='inner', axis=1)
	streamflows['simulated'] += 0.0001


	if OBJECTIVE == 'KGE':
		# Compute objective function score
		KGE = hydroStats.KGE(s=streamflows['simulated'],o=streamflows['observed'],warmup=WarmupDays)
		print("run_id: "+str(run_id)+", KGE: "+"{0:.3f}".format(KGE))
		with open(os.path.join(SubCatchmentPath,"runs_log.csv"), "a") as myfile:
			myfile.write(str(run_id)+","+str(KGE)+"\n")
		return KGE, # If using just one objective function, put a comma at the end!!!

	elif OBJECTIVE == 'COR':

		COR = hydroStats.correlation(s=streamflows['simulated'],o=streamflows['observed'],warmup=WarmupDays)
		print("run_id: "+str(run_id)+", COR "+"{0:.3f}".format(COR))
		with open(os.path.join(SubCatchmentPath,"runs_log.csv"), "a") as myfile:
			myfile.write(str(run_id)+","+str(COR)+"\n")
		return COR, # If using just one objective function, put a comma at the end!!!

	elif OBJECTIVE == 'NSE':
		NSE = hydroStats.NS(s=streamflows['simulated'], o=streamflows['observed'], warmup=WarmupDays)
		print("run_id: " + str(run_id) + ", NSE: " + "{0:.3f}".format(NSE))
		with open(os.path.join(SubCatchmentPath, "runs_log.csv"), "a") as myfile:
			myfile.write(str(run_id) + "," + str(NSE) + "\n")
		return NSE,  # If using just one objective function, put a comma at the end!!!
	else:
		raise ValueError

########################################################################
#   Perform calibration using the DEAP module
########################################################################

creator.create("FitnessMin", base.Fitness, weights=(maxDeap,))
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("attr_float", random.uniform, 0, 1)

# Structure initializers
toolbox.register("Individual", tools.initRepeat, creator.Individual, toolbox.attr_float, len(ParamRanges))
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

toolbox.register("evaluate", RunModel)
toolbox.register("mate", tools.cxBlend, alpha=0.15)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.3, indpb=0.3)
toolbox.register("select", tools.selNSGA2)

toolbox.decorate("mate", checkBounds(0, 1))
toolbox.decorate("mutate", checkBounds(0, 1))

if __name__ == "__main__":

	t = time.time()

	if use_multiprocessing is True:
		pool_size = multiprocessing.cpu_count() * 1
		if pool_size > pool_limit: pool_size = pool_limit
		print(f'Pool size: {pool_size}')
		pool = multiprocessing.Pool(processes=pool_size)
		toolbox.register("map", pool.map)
	

	# For someone reason, if sum of cxpb and mutpb is not one, a lot less Pareto optimal solutions are produced
	cxpb = 0.9
	mutpb = 0.1

	startlater = False
	checkpoint = os.path.join(SubCatchmentPath, "checkpoint.pkl")
	if os.path.exists(os.path.join(checkpoint)):
		with open(checkpoint, "rb" ) as cp_file:
			cp = pickle.load(cp_file)
			population = cp["population"]
			start_gen = cp["generation"]
			random.setstate(cp["rndstate"])
			if start_gen > 0:
				offspring = cp["offspring"]
				halloffame =  cp["halloffame"]
				startlater = True
				gen = start_gen

	else:
		population = toolbox.population(n=mu)
		# Numbering of runs
		for ii in range(mu):
			population[ii][-1]= float(gen * 1000 + ii+1)

		#first run parameter set:
		if define_first_run:
			raise NotImplementedError

	effmax = np.zeros(shape=(ngen+1,1))*np.NaN
	effmin = np.zeros(shape=(ngen+1,1))*np.NaN
	effavg = np.zeros(shape=(ngen+1,1))*np.NaN
	effstd = np.zeros(shape=(ngen+1,1))*np.NaN
	if startlater == False:
		halloffame = tools.ParetoFront()

		# saving population
		cp = dict(population=population, generation=gen, rndstate=random.getstate())
		with open(checkpoint, "wb") as cp_file:
			pickle.dump(cp, cp_file)
		cp_file.close()


		# Evaluate the individuals with an invalid fitness
		invalid_ind = [ind for ind in population if not ind.fitness.valid]
		fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
		for ind, fit in zip(invalid_ind, fitnesses):
			ind.fitness.values = fit

		halloffame.update(population)

		# Loop through the different objective functions and calculate some statistics
		# from the Pareto optimal population

		for ii in range(1):
			effmax[0,ii] = np.amax([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
			effmin[0,ii] = np.amin([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
			effavg[0,ii] = np.average([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
			effstd[0,ii] = np.std([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
		gen = 0
		print(">> gen: "+str(gen)+", effmax_KGE: "+"{0:.3f}".format(effmax[gen,0]))
		gen = 1

	# Begin the generational process
	conditions = {"ngen" : False, "StallFit" : False}
	while not any(conditions.values()):
		if startlater == False:
			# Vary the population
			offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)

			# put in the number of run
			for ii in range(lambda_):
				offspring[ii][-1] = float(gen * 1000 + ii + 1)

		# saving population
		cp = dict(population=population, generation=gen, rndstate=random.getstate(), offspring=offspring, halloffame=halloffame)
		with open(checkpoint, "wb") as cp_file:
			pickle.dump(cp, cp_file)
		cp_file.close()
		startlater = False

		# Evaluate the individuals with an invalid fitness
		invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
		fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
		for ind, fit in zip(invalid_ind, fitnesses):
			ind.fitness.values = fit

		# Update the hall of fame with the generated individuals
		if halloffame is not None:
			halloffame.update(offspring)

		# Select the next generation population
		population[:] = toolbox.select(population + offspring, mu)

		# put in the number of run
		#for ii in xrange(mu):
		#	population[ii][-1] = float(gen * 1000 + ii + 1)

		# Loop through the different objective functions and calculate some statistics
		# from the Pareto optimal population
		for ii in range(1):
			effmax[gen,ii] = np.amax([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
			effmin[gen,ii] = np.amin([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
			effavg[gen,ii] = np.average([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
			effstd[gen,ii] = np.std([halloffame[x].fitness.values[ii] for x in range(len(halloffame))])
		print(">> gen: "+str(gen)+", effmax_KGE: "+"{0:.3f}".format(effmax[gen,0]))

		# Terminate the optimization after ngen generations
		if gen >= ngen:
			print(">> Termination criterion ngen fulfilled.")
			conditions["ngen"] = True

		gen += 1
		# Copied and modified from algorithms.py eaMuPlusLambda until here

	# Finito
	if use_multiprocessing is True:
		pool.close()
	elapsed = time.time() - t
	print(">> Time elapsed: "+"{0:.2f}".format(elapsed)+" s")

	########################################################################
	#   Save calibration results
	########################################################################

	# Save history of the change in objective function scores during calibration to csv file
	print(">> Saving optimization history (front_history.csv)")
	front_history = pd.DataFrame({'gen':list(range(gen)),
									  'effmax_R':effmax[:,0],
									  'effmin_R':effmin[:,0],
									  'effstd_R':effstd[:,0],
									  'effavg_R':effavg[:,0],
									  })
	front_history.to_csv(os.path.join(SubCatchmentPath,"front_history.csv"),',')
	# as numpy  numpy.asarray  ; numpy.savetxt("foo.csv", a, delimiter=","); a.tofile('foo.csv',sep=',',format='%10.5f')

	# Compute overall efficiency scores from the objective function scores for the
	# solutions in the Pareto optimal front
	# The overall efficiency reflects the proximity to R = 1, NSlog = 1, and B = 0 %
	front = np.array([ind.fitness.values for ind in halloffame])
	effover = 1 - np.sqrt((1-front[:,0]) ** 2)
	best = np.argmax(effover)

	# Convert the scaled parameter values of halloffame ranging from 0 to 1 to unscaled parameter values
	paramvals = np.zeros(shape=(len(halloffame),len(halloffame[0])))
	paramvals[:] = np.NaN
	for kk in range(len(halloffame)):
		for ii in range(len(ParamRanges)):
			paramvals[kk][ii] = halloffame[kk][ii]*(float(ParamRanges.iloc[ii,1])-float(ParamRanges.iloc[ii,0]))+float(ParamRanges.iloc[ii,0])

	# Save Pareto optimal solutions to csv file
	# The table is sorted by overall efficiency score
	print(">> Saving Pareto optimal solutions (pareto_front.csv)")
	ind = np.argsort(effover)[::-1]
	pareto_front = pd.DataFrame({'effover':effover[ind],'R':front[ind,0]})
	for ii in range(len(ParamRanges)):
		pareto_front["param_"+str(ii).zfill(2)+"_"+ParamRanges.index[ii]] = paramvals[ind,ii]
	pareto_front.to_csv(os.path.join(SubCatchmentPath,"pareto_front.csv"),',')

	# Select the "best" parameter set and run Model for the entire forcing period
	Parameters = paramvals[best,:]


	if redo_best_run:
		print(">> Running Model using the \"best\" parameter set")
		# Note: The following code must be identical to the code near the end where Model is run
		# using the "best" parameter set. This code:
		# 1) Modifies the settings file containing the unscaled parameter values amongst other things
		# 2) Makes a .bat file to run Model
		# 3) Runs Model and loads the simulated streamflow
		# Random number is appended to settings and .bat files to avoid simultaneous editing

		run_id = str(gen).zfill(2) + "_best"
		template_xml_new = template_xml
		directory_run = os.path.join(SubCatchmentPath, run_id)
		template_xml_new = template_xml_new.replace("%root", root)
		for ii in range(0,len(ParamRanges)):
			template_xml_new = template_xml_new.replace("%"+ParamRanges.index[ii],str(Parameters[ii]))
		template_xml_new = template_xml_new.replace('%run_id', directory_run)

		os.mkdir(directory_run)

		#template_xml_new = template_xml_new.replace('%InitModel',"1")
		f = open(os.path.join(directory_run,ModelSettings_template[:-4]+'-Run'+run_id+'.ini'), "w")
		f.write(template_xml_new)
		f.close()
		template_bat_new = template_bat
		template_bat_new = template_bat_new.replace('%run',ModelSettings_template[:-4]+'-Run'+run_id+'.ini')

		runfile = os.path.join(directory_run, RunModel_template[:-4] + run_id)
		if platform == "win32":
			runfile = runfile + ".bat"
		else:
			runfile = runfile + ".sh"
		f = open(runfile, "w")
		f.write(template_bat_new)
		f.close()

		currentdir = os.getcwd()
		os.chdir(directory_run)

		p = Popen(runfile, shell=True, stdout=PIPE, stderr=PIPE, bufsize=16*1024*1024)
		output, errors = p.communicate()
		f = open("log"+run_id+".txt",'w')
		content = "OUTPUT:\n"+str(output)+"\nERRORS:\n"+str(errors)
		f.write(content)
		f.close()
		os.chdir(currentdir)

		Qsim_tss = os.path.join(directory_run,dischargetss)
		
		simulated_streamflow = pd.read_table(Qsim_tss,sep=r"\s+",index_col=0,skiprows=4,header=None,skipinitialspace=True)
		simulated_streamflow[1][simulated_streamflow[1]==1e31] = np.nan
		Qsim = simulated_streamflow[1].values

		# Save simulated streamflow to disk
		print(">> Saving \"best\" simulated streamflow (streamflow_simulated_best.csv)")
		Qsim = pd.DataFrame(data=Qsim, index=pd.date_range(ForcingStart, periods=len(Qsim), freq=frequen))
		Qsim.to_csv(os.path.join(SubCatchmentPath,"streamflow_simulated_best.csv"),',',header="")
		try: os.remove(os.path.join(SubCatchmentPath,"out",'streamflow_simulated_best.tss'))
		except: pass
		#os.rename(Qsim_tss, os.path.join(SubCatchmentPath,"out",'streamflow_simulated_best.tss'))

	"""
	# Delete all .xml, .bat, .tmp, and .txt files created for the runs
	for filename in glob.glob(os.path.join(SubCatchmentPath,"*.xml")):
		os.remove(filename)
	for filename in glob.glob(os.path.join(SubCatchmentPath,"*.bat")):
		os.remove(filename)
	for filename in glob.glob(os.path.join(SubCatchmentPath,"*.tmp")):
		os.remove(filename)
	for filename in glob.glob(os.path.join(SubCatchmentPath,"*.txt")):
		os.remove(filename)
	"""
