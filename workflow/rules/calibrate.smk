"""Calibration-specific rules for GEB evolutionary algorithm optimization.

This module contains rules specific to calibration workflows using DEAP:
- Population generation and management
- Fitness aggregation and selection
- Offspring creation
- Pareto front computation
"""

import random
import yaml
import numpy as np
from deap import base, creator, tools, algorithms
import array

# Calibration-specific configuration
NGEN = config.get("NGEN", 5)  # Number of generations
MU = config.get("MU", 10)  # Population size
LAMBDA = config.get("LAMBDA", 20)  # Offspring size per generation

# Parameter configuration (will be read from config later, but for now use defaults)
# These define the parameter space to explore
PARAMETERS = config.get("PARAMETERS", {
    "param1": {"min": 0.0, "max": 1.0},
    "param2": {"min": 0.0, "max": 1.0},
})
N_PARAMETERS = len(PARAMETERS)

# Initialize DEAP components
creator.create("FitnessMulti", base.Fitness, weights=(1.0,))  # Single objective for now
creator.create("Individual", array.array, typecode="d", fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0, 1)
toolbox.register(
    "individual",
    tools.initRepeat,
    creator.Individual,
    toolbox.attr_float,
    N_PARAMETERS,
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxBlend, alpha=0.15)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.3, indpb=0.3)
toolbox.register("select", tools.selNSGA2)

# Set random seed for reproducibility
random.seed(config.get("SEED", 42))

# Generate initial population (generation 0)
def generate_initial_population():
    """Generate the initial random population."""
    population = toolbox.population(n=MU)
    individuals = []
    for i, ind in enumerate(population):
        label = f"0_{i:03d}"  # Single digit for generation to match wildcard pattern
        individuals.append({
            "label": label,
            "generation": 0,
            "individual_id": f"{i:03d}",
            "values": [float(x) for x in ind]
        })
    return individuals

INITIAL_POPULATION = generate_initial_population()

# Generate parameters for generation 0 (initial population)
rule generate_initial_parameters:
    input:
        f"{BASE_DIR}/base_build.done" if BASE_DIR else "base_build.done"
    output:
        f"{BASE_DIR}/generation_0_population.yml" if BASE_DIR else "generation_0_population.yml"
    run:
        # Save initial population
        import os
        if BASE_DIR:
            os.makedirs(BASE_DIR, exist_ok=True)
        with open(output[0], "w") as f:
            yaml.dump({"individuals": INITIAL_POPULATION}, f, default_flow_style=False)

# Generate parameter file for a specific individual
rule generate_individual_parameters:
    input:
        pop_file=lambda wildcards: f"{BASE_DIR}/generation_{wildcards.gen}_population.yml" if BASE_DIR else f"generation_{wildcards.gen}_population.yml"
    output:
        params=f"{RUNS_DIR}/{{gen}}_{{ind}}/parameters.yml"
    run:
        import os
        
        # Load the population file
        with open(input.pop_file, "r") as f:
            pop_data = yaml.safe_load(f)
        
        # Find the individual
        label = f"{wildcards.gen}_{wildcards.ind}"
        individual_data = None
        for ind in pop_data["individuals"]:
            if ind["label"] == label:
                individual_data = ind
                break
        
        if individual_data is None:
            raise ValueError(f"Individual {label} not found in population file")
        
        # Convert normalized parameters to actual values
        actual_params = {}
        for i, (param_name, param_config) in enumerate(PARAMETERS.items()):
            actual_params[param_name] = float(
                param_config["min"] + individual_data["values"][i] * (param_config["max"] - param_config["min"])
            )
        
        # Prepare data structure for YAML
        params_data = {
            "label": label,
            "generation": int(wildcards.gen),
            "individual_id": wildcards.ind,
            "normalized_values": individual_data["values"],
            "parameters": actual_params
        }
        
        # Save parameters as YAML
        os.makedirs(os.path.dirname(output.params), exist_ok=True)
        with open(output.params, "w") as f:
            yaml.dump(params_data, f, default_flow_style=False, sort_keys=False)

# Aggregate fitness for a generation and create next generation
# This is a checkpoint because it creates files (offspring population) that will be needed
# as input for the next generation's rules
checkpoint create_next_generation:
    input:
        fitness_files=lambda wildcards: [
            f"{RUNS_DIR}/{wildcards.gen}_{i:03d}/fitness.yml"
            for i in range(MU if int(wildcards.gen) == 0 else LAMBDA)
        ],
        prev_pop=lambda wildcards: f"{BASE_DIR}/generation_{int(wildcards.gen)}_population.yml" if BASE_DIR else f"generation_{int(wildcards.gen)}_population.yml"
    output:
        next_pop=f"{BASE_DIR}/generation_{{gen,[0-9]+}}_selected.yml" if BASE_DIR else "generation_{gen,[0-9]+}_selected.yml"
    run:
        gen = int(wildcards.gen)
        
        # Load population for this generation
        with open(input.prev_pop, "r") as f:
            pop_data = yaml.safe_load(f)
        
        # Load all fitness values for this generation
        individuals_with_fitness = []
        for i, fitness_file in enumerate(input.fitness_files):
            with open(fitness_file, "r") as f:
                fitness_data = yaml.safe_load(f)
            
            # Find corresponding individual
            label = f"{gen:02d}_{i:03d}"
            ind_data = None
            for ind in pop_data["individuals"]:
                if ind["label"] == label:
                    ind_data = ind.copy()
                    break
            
            if ind_data:
                ind_data["fitness"] = fitness_data["fitness"] if isinstance(fitness_data["fitness"], list) else [fitness_data["fitness"]]
                individuals_with_fitness.append(ind_data)
        
        # Save current generation with fitness
        with open(output.next_pop, "w") as f:
            yaml.dump({"individuals": individuals_with_fitness}, f, default_flow_style=False)
        
        # If not the last generation, create offspring
        if gen < NGEN - 1:
            # Convert to DEAP individuals for selection
            deap_individuals = []
            for ind_data in individuals_with_fitness:
                ind = creator.Individual(ind_data["values"])
                ind.fitness.values = tuple(ind_data["fitness"])
                deap_individuals.append(ind)
            
            # Select best individuals
            selected = toolbox.select(deap_individuals, MU)
            
            # Create offspring using varOr
            offspring = algorithms.varOr(selected, toolbox, LAMBDA, cxpb=0.7, mutpb=0.3)
            
            # Ensure bounds
            for child in offspring:
                for j in range(len(child)):
                    child[j] = max(0.0, min(1.0, child[j]))
            
            # Convert offspring to serializable format
            offspring_data = []
            for i, child in enumerate(offspring):
                offspring_data.append({
                    "label": f"{gen+1:02d}_{i:03d}",
                    "generation": gen + 1,
                    "individual_id": f"{i:03d}",
                    "values": [float(x) for x in child]
                })
            
            # Save offspring population
            pop_file = f"{BASE_DIR}/generation_{gen+1}_population.yml" if BASE_DIR else f"generation_{gen+1}_population.yml"
            with open(pop_file, "w") as f:
                yaml.dump({"individuals": offspring_data}, f, default_flow_style=False)
        
        # Save generation summary
        summary_data = {
            "generation": gen,
            "num_individuals": len(individuals_with_fitness),
            "fitnesses": [ind["fitness"] for ind in individuals_with_fitness],
            "mean_fitness": [float(np.mean([ind["fitness"][i] for ind in individuals_with_fitness])) 
                           for i in range(len(individuals_with_fitness[0]["fitness"]))],
            "max_fitness": [float(np.max([ind["fitness"][i] for ind in individuals_with_fitness])) 
                          for i in range(len(individuals_with_fitness[0]["fitness"]))],
            "min_fitness": [float(np.min([ind["fitness"][i] for ind in individuals_with_fitness])) 
                          for i in range(len(individuals_with_fitness[0]["fitness"]))],
        }
        
        summary_file = f"{BASE_DIR}/generation_{gen}_summary.yml" if BASE_DIR else f"generation_{gen}_summary.yml"
        with open(summary_file, "w") as f:
            yaml.dump(summary_data, f, default_flow_style=False)

# Helper function to aggregate all generation selected files
def get_all_generation_selected_files(wildcards):
    """Get all generation selected files by iterating through checkpoints."""
    selected_files = []
    for gen in range(NGEN):
        if gen > 0:
            # Wait for the checkpoint from previous generation
            checkpoints.create_next_generation.get(gen=gen-1)
        selected_file = f"{BASE_DIR}/generation_{gen}_selected.yml" if BASE_DIR else f"generation_{gen}_selected.yml"
        selected_files.append(selected_file)
    return selected_files

# Final aggregation
rule complete_calibration:
    input:
        get_all_generation_selected_files
    output:
        touch(f"{BASE_DIR}/calibration_complete.done" if BASE_DIR else "calibration_complete.done")
    run:
        print(f"Calibration complete: {NGEN} generations")
        
        # Create final summary with Pareto front
        all_individuals = []
        for gen in range(NGEN):
            selected_file = f"{BASE_DIR}/generation_{gen}_selected.yml" if BASE_DIR else f"generation_{gen}_selected.yml"
            with open(selected_file, "r") as f:
                gen_data = yaml.safe_load(f)
                all_individuals.extend(gen_data["individuals"])
        
        # Find Pareto front (best individuals across all generations)
        deap_all = []
        for ind_data in all_individuals:
            ind = creator.Individual(ind_data["values"])
            ind.fitness.values = tuple(ind_data["fitness"])
            ind.label = ind_data["label"]
            deap_all.append(ind)
        
        pareto_front = tools.ParetoFront()
        pareto_front.update(deap_all)
        
        # Save Pareto front
        pareto_data = []
        for ind in pareto_front:
            pareto_data.append({
                "label": ind.label,
                "values": [float(x) for x in ind],
                "fitness": list(ind.fitness.values)
            })
        
        pareto_file = f"{BASE_DIR}/pareto_front.yml" if BASE_DIR else "pareto_front.yml"
        with open(pareto_file, "w") as f:
            yaml.dump({"pareto_front": pareto_data}, f, default_flow_style=False)
        
        print(f"Pareto front contains {len(pareto_data)} individuals")
