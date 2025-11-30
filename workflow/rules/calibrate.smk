"""Calibration-specific rules for GEB evolutionary algorithm optimization.

This module contains rules specific to calibration workflows using DEAP:
- Population generation and management
- Fitness aggregation and selection
- Offspring creation
- Pareto front computation

The calibration workflow uses RUNS_DIR (defined in common.smk) to organize
individual model runs in the format: {RUNS_DIR}/{gen}_{ind}/
This pattern is similar to benchmark's {BENCHMARK_DIR}/{catchment}/ pattern.
"""

import random
import yaml
import numpy as np
from deap import base, creator, tools, algorithms
import array
from pathlib import Path

# Load model configuration to get calibration settings
def load_model_config():
    """Load the model.yml configuration file."""
    model_file = Path("model.yml")
    if not model_file.exists():
        raise FileNotFoundError(f"Model configuration file not found: {model_file}")
    
    with open(model_file, "r") as f:
        return yaml.safe_load(f)

MODEL_CONFIG = load_model_config()
CALIBRATION_CONFIG = MODEL_CONFIG["calibration"]

# Calibration-specific configuration from model.yml
NGEN = CALIBRATION_CONFIG["DEAP"]["ngen"]  # Number of generations
MU = CALIBRATION_CONFIG["DEAP"]["mu"]  # Population size
LAMBDA = CALIBRATION_CONFIG["DEAP"]["lambda_"]  # Offspring size per generation

# Parameter configuration from model.yml calibration section
PARAMETERS = {}
for param_name, param_config in CALIBRATION_CONFIG["parameters"].items():
    PARAMETERS[param_name] = {
        "min": param_config["min"],
        "max": param_config["max"],
        "variable": param_config["variable"]
    }

N_PARAMETERS = len(PARAMETERS)

# Fitness configuration from calibration targets
CALIBRATION_TARGETS = CALIBRATION_CONFIG["calibration_targets"]
weights = [CALIBRATION_TARGETS[target] for target in CALIBRATION_TARGETS.keys()]

# Initialize DEAP components
creator.create("FitnessMulti", base.Fitness, weights=weights)
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
toolbox.register("mate", tools.cxBlend, alpha=CALIBRATION_CONFIG["DEAP"]["blend_alpha"])
toolbox.register("mutate", tools.mutGaussian, 
                 mu=0, 
                 sigma=CALIBRATION_CONFIG["DEAP"]["gaussian_sigma"], 
                 indpb=CALIBRATION_CONFIG["DEAP"]["gaussian_indpb"])
toolbox.register("select", tools.selNSGA2)

# Set random seed for reproducibility
random.seed(42)

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
        "base_build.done"
    output:
        "generation_0_population.yml"
    run:
        # Save initial population
        with open(output[0], "w") as f:
            yaml.dump({"individuals": INITIAL_POPULATION}, f, default_flow_style=False)

# Generate parameter file for a specific individual
rule generate_individual_parameters:
    input:
        pop_file=lambda wildcards: "generation_0_population.yml" if int(wildcards.gen) == 0 else f"generation_{int(wildcards.gen) - 1}_next.yml",
        # Ensure previous generation's checkpoint is complete before creating params for the next generation.
        # This is the key to resolving the cyclic dependency for generational workflows.
        checkpoint_done=lambda wildcards: checkpoints.create_next_generation.get(gen=int(wildcards.gen) - 1).output if int(wildcards.gen) > 0 else []
    output:
        params=f"{RUNS_DIR}/{{gen}}_{{ind}}/parameters.yml"
    run:
        import yaml
        
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
            actual_params[param_config["variable"]] = float(
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
        Path(output.params).parent.mkdir(parents=True, exist_ok=True)
        with open(output.params, "w") as f:
            yaml.dump(params_data, f, default_flow_style=False, sort_keys=False)

# Aggregate fitness for a generation and create next generation
checkpoint create_next_generation:
    input:
        fitness_files=lambda wildcards: [
            f"{RUNS_DIR}/{wildcards.gen}_{i:03d}/fitness.yml"
            for i in range(MU if int(wildcards.gen) == 0 else LAMBDA)
        ],
        prev_pop=lambda wildcards: "generation_0_population.yml" if int(wildcards.gen) == 0 else f"generation_{int(wildcards.gen) - 1}_next.yml"
    output:
        selected_pop="generation_{gen}_selected.yml",
        summary="generation_{gen}_summary.yml",
        next_pop="generation_{gen}_next.yml"
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
            label = f"{gen}_{i:03d}"
            ind_data = None
            for ind in pop_data["individuals"]:
                if ind["label"] == label:
                    ind_data = ind.copy()
                    break
            
            if ind_data:
                ind_data["fitness"] = fitness_data["fitness"] if isinstance(fitness_data["fitness"], list) else [fitness_data["fitness"]]
                individuals_with_fitness.append(ind_data)
        
        # Save current generation with fitness
        with open(output.selected_pop, "w") as f:
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
            offspring = algorithms.varOr(selected, toolbox, LAMBDA, 
                                       cxpb=CALIBRATION_CONFIG["DEAP"]["crossover_prob"], 
                                       mutpb=CALIBRATION_CONFIG["DEAP"]["mutation_prob"])
            
            # Ensure bounds
            for child in offspring:
                for j in range(len(child)):
                    child[j] = max(0.0, min(1.0, child[j]))
            
            # Convert offspring to serializable format
            offspring_data = []
            for i, child in enumerate(offspring):
                offspring_data.append({
                    "label": f"{gen+1}_{i:03d}",
                    "generation": gen + 1,
                    "individual_id": f"{i:03d}",
                    "values": [float(x) for x in child]
                })
            
            # Save offspring population for the next generation
            with open(output.next_pop, "w") as f:
                yaml.dump({"individuals": offspring_data}, f, default_flow_style=False)
        else:
            # Last generation - create empty population file
            with open(output.next_pop, "w") as f:
                yaml.dump({"individuals": []}, f, default_flow_style=False)
        
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
        
        with open(output.summary, "w") as f:
            yaml.dump(summary_data, f, default_flow_style=False)


# Helper function to aggregate all generation selected files from the checkpoint
def aggregate_checkpoint_outputs(wildcards):
    """Get all generation_selected.yml files by iterating through checkpoints."""
    # Trigger the final checkpoint to ensure all generations are complete
    checkpoints.create_next_generation.get(gen=NGEN - 1)
    # Now we can safely construct the list of all selected files
    return [f"generation_{gen}_selected.yml" for gen in range(NGEN)]


# Final aggregation
rule complete_calibration:
    input:
        aggregate_checkpoint_outputs
    output:
        touch("calibration_complete.done")
    run:
        print(f"Calibration complete: {NGEN} generations")
        
        # Create final summary with Pareto front
        all_individuals = []
        for gen in range(NGEN):
            selected_file = f"generation_{gen}_selected.yml"
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
        
        pareto_file = "pareto_front.yml"
        with open(pareto_file, "w") as f:
            yaml.dump({"pareto_front": pareto_data}, f, default_flow_style=False)
        
        print(f"Pareto front contains {len(pareto_data)} individuals")
