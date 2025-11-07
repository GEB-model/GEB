"""Benchmark-specific rules for GEB model benchmarking.

This module contains rules for running the complete GEB pipeline
on multiple catchments. It reuses patterns from common.smk for consistency.
"""

# Get configuration - no fallbacks
CATCHMENTS = config["CATCHMENTS"]
CATCHMENT_NAMES = list(CATCHMENTS.keys())
BENCHMARK_DIR = "benchmarks"  # Base directory for all benchmark runs

# Rule to initialize a catchment model
rule setup_catchment:
    output:
        model_config=f"{BENCHMARK_DIR}/{{catchment}}/model.yml",
        build_config=f"{BENCHMARK_DIR}/{{catchment}}/build.yml",
        update_config=f"{BENCHMARK_DIR}/{{catchment}}/update.yml"
    params:
        basin_id=lambda wildcards: CATCHMENTS[wildcards.catchment]
    log:
        f"{BENCHMARK_DIR}/{{catchment}}/logs/setup.log"
    shell:
        """
        mkdir -p $(dirname {log})
        geb init --basin-id {params.basin_id} \
                 --working-directory {BENCHMARK_DIR}/{wildcards.catchment} \
                 --overwrite 2>&1 | tee {log}
        """

# Build rule - similar to calibrate.smk's pattern
rule build_catchment:
    input:
        config=f"{BENCHMARK_DIR}/{{catchment}}/model.yml",
        build_config=f"{BENCHMARK_DIR}/{{catchment}}/build.yml",
        update_config=f"{BENCHMARK_DIR}/{{catchment}}/update.yml"
    output:
        touch(f"{BENCHMARK_DIR}/{{catchment}}/build.done")
    log:
        f"{BENCHMARK_DIR}/{{catchment}}/logs/build.log"
    shell:
        """
        mkdir -p $(dirname {log})
        geb build -wd {BENCHMARK_DIR}/{wildcards.catchment} 2>&1 | tee {log}
        """

# Spinup rule
rule spinup_catchment:
    input:
        f"{BENCHMARK_DIR}/{{catchment}}/build.done"
    output:
        touch(f"{BENCHMARK_DIR}/{{catchment}}/spinup.done")
    log:
        f"{BENCHMARK_DIR}/{{catchment}}/logs/spinup.log"
    shell:
        """
        geb spinup -wd {BENCHMARK_DIR}/{wildcards.catchment} 2>&1 | tee {log}
        """

# Run rule
rule run_catchment:
    input:
        f"{BENCHMARK_DIR}/{{catchment}}/spinup.done"
    output:
        touch(f"{BENCHMARK_DIR}/{{catchment}}/run.done")
    log:
        f"{BENCHMARK_DIR}/{{catchment}}/logs/run.log"
    shell:
        """
        geb run -wd {BENCHMARK_DIR}/{wildcards.catchment} 2>&1 | tee {log}
        """

# Evaluate rule - similar to calibrate.smk's evaluate_individual
rule evaluate_catchment:
    input:
        f"{BENCHMARK_DIR}/{{catchment}}/run.done"
    output:
        f"{BENCHMARK_DIR}/{{catchment}}/evaluation.yml"
    log:
        f"{BENCHMARK_DIR}/{{catchment}}/logs/evaluate.log"
    params:
        methods=lambda wildcards: config["EVALUATION_METHODS"]
    shell:
        """
        geb evaluate -wd {BENCHMARK_DIR}/{wildcards.catchment} --methods {params.methods} 2>&1 | tee {log}
        """