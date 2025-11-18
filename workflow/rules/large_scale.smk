"""Large-scale multi-basin cluster rules for GEB model.

This module contains rules for running the complete GEB pipeline
on multiple basin clusters created by init_multiple command.
"""

import os
from pathlib import Path

# Get configuration with defaults
CLUSTER_PREFIX = config.get("CLUSTER_PREFIX", "test")
LARGE_SCALE_DIR = config.get("LARGE_SCALE_DIR", "../models/large_scale")
EVALUATION_METHODS = config.get("EVALUATION_METHODS", "plot_discharge,evaluate_discharge")

# Dynamically discover cluster directories
def get_cluster_names():
    """Find all cluster directories matching the pattern."""
    large_scale_path = Path(LARGE_SCALE_DIR)
    if not large_scale_path.exists():
        return []
    
    cluster_dirs = []
    for cluster_dir in large_scale_path.glob(f"{CLUSTER_PREFIX}_*"):
        if cluster_dir.is_dir() and (cluster_dir / "base").exists():
            cluster_dirs.append(cluster_dir.name)
    
    return sorted(cluster_dirs)

CLUSTER_NAMES = get_cluster_names()

if not CLUSTER_NAMES:
    raise ValueError(f"No cluster directories found in {LARGE_SCALE_DIR} matching pattern {CLUSTER_PREFIX}_*")

print(f"Found {len(CLUSTER_NAMES)} clusters: {CLUSTER_NAMES}")

# Configure which clusters need high memory (can be set in config)
HIGH_MEM_CLUSTERS = config.get("HIGH_MEM_CLUSTERS", [])

# Rule order: prefer standard run over high-memory run
ruleorder: run_cluster > run_cluster_highmem

# Rule to build a cluster
rule build_cluster:
    output:
        touch(f"{LARGE_SCALE_DIR}/{{cluster}}/base/build.done")
    params:
        cluster_dir=lambda wildcards: f"{LARGE_SCALE_DIR}/{wildcards.cluster}/base"
    log:
        f"{LARGE_SCALE_DIR}/{{cluster}}/base/logs/build.log"
    resources:
        mem_mb=32000,  # Increased from 16GB to 32GB
        runtime=240,
        cpus=2,
        slurm_partition="ivm",
        slurm_account=os.environ.get("USER", "default")
    shell:
        """
        mkdir -p $(dirname {log})
        cd {params.cluster_dir}
        geb build 2>&1 | tee {log}
        """

# Rule to run spinup for a cluster
rule spinup_cluster:
    input:
        f"{LARGE_SCALE_DIR}/{{cluster}}/base/build.done"
    output:
        touch(f"{LARGE_SCALE_DIR}/{{cluster}}/base/spinup.done")
    params:
        cluster_dir=lambda wildcards: f"{LARGE_SCALE_DIR}/{wildcards.cluster}/base"
    log:
        f"{LARGE_SCALE_DIR}/{{cluster}}/base/logs/spinup.log"
    resources:
        mem_mb=48000,  # Increased from 24GB to 48GB
        runtime=480,
        cpus=4,
        slurm_partition="ivm",
        slurm_account=os.environ.get("USER", "default")
    shell:
        """
        mkdir -p $(dirname {log})
        cd {params.cluster_dir}
        geb spinup 2>&1 | tee {log}
        """

# Rule to run main simulation for a cluster
rule run_cluster:
    input:
        f"{LARGE_SCALE_DIR}/{{cluster}}/base/spinup.done"
    output:
        touch(f"{LARGE_SCALE_DIR}/{{cluster}}/base/run.done")
    params:
        cluster_dir=lambda wildcards: f"{LARGE_SCALE_DIR}/{wildcards.cluster}/base"
    log:
        f"{LARGE_SCALE_DIR}/{{cluster}}/base/logs/run.log"
    resources:
        mem_mb=56000,  # Increased from 32GB to 56GB (max for ivm partition)
        runtime=1440,
        cpus=6,
        slurm_partition="ivm",  # Using regular ivm partition (56GB < 60GB limit)
        slurm_account=os.environ.get("USER", "default")
    shell:
        """
        mkdir -p $(dirname {log})
        cd {params.cluster_dir}
        # Set scratch directory for temporary files to reduce memory usage
        export TMPDIR=/scratch/$SLURM_JOB_ID
        mkdir -p $TMPDIR
        geb run 2>&1 | tee {log}
        # Clean up scratch files
        rm -rf $TMPDIR
        """

# Rule for high-memory simulation runs (use when standard run needs >60GB)
rule run_cluster_highmem:
    input:
        f"{LARGE_SCALE_DIR}/{{cluster}}/base/spinup.done"
    output:
        touch(f"{LARGE_SCALE_DIR}/{{cluster}}/base/run.done")
    params:
        cluster_dir=lambda wildcards: f"{LARGE_SCALE_DIR}/{wildcards.cluster}/base"
    log:
        f"{LARGE_SCALE_DIR}/{{cluster}}/base/logs/run_highmem.log"
    wildcard_constraints:
        cluster="|".join(HIGH_MEM_CLUSTERS) if HIGH_MEM_CLUSTERS else "NONE"
    resources:
        mem_mb=128000,  # 128GB for large simulations
        runtime=1440,
        cpus=8,
        slurm_partition="ivm-fat",  # Use fat partition for high RAM jobs
        slurm_account=os.environ.get("USER", "default")
    shell:
        """
        mkdir -p $(dirname {log})
        cd {params.cluster_dir}
        # Set scratch directory for temporary files
        export TMPDIR=/scratch/$SLURM_JOB_ID
        mkdir -p $TMPDIR
        geb run 2>&1 | tee {log}
        # Clean up scratch files
        rm -rf $TMPDIR
        """

# Rule to evaluate a cluster
rule evaluate_cluster:
    input:
        f"{LARGE_SCALE_DIR}/{{cluster}}/base/run.done"
    output:
        touch(f"{LARGE_SCALE_DIR}/{{cluster}}/base/evaluate.done")
    params:
        cluster_dir=lambda wildcards: f"{LARGE_SCALE_DIR}/{wildcards.cluster}/base",
        methods=EVALUATION_METHODS
    log:
        f"{LARGE_SCALE_DIR}/{{cluster}}/base/logs/evaluate.log"
    resources:
        mem_mb=16000,
        runtime=120,
        cpus=2,
        slurm_partition="ivm",
        slurm_account=os.environ.get("USER", "default")
    shell:
        """
        mkdir -p $(dirname {log})
        cd {params.cluster_dir}
        geb evaluate --methods {params.methods} 2>&1 | tee {log}
        """

# Rule to run complete pipeline for a single cluster
rule complete_cluster:
    input:
        f"{LARGE_SCALE_DIR}/{{cluster}}/base/evaluate.done"
    output:
        touch(f"{LARGE_SCALE_DIR}/{{cluster}}/complete.done")

# Rule to run all clusters through build phase
rule build_all_clusters:
    input:
        expand(f"{LARGE_SCALE_DIR}/{{cluster}}/base/build.done", cluster=CLUSTER_NAMES)
    output:
        touch(f"{LARGE_SCALE_DIR}/all_builds.done")

# Rule to run all clusters through spinup phase
rule spinup_all_clusters:
    input:
        expand(f"{LARGE_SCALE_DIR}/{{cluster}}/base/spinup.done", cluster=CLUSTER_NAMES)
    output:
        touch(f"{LARGE_SCALE_DIR}/all_spinups.done")

# Rule to run all clusters through main simulation
rule run_all_clusters:
    input:
        expand(f"{LARGE_SCALE_DIR}/{{cluster}}/base/run.done", cluster=CLUSTER_NAMES)
    output:
        touch(f"{LARGE_SCALE_DIR}/all_runs.done")

# Rule to evaluate all clusters
rule evaluate_all_clusters:
    input:
        expand(f"{LARGE_SCALE_DIR}/{{cluster}}/base/evaluate.done", cluster=CLUSTER_NAMES)
    output:
        touch(f"{LARGE_SCALE_DIR}/all_evaluations.done")

# Rule to run complete pipeline for all clusters
rule all_clusters_complete:
    input:
        expand(f"{LARGE_SCALE_DIR}/{{cluster}}/complete.done", cluster=CLUSTER_NAMES)
    output:
        touch(f"{LARGE_SCALE_DIR}/all_complete.done")
    shell:
        f"""
        echo "All {len(CLUSTER_NAMES)} clusters completed successfully!"
        echo "Completed clusters: {' '.join(CLUSTER_NAMES)}"
        """

# Default target - run everything
rule all:
    input:
        expand(f"{LARGE_SCALE_DIR}/{{cluster}}/complete.done", cluster=CLUSTER_NAMES)
