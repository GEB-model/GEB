"""Large-scale multi-basin cluster rules for GEB model.

This module contains rules for running the complete GEB pipeline
on multiple basin clusters created by init_m    resources:
        mem_mb=lambda wildcards: get_resources(wildcards.cluster)[0],
        runtime=11520,  # 8 days
        cpus=6,
        slurm_partition=lambda wildcards: get_resources(wildcards.cluster)[1],
        partition_arg=lambda wildcards: get_resources(wildcards.cluster)[2],
        slurm_account="ivm"
"""

import os
import yaml
from pathlib import Path

# Get configuration with defaults
# CLUSTER_PREFIX should match the value in config/large_scale.yml
CLUSTER_PREFIX = "Europe"
LARGE_SCALE_DIR = "/scistor/ivm/tbr910/GEB/models/large_scale"
EVALUATION_METHODS = "plot_discharge,evaluate_discharge"

# Cache for basin areas to avoid repeated file reads
_basin_area_cache = {}

def get_basin_area_km2(cluster_name):
    """Read basin area from cluster's model.yml file with caching."""
    if cluster_name in _basin_area_cache:
        return _basin_area_cache[cluster_name]
        
    model_yml_path = Path(LARGE_SCALE_DIR) / cluster_name / "base" / "model.yml"
    
    if not model_yml_path.exists():
        _basin_area_cache[cluster_name] = 30000.0
        return 30000.0  # Default area for unknown basins
    
    try:
        with open(model_yml_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Extract basin area if available
        if 'basin' in config_data and 'total_area_km2' in config_data['basin']:
            area = float(config_data['basin']['total_area_km2'])
            _basin_area_cache[cluster_name] = area
            return area
        else:
            _basin_area_cache[cluster_name] = 30000.0
            return 30000.0  # Default area
    except Exception as e:
        _basin_area_cache[cluster_name] = 30000.0
        return 30000.0  # Default area

def get_resources(cluster_name, phase="build"):
    """Get both memory allocation and SLURM partition based on basin size and strategic assignment."""
    area_km2 = get_basin_area_km2(cluster_name)
    
    # Memory allocation based on basin size
    if area_km2 < 200000:  # Smaller basins
        memory_mb = 60000   # 60GB
    else:  # Larger basins
        memory_mb = 200000  # 200GB
    
    # Strategic partition assignment to balance load and limit jobs per partition
    # Assign clusters to partitions based on size and index to ensure balanced distribution
    try:
        cluster_index = CLUSTER_NAMES.index(cluster_name)
    except ValueError:
        cluster_index = 0
    
    # Strategic assignment: prioritize largest basins for ivm-fat, then distribute others
    if area_km2 >= 650000:  # Largest basins (>650k km²) go to ivm-fat
        partition_index = 2  # ivm-fat
    elif area_km2 >= 500000:  # Very large basins (500k-650k km²) go to defq only
        partition_index = 0  # defq
    elif area_km2 >= 200000:  # Medium basins (200k-500k km²) alternate between defq and ivm-fat
        # Alternate between defq and ivm-fat for medium basins
        partition_index = 0 if (cluster_index % 2 == 0) else 2
    else:  # Small basins (<200k km²) go to ivm or defq
        # Small basins alternate between defq and ivm (ivm only gets small basins)
        partition_index = 0 if (cluster_index % 2 == 0) else 1
    
    if partition_index == 0:
        partition = "defq"
        partition_arg = ""  # defq uses default queue (no partition specified)
        # For defq, always use 200GB (since defq handles any memory size)
        memory_mb = 200000  # This overrides the basin-size based allocation!
    elif partition_index == 1:
        partition = "ivm"
        partition_arg = "--partition=ivm"
    else:  # partition_index == 2
        partition = "ivm-fat"
        partition_arg = "--partition=ivm-fat"
        # Ensure ivm-fat gets enough memory
        memory_mb = max(memory_mb, 200000)
    
    return memory_mb, partition, partition_arg

# Dynamically discover cluster directories (run only once)
def get_cluster_names():
    """Find all cluster directories matching the pattern."""
    large_scale_path = Path(LARGE_SCALE_DIR)
    if not large_scale_path.exists():
        return []
    
    cluster_dirs = []
    # Use explicit pattern to avoid f-string issues with Snakemake 7.x
    for cluster_dir in large_scale_path.glob("Europe_*"):
        if cluster_dir.is_dir() and (cluster_dir / "base").exists():
            cluster_dirs.append(cluster_dir.name)
    
    return sorted(cluster_dirs)

# Cache the cluster names to avoid repeated discovery
if 'CLUSTER_NAMES' not in globals():
    CLUSTER_NAMES = get_cluster_names()
    if not CLUSTER_NAMES:
        raise ValueError("No cluster directories found in " + LARGE_SCALE_DIR + " matching pattern Europe_*")

# Cluster names cached for efficient access

# Rule to build a cluster
rule build_cluster:
    output:
        LARGE_SCALE_DIR + "/{cluster}/base/build.done"
    params:
        cluster_dir=lambda wildcards: LARGE_SCALE_DIR + "/" + wildcards.cluster + "/base"
    log:
        LARGE_SCALE_DIR + "/{cluster}/base/logs/build.log"
    resources:
        mem_mb=lambda wildcards: get_resources(wildcards.cluster)[0],
        runtime=11520,  # 8 days
        cpus=2,
        slurm_partition=lambda wildcards: get_resources(wildcards.cluster)[1],
        partition_arg=lambda wildcards: get_resources(wildcards.cluster)[2],
        slurm_account="ivm"
    group: lambda wildcards: get_resources(wildcards.cluster)[1]  # Use partition name as group
    shell:
        """
        mkdir -p $(dirname {log})
        cd {params.cluster_dir}
        geb build --continue &> {log}
        touch {output}
        """

# Rule to run spinup for a cluster
rule spinup_cluster:
    input:
        LARGE_SCALE_DIR + "/{cluster}/base/build.done"  # This depends on build completion
    output:
        LARGE_SCALE_DIR + "/{cluster}/base/spinup.done"
    params:
        cluster_dir=lambda wildcards: LARGE_SCALE_DIR + "/" + wildcards.cluster + "/base"
    log:
        LARGE_SCALE_DIR + "/{cluster}/base/logs/spinup.log"
    resources:
        mem_mb=lambda wildcards: get_resources(wildcards.cluster)[0],
        runtime=11520,  # 8 days
        cpus=4,
        slurm_partition=lambda wildcards: get_resources(wildcards.cluster)[1],
        partition_arg=lambda wildcards: get_resources(wildcards.cluster)[2],
        slurm_account="ivm"
    group: lambda wildcards: get_resources(wildcards.cluster)[1]  # Use partition name as group
    shell:
        """
        mkdir -p $(dirname {log})
        cd {params.cluster_dir}
        geb spinup --continue &> {log}
        touch {output}
        """

# Rule to run main simulation for a cluster
rule run_cluster:
    input:
        LARGE_SCALE_DIR + "/{cluster}/base/spinup.done"
    output:
        LARGE_SCALE_DIR + "/{cluster}/base/run.done"
    params:
        cluster_dir=lambda wildcards: LARGE_SCALE_DIR + "/" + wildcards.cluster + "/base"
    log:
        LARGE_SCALE_DIR + "/{cluster}/base/logs/run.log"
    resources:
        mem_mb=lambda wildcards: get_resources(wildcards.cluster, 56000)[0],
        runtime=11520,  # 8 days
        cpus=6,
        slurm_partition=lambda wildcards: get_resources(wildcards.cluster, 56000)[1],
        partition_arg=lambda wildcards: get_resources(wildcards.cluster, 56000)[2],
        slurm_account="ivm"
    group: lambda wildcards: get_resources(wildcards.cluster)[1]  # Use partition name as group
    shell:
        """
        mkdir -p $(dirname {log})
        cd {params.cluster_dir}
        # Set scratch directory for temporary files to reduce memory usage
        export TMPDIR=/scratch/$SLURM_JOB_ID
        mkdir -p $TMPDIR
        geb run --continue &> {log}
        # Clean up scratch files
        rm -rf $TMPDIR
        touch {output}
        """

# Rule to evaluate a cluster
rule evaluate_cluster:
    input:
        LARGE_SCALE_DIR + "/{cluster}/base/run.done"
    output:
        LARGE_SCALE_DIR + "/{cluster}/base/evaluate.done"
    params:
        cluster_dir=lambda wildcards: LARGE_SCALE_DIR + "/" + wildcards.cluster + "/base",
        methods=EVALUATION_METHODS
    log:
        LARGE_SCALE_DIR + "/{cluster}/base/logs/evaluate.log"
    resources:
        mem_mb=16000,
        runtime=11520,  # 8 days
        cpus=2,
        slurm_partition="ivm",
        slurm_account="ivm"
    group: "ivm"  # Evaluations always run on ivm partition
    shell:
        """
        mkdir -p $(dirname {log})
        cd {params.cluster_dir}
        geb evaluate --continue --methods {params.methods} &> {log}
        touch {output}
        """

# Default target - run everything
rule all:
    input:
        expand(LARGE_SCALE_DIR + "/{cluster}/base/evaluate.done", cluster=CLUSTER_NAMES)
    shell:
        f"""
        echo "All {len(CLUSTER_NAMES)} clusters completed!"
        """

# Convenience rules for running specific phases across all clusters
rule build_all:
    input:
        expand(LARGE_SCALE_DIR + "/{cluster}/base/build.done", cluster=CLUSTER_NAMES)

rule spinup_all:
    input:
        expand(LARGE_SCALE_DIR + "/{cluster}/base/spinup.done", cluster=CLUSTER_NAMES)

rule run_all:
    input:
        expand(LARGE_SCALE_DIR + "/{cluster}/base/run.done", cluster=CLUSTER_NAMES)

rule evaluate_all:
    input:
        expand(LARGE_SCALE_DIR + "/{cluster}/base/evaluate.done", cluster=CLUSTER_NAMES)
