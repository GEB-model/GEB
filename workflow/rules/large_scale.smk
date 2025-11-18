"""Large-scale multi-basin cluster rules for GEB model.

This module contains rules for running the complete GEB pipeline
on multiple basin clusters created by init_multiple command.
"""

import os
import yaml
from pathlib import Path

# Get configuration with defaults
CLUSTER_PREFIX = config.get("CLUSTER_PREFIX", "WE")
LARGE_SCALE_DIR = config.get("LARGE_SCALE_DIR", "../models/large_scale")
EVALUATION_METHODS = config.get("EVALUATION_METHODS", "plot_discharge,evaluate_discharge")

def get_basin_area_km2(cluster_name):
    """Read basin area from cluster's model.yml file."""
    model_yml_path = Path(LARGE_SCALE_DIR) / cluster_name / "base" / "model.yml"
    
    if not model_yml_path.exists():
        print(f"Warning: model.yml not found for {cluster_name}, using default area")
        return 30000.0  # Default area for unknown basins
    
    try:
        with open(model_yml_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Extract basin area if available
        if 'basin' in config_data and 'total_area_km2' in config_data['basin']:
            area = float(config_data['basin']['total_area_km2'])
            print(f"Basin area for {cluster_name}: {area:,.0f} km²")
            return area
        else:
            print(f"Warning: No basin area found for {cluster_name}, using default")
            return 30000.0  # Default area
    except Exception as e:
        print(f"Error reading basin area for {cluster_name}: {e}")
        return 30000.0  # Default area

def get_resources(cluster_name, base_memory_mb):
    """Get both memory allocation and SLURM partition based on basin size."""
    area_km2 = get_basin_area_km2(cluster_name)
    
    # Define scaling based on basin size
    if area_km2 > 600000:  # Very large basins (like Danube ~817k km²)
        memory_mb = min(base_memory_mb * 4, 128000)  # Up to 128GB
        partition = "ivm-fat" 
    elif area_km2 > 35000:  # Large basins (your preferred threshold)
        memory_mb = min(base_memory_mb * 2, 56000)  # Up to 56GB
        partition = "ivm"
    else:  # Standard basins
        memory_mb = base_memory_mb
        partition = "ivm"
    
    return memory_mb, partition

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

# Print basin areas and memory allocation for each cluster (for debugging)
for cluster_name in CLUSTER_NAMES:
    area_km2 = get_basin_area_km2(cluster_name)
    build_mem, partition = get_resources(cluster_name, 32000)
    run_mem, _ = get_resources(cluster_name, 56000)
    print(f"Cluster {cluster_name}: {area_km2:,.0f} km² → {partition} partition (build: {build_mem/1000:.0f}GB, run: {run_mem/1000:.0f}GB)")

# Rule to build a cluster
rule build_cluster:
    output:
        touch(f"{LARGE_SCALE_DIR}/{{cluster}}/base/build.done")
    params:
        cluster_dir=lambda wildcards: f"{LARGE_SCALE_DIR}/{wildcards.cluster}/base"
    log:
        f"{LARGE_SCALE_DIR}/{{cluster}}/base/logs/build.log"
    resources:
        mem_mb=lambda wildcards: get_resources(wildcards.cluster, 32000)[0],
        runtime=240,
        cpus=2,
        slurm_partition=lambda wildcards: get_resources(wildcards.cluster, 32000)[1],
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
        mem_mb=lambda wildcards: get_resources(wildcards.cluster, 48000)[0],
        runtime=480,
        cpus=4,
        slurm_partition=lambda wildcards: get_resources(wildcards.cluster, 48000)[1],
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
        mem_mb=lambda wildcards: get_resources(wildcards.cluster, 56000)[0],
        runtime=8640,
        cpus=6,
        slurm_partition=lambda wildcards: get_resources(wildcards.cluster, 56000)[1],
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

# Default target - run everything
rule all:
    input:
        expand(f"{LARGE_SCALE_DIR}/{{cluster}}/base/evaluate.done", cluster=CLUSTER_NAMES)
    shell:
        f"""
        echo "All {len(CLUSTER_NAMES)} clusters completed successfully!"
        echo "Completed clusters: {' '.join(CLUSTER_NAMES)}"
        """

# Convenience rules for running specific phases across all clusters
rule build_all:
    input:
        expand(f"{LARGE_SCALE_DIR}/{{cluster}}/base/build.done", cluster=CLUSTER_NAMES)

rule spinup_all:
    input:
        expand(f"{LARGE_SCALE_DIR}/{{cluster}}/base/spinup.done", cluster=CLUSTER_NAMES)

rule run_all:
    input:
        expand(f"{LARGE_SCALE_DIR}/{{cluster}}/base/run.done", cluster=CLUSTER_NAMES)

rule evaluate_all:
    input:
        expand(f"{LARGE_SCALE_DIR}/{{cluster}}/base/evaluate.done", cluster=CLUSTER_NAMES)
