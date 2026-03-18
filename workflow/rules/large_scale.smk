"""Large-scale multi-basin cluster rules for GEB model.

This module contains rules for running the complete GEB pipeline
on multiple basin clusters created by the 'geb init_multiple' command.
"""

import os
import yaml
from functools import lru_cache
from pathlib import Path

# Get configuration with defaults
# LARGE_SCALE_DIR and CLUSTER_PREFIX are read from config/large_scale.yml so that
# users only need to change them in one place.
# When not set in the config, fall back to GEB_ROOT env var (always exported by
# the snake alias) so that symlinked paths are preserved.  Using workflow.basedir
# directly would resolve symlinks and break target-path matching.
_geb_root = os.environ.get("GEB_ROOT", str(Path(workflow.basedir).parent))
CLUSTER_PREFIX = config.get("CLUSTER_PREFIX", "Europe")
LARGE_SCALE_DIR = config.get("LARGE_SCALE_DIR", str(Path(_geb_root).parent / "models" / "large_scale"))
EVALUATION_METHODS = config.get("EVALUATION_METHODS", "hydrology.plot_discharge,hydrology.evaluate_discharge")

# Cache for basin areas to avoid repeated file reads
@lru_cache(maxsize=None)
def get_basin_area_km2(cluster_name):
    """Read basin area from cluster's model.yml file with caching."""
    model_yml_path = Path(LARGE_SCALE_DIR) / cluster_name / "base" / "model.yml"
    area = 30000.0  # default for missing or unreadable files

    if model_yml_path.exists():
        try:
            with open(model_yml_path) as f:
                config_data = yaml.safe_load(f)
            area = float(config_data.get("basin", {}).get("total_area_km2", 30000.0))
        except Exception:
            pass

    return area

@lru_cache(maxsize=None)
def get_resources(cluster_name):
    """Get SLURM resources based on basin size.

    SLURM on this cluster does NOT use --mem for placement decisions (observed
    repeatedly: all jobs pile onto one node regardless of --mem). CPU counts ARE
    always enforced as a hard scheduling constraint. We exploit this to control
    concurrency by requesting enough CPUs per job to limit jobs per node:

      defq   node001/002:  64 CPUs,  ~1031 GB  -> 32 CPUs/job -> 2 jobs/node -> 4 total
      ivm-fat node243:    128 CPUs,   ~773 GB  -> 64 CPUs/job -> 2 jobs/node -> 2 total
      Grand total: 6 concurrent jobs

    Memory safety (post ~50% reduction, Mar 2026; worst-case ~267 GB/job):
      defq:    2 x 267 GB = 534 GB  < 1031 GB  OK
      ivm-fat: 2 x 267 GB = 534 GB  <  773 GB  OK

    node003-015 (~257 GB) are always excluded as too small for any basin build.

    Args:
        cluster_name: Name of the cluster directory (e.g. "Europe_007").

    Returns:
        Tuple of (memory_mb, partition_name, cpus_per_task, slurm_extra).
    """
    area_km2 = get_basin_area_km2(cluster_name)
    exclude = "--exclude=node[003-015]"

    if area_km2 >= 700000:
        # Very large basin: pin to defq (1031 GB) in case ivm-fat (773 GB) is too small.
        # 32 CPUs -> max 2 jobs per defq node.
        partition = "defq"
        cpus = 32
        memory_mb = 800000
    elif area_km2 >= 500000:
        # Medium-large basin: route to ivm-fat; 64 CPUs -> max 2 jobs on node243.
        partition = "ivm-fat"
        cpus = 64
        memory_mb = 300000
    else:
        # Small basin: route to defq; 32 CPUs -> max 2 jobs per defq node.
        partition = "defq"
        cpus = 32
        memory_mb = 300000

    return memory_mb, partition, cpus, exclude


# Dynamically discover cluster directories (run only once)
def get_cluster_names():
    """Find all cluster directories matching the CLUSTER_PREFIX pattern."""
    large_scale_path = Path(LARGE_SCALE_DIR)
    if not large_scale_path.exists():
        return []

    cluster_dirs = [
        d.name
        for d in large_scale_path.glob(f"{CLUSTER_PREFIX}_*")
        if d.is_dir() and (d / "base").exists()
    ]
    return sorted(cluster_dirs)

CLUSTER_NAMES = get_cluster_names()
if not CLUSTER_NAMES:
    raise ValueError(
        f"No cluster directories found in {LARGE_SCALE_DIR} "
        f"matching pattern {CLUSTER_PREFIX}_*"
    )

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
        cpus_per_task=lambda wildcards: get_resources(wildcards.cluster)[2],
        slurm_partition=lambda wildcards: get_resources(wildcards.cluster)[1],
        slurm_account="ivm",
        slurm_extra=lambda wildcards: get_resources(wildcards.cluster)[3],
    shell:
        """
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
        cpus_per_task=lambda wildcards: get_resources(wildcards.cluster)[2],
        slurm_partition=lambda wildcards: get_resources(wildcards.cluster)[1],
        slurm_account="ivm",
        slurm_extra=lambda wildcards: get_resources(wildcards.cluster)[3],
    shell:
        """
        cd {params.cluster_dir}
        geb spinup &> {log}
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
        mem_mb=lambda wildcards: get_resources(wildcards.cluster)[0],
        runtime=11520,  # 8 days
        cpus_per_task=lambda wildcards: get_resources(wildcards.cluster)[2],
        slurm_partition=lambda wildcards: get_resources(wildcards.cluster)[1],
        slurm_account="ivm",
        slurm_extra=lambda wildcards: get_resources(wildcards.cluster)[3],
    shell:
        """
        cd {params.cluster_dir}
        geb run &> {log}
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
        runtime=240,  # 4 hours; evaluation is lightweight
        cpus_per_task=2,
        slurm_partition="defq,ivm-fat",
        slurm_account="ivm",
    shell:
        """
        cd {params.cluster_dir}
        geb evaluate --methods {params.methods} --include-spinup &> {log}
        touch {output}
        """

# Default target - run everything
rule all:
    input:
        expand(LARGE_SCALE_DIR + "/{cluster}/base/evaluate.done", cluster=CLUSTER_NAMES)

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

