"""Large-scale multi-basin cluster rules for GEB model.

This module contains rules for running the complete GEB pipeline
on multiple basin clusters created by the 'geb init_multiple' command.
"""

import os
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

def get_resources(cluster_name):
    """Get SLURM resources for a cluster job.

    All clusters use the same resource allocation. CPU count is the effective
    concurrency knob (SLURM on this cluster ignores --mem for placement).

    Args:
        cluster_name: Name of the cluster directory (e.g. "Europe_007").

    Returns:
        Tuple of (memory_mb, partition_name, cpus_per_task, slurm_extra).
    """
    exclude = "--exclude=node[003-015]"

    # Resource strategy (updated Mar 2026):
    # Actual peak RSS from SACCT across all phases:
    #   build_cluster:  max ~190 GB  (job 1239390)
    #   spinup_cluster: max ~287 GB  (job 1234070)
    #   run_cluster:    max ~263 GB  (job 1234403)
    # → 300 GB covers all three phases with a ~1.05-1.6× safety margin.
    #
    # SLURM on this cluster does not use --mem for placement, so CPUs are the
    # effective concurrency knob. Using partition "defq,ivm-fat" allows SLURM to
    # schedule on whichever node is free first:
    #   defq  node001/002: 64 CPUs, 1031 GB → 2 jobs × 32 CPUs = 64 CPUs filled
    #                                          2 jobs × 300 GB  = 600 GB  < 1031 GB ✓
    #                                          → 4 concurrent jobs (2 nodes)
    #   ivm-fat node243:  128 CPUs,  773 GB → RAM limits to 2 jobs (2×300=600 GB)
    #                                          → 2 concurrent jobs
    #   Grand total: up to 6 concurrent jobs
    partition = "defq,ivm-fat"
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
        # Route temp files to the per-job scratch directory so that large
        # intermediate zarr files written by write_zarr do not fill /tmp,
        # which caused Python exit-code-1 crashes on some nodes (e.g. Europe_016).
        if [ -d "/scratch/$SLURM_JOB_ID" ]; then export TMPDIR=/scratch/$SLURM_JOB_ID; fi
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
        if [ -d "/scratch/$SLURM_JOB_ID" ]; then export TMPDIR=/scratch/$SLURM_JOB_ID; fi
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
        if [ -d "/scratch/$SLURM_JOB_ID" ]; then export TMPDIR=/scratch/$SLURM_JOB_ID; fi
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

