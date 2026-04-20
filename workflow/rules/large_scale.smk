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

    Memory profiling across all 17 European clusters (April 2026) shows that
    setup_elevation dominates peak RSS at ~105 GB. 200 GB provides ~90 %
    headroom above the observed peak, avoiding OOM on nodes with limited RAM.
    defq has MaxTime=7 days; build/run stay within that, spinup uses ivm-fat.

    CPU efficiency profiling shows spinup uses only ~1.4% CPU efficiency with
    64 cores — the land surface model is memory-bandwidth bound, not compute
    bound at high core counts. Spinup CPUs are therefore kept low (16) so that
    more jobs can share the two ivm-fat nodes concurrently:
      node001: 64 CPUs → 4 × 16-CPU jobs
      node243: 128 CPUs → 8 × 16-CPU jobs
    Build and run use 4 CPUs (mostly single-threaded I/O-bound work).
    """
    mem_mb = 200000
    cpus_build_run = 4
    cpus_spinup = 16
    partition = "defq,ivm,ivm-fat"
    extra = ""
    return mem_mb, partition, cpus_build_run, cpus_spinup, extra


# defq MaxTime = 7 days (10080 min); build and run fit within 5 days.
# ivm-fat has MaxTime=UNLIMITED, so spinup runtime is set to a large value (30
# days) to prevent Snakemake's default-resources fallback from being applied as
# seconds instead of minutes (plugin behaviour difference for default vs explicit).
RUNTIME_BUILD_MIN = 7200    # 5 days
RUNTIME_SPINUP_MIN = 43200  # 30 days (effectively unlimited on ivm-fat)
RUNTIME_RUN_MIN = 7200      # 5 days


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
        runtime=RUNTIME_BUILD_MIN,
        cpus_per_task=lambda wildcards: get_resources(wildcards.cluster)[2],
        slurm_partition=lambda wildcards: get_resources(wildcards.cluster)[1],
        slurm_account="ivm",
        slurm_extra=lambda wildcards: get_resources(wildcards.cluster)[4],
    shell:
        """
        # Route temp files to the per-job scratch directory so that large
        # intermediate zarr files written by write_zarr do not fill /tmp,
        # which caused Python exit-code-1 crashes on some nodes (e.g. Europe_016).
        if [ -d "/scratch/$SLURM_JOB_ID" ]; then export TMPDIR=/scratch/$SLURM_JOB_ID; fi
        # Cap Numba TBB thread count to the allocated CPUs. Without this, TBB
        # reads os.cpu_count() (the full node, e.g. 128 on node243) and spawns
        # far more threads than the job has CPU slots, causing oversubscription
        # when multiple jobs share the same node.
        export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
        cd {params.cluster_dir}
        # Ensure input data version matches the current GEB version before building.
        # This auto-runs any run-update methods (e.g. new data setup steps added in
        # version updates) so that --continue builds pick up new required data.
        geb update-version &>> {log} || true
        geb build --continue &>> {log}
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
        runtime=RUNTIME_SPINUP_MIN,
        cpus_per_task=lambda wildcards: get_resources(wildcards.cluster)[3],
        # ivm-fat has MaxTime=UNLIMITED; runtime is set explicitly here to avoid
        # Snakemake's default-resources fallback being applied as seconds instead
        # of minutes, which would cap jobs at 3:12:00.
        slurm_partition="ivm-fat",
        slurm_account="ivm",
        slurm_extra=lambda wildcards: get_resources(wildcards.cluster)[4],
    shell:
        """
        if [ -d "/scratch/$SLURM_JOB_ID" ]; then export TMPDIR=/scratch/$SLURM_JOB_ID; fi
        export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
        # Enable AVX/AVX-512 instructions. ivm-fat nodes (zen2/zen4) support AVX2/AVX-512
        # and do not suffer from the frequency-throttling issue seen on some Intel CPUs.
        export NUMBA_ENABLE_AVX=1
        cd {params.cluster_dir}
        # Ensure input data version matches the current GEB version before running.
        geb update-version &>> {log}
        geb spinup --timing &>> {log}
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
        runtime=RUNTIME_RUN_MIN,
        cpus_per_task=lambda wildcards: get_resources(wildcards.cluster)[2],
        slurm_partition=lambda wildcards: get_resources(wildcards.cluster)[1],
        slurm_account="ivm",
        slurm_extra=lambda wildcards: get_resources(wildcards.cluster)[4],
    shell:
        """
        if [ -d "/scratch/$SLURM_JOB_ID" ]; then export TMPDIR=/scratch/$SLURM_JOB_ID; fi
        export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
        export NUMBA_ENABLE_AVX=1
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

