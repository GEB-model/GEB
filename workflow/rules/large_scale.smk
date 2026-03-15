"""Large-scale multi-basin cluster rules for GEB model.

This module contains rules for running the complete GEB pipeline
on multiple basin clusters created by the 'geb init_multiple' command.
"""

import os
import yaml
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

def get_resources(cluster_name):
    """Get memory allocation and SLURM partition based on basin size.

    Memory strategy: route all jobs to nodes with enough physical memory to
    prevent OOM failures.  The defq partition contains two 1 TB nodes
    (node001/002, each 1008 GB) and several 251 GB nodes.  Requesting 900 GB
    ensures SLURM places jobs only on the 1 TB nodes and the cgroup limit is
    high enough to cover observed peak usage.

    Empirical peak RSS observed in March 2026 runs:
      -  87k km² basin:  254 GB  → ivm (128 GB) is too small; defq required
      - 130k km² basin:  533 GB  → 300 GB defq allocation caused OOM
      - 281k km² basin:  298 GB  → just under the old 300 GB limit
      - 672k km² basin:  398 GB  → exceeded the old 300 GB limit

    Partition nodes and usable memory (from sinfo):
      defq  node001/002:  ~1008 GB  ← target for all non-ivm-fat jobs
      defq  node003-015:   ~251 GB  ← too small; avoided by requesting 900 GB
      ivm-fat node243:     ~755 GB  ← dedicated fat node, serialised

    Args:
        cluster_name: Name of the cluster directory (e.g. "Europe_007").

    Returns:
        Tuple of (memory_mb, partition_name).
    """
    area_km2 = get_basin_area_km2(cluster_name)

    if area_km2 >= 700000:
        # The very largest basins go to the dedicated ivm-fat node (755 GB).
        # ivm_fat_slots=1 serialises all ivm-fat jobs since there is only one
        # such node, preventing two jobs from competing for memory.
        partition = "ivm-fat"
        memory_mb = 700000  # 700 GB, safely below the ~755 GB node limit
    elif area_km2 < 150000:
        # Small basins: empirically need ~250-300 GB.  Requesting 480 GB allows
        # two such jobs to co-schedule on a single 1031 GB node (480+480=960 GB),
        # doubling throughput vs. the 900 GB tier where only 1 job fits per node.
        partition = "defq"
        memory_mb = 480000  # 480 GB → 2 jobs fit per 1031 GB node
    else:
        # All other basins run on the two 1 TB defq nodes (node001/002).
        # Empirical peak usage exceeds 500 GB for some basins, and even basins
        # < 100k km² need > 250 GB, so the ivm partition (128 GB) and the
        # small defq nodes (251 GB) are both insufficient.  Requesting 900 GB
        # guarantees placement on node001/002 and avoids cgroup OOM kills.
        partition = "defq"
        memory_mb = 900000  # 900 GB, safely below the ~1031 GB node001/002 limit

    return memory_mb, partition


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

# Cache the cluster names to avoid repeated discovery
if 'CLUSTER_NAMES' not in globals():
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
        cpus_per_task=2,
        slurm_partition=lambda wildcards: get_resources(wildcards.cluster)[1],
        slurm_account="ivm",
        # Consume 1 ivm_fat_slots token for ivm-fat jobs so Snakemake's scheduler
        # prevents two jobs from sharing the single 773 GB node243 simultaneously.
        ivm_fat_slots=lambda wildcards: 1 if get_resources(wildcards.cluster)[1] == "ivm-fat" else 0,
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
        cpus_per_task=6,
        slurm_partition=lambda wildcards: get_resources(wildcards.cluster)[1],
        slurm_account="ivm",
        ivm_fat_slots=lambda wildcards: 1 if get_resources(wildcards.cluster)[1] == "ivm-fat" else 0,
    shell:
        """
        mkdir -p $(dirname {log})
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
        cpus_per_task=8,
        slurm_partition=lambda wildcards: get_resources(wildcards.cluster)[1],
        slurm_account="ivm",
        ivm_fat_slots=lambda wildcards: 1 if get_resources(wildcards.cluster)[1] == "ivm-fat" else 0,
    shell:
        """
        mkdir -p $(dirname {log})
        cd {params.cluster_dir}
        # Set scratch directory for temporary files to reduce memory usage
        export TMPDIR=/scratch/$SLURM_JOB_ID
        mkdir -p $TMPDIR
        geb run &> {log}
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
        cpus_per_task=2,
        slurm_partition="ivm",
        slurm_account="ivm",
    shell:
        """
        mkdir -p $(dirname {log})
        cd {params.cluster_dir}
        geb evaluate --methods {params.methods} --include-spinup &> {log}
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

