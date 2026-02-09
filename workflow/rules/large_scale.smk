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
EVALUATION_METHODS = "hydrology.plot_discharge,hydrology.evaluate_discharge"

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
    """Get both memory allocation and SLURM partition based on basin size and balanced distribution."""
    area_km2 = get_basin_area_km2(cluster_name)
    
    # Memory allocation based on basin size
    if area_km2 < 200000:  # Smaller basins
        memory_mb = 60000   # 60GB
    else:  # Larger basins
        memory_mb = 200000  # 200GB
    
    # Get cluster index for balanced distribution
    try:
        cluster_index = CLUSTER_NAMES.index(cluster_name)
    except ValueError:
        cluster_index = 0
    
    # IMPROVED BALANCED DISTRIBUTION STRATEGY:
    # Distribute jobs more evenly across partitions to avoid bottlenecks
    # Priority: spread large jobs across defq and ivm-fat, minimize ivm usage
    
    if area_km2 >= 650000:  # Largest basins (>650k km²) - split between defq and ivm-fat
        # Alternate the biggest basins between defq and ivm-fat to spread load
        partition_index = 0 if (cluster_index % 2 == 0) else 2
    elif area_km2 >= 400000:  # Large basins (400k-650k km²) - favor defq with some ivm-fat
        # Most go to defq (has good capacity), some to ivm-fat
        if cluster_index % 3 == 0:
            partition_index = 2  # ivm-fat
        else:
            partition_index = 0  # defq
    elif area_km2 >= 200000:  # Medium basins (200k-400k km²) - defq and ivm
        # Alternate between defq and ivm (avoid overloading ivm-fat)
        partition_index = 0 if (cluster_index % 2 == 0) else 1
    else:  # Small basins (<200k km²) - mainly ivm with some defq
        # Small jobs can go to ivm (less resource contention) with some defq
        partition_index = 1 if (cluster_index % 3 != 0) else 0
    
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

# ============================================================================
# PARALLEL EXECUTION CONFIGURATION
# ============================================================================

# Enable parallel execution mode via config (--config parallel_batch_size=N)
PARALLEL_BATCH_SIZE = config.get("parallel_batch_size", 0)
PARALLEL_MODE = PARALLEL_BATCH_SIZE > 0

if PARALLEL_MODE:
    print(f"🚀 PARALLEL MODE ENABLED: Batching {PARALLEL_BATCH_SIZE} clusters per SLURM job")
    print(f"   Jobs will run clusters in parallel using & and wait pattern")
else:
    print(f"📋 STANDARD MODE: Each cluster gets separate SLURM jobs")

def get_parallel_batches():
    """Split clusters into batches for parallel execution."""
    if not PARALLEL_MODE:
        return []
    
    batches = []
    for i in range(0, len(CLUSTER_NAMES), PARALLEL_BATCH_SIZE):
        batch = CLUSTER_NAMES[i:i+PARALLEL_BATCH_SIZE]
        batches.append((i // PARALLEL_BATCH_SIZE, batch))
    return batches

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
        mem_mb=lambda wildcards: get_resources(wildcards.cluster, 56000)[0],
        runtime=11520,  # 8 days
        cpus=6,
        slurm_partition=lambda wildcards: get_resources(wildcards.cluster, 56000)[1],
        partition_arg=lambda wildcards: get_resources(wildcards.cluster, 56000)[2],
        slurm_account="ivm"
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
        cpus=2,
        slurm_partition="ivm",
        partition_arg="--partition=ivm",
        slurm_account="ivm"
    shell:
        """
        mkdir -p $(dirname {log})
        cd {params.cluster_dir}
        geb evaluate --methods {params.methods} &> {log}
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

# ============================================================================
# PARALLEL BATCH RULES (enabled with --config parallel_batch_size=N)
# Uses & and wait to run multiple clusters in parallel within one SLURM job
# ============================================================================

if PARALLEL_MODE:
    # Build phase - parallel batches
    rule build_parallel_batch:
        output:
            [f"{LARGE_SCALE_DIR}/{c}/base/build.done" for c in get_parallel_batches()[0][1]] if get_parallel_batches() else []
        params:
            clusters=lambda wildcards, input, output: [p.split('/')[6] for p in output],
            batch_id=0
        threads: 2 * PARALLEL_BATCH_SIZE
        resources:
            mem_mb=60000 * PARALLEL_BATCH_SIZE,
            runtime=11520,
            slurm_partition="defq",
            partition_arg="",
            slurm_account="ivm"
        log:
            LARGE_SCALE_DIR + "/.parallel/build_batch_0.log"
        shell:
            """
            mkdir -p $(dirname {log})
            echo "🚀 Parallel build: {params.clusters}" | tee {log}
            
            pids=()
            for cluster in {params.clusters}; do
                cluster_dir="{LARGE_SCALE_DIR}/$cluster/base"
                mkdir -p "$cluster_dir/logs"
                (
                    cd "$cluster_dir"
                    geb build --continue &> logs/build.log
                    touch build.done
                    echo "✓ $cluster" | tee -a {log}
                ) &
                pids+=($!)
            done
            
            failed=0
            for pid in "${{pids[@]}}"; do
                wait $pid || failed=$((failed + 1))
            done
            
            [ $failed -eq 0 ] || (echo "✗ $failed clusters failed" | tee -a {log} && exit 1)
            echo "✓ All clusters completed" | tee -a {log}
            """
    
    # Generate rules for each batch dynamically
    for batch_id, clusters in get_parallel_batches():
        if batch_id == 0:
            continue  # Already defined above as template
        
        # This is a workaround - Snakemake doesn't easily support dynamic rule generation
        # We'll use a single rule with wildcard batch_id instead
        pass
    
    # Actually, let's use a cleaner approach with wildcards
    rule build_batch_parallel:
        output:
            touch(LARGE_SCALE_DIR + "/.parallel/build_batch_{batch_id}.done")
        params:
            clusters=lambda wildcards: get_parallel_batches()[int(wildcards.batch_id)][1],
            outputs=lambda wildcards: [f"{LARGE_SCALE_DIR}/{c}/base/build.done" 
                                       for c in get_parallel_batches()[int(wildcards.batch_id)][1]]
        threads: 2 * PARALLEL_BATCH_SIZE
        resources:
            mem_mb=60000 * PARALLEL_BATCH_SIZE,
            runtime=11520,
            slurm_partition="defq",
            partition_arg="",
            slurm_account="ivm"
        log:
            LARGE_SCALE_DIR + "/.parallel/logs/build_batch_{batch_id}.log"
        shell:
            """
            mkdir -p $(dirname {log})
            echo "🚀 Parallel build batch {wildcards.batch_id}: {params.clusters}" | tee {log}
            
            pids=()
            for cluster in {params.clusters}; do
                cluster_dir="{LARGE_SCALE_DIR}/$cluster/base"
                mkdir -p "$cluster_dir/logs"
                echo "  Starting $cluster..." | tee -a {log}
                (
                    cd "$cluster_dir"
                    geb build --continue &> logs/build.log
                    if [ $? -eq 0 ]; then
                        touch build.done
                        echo "  ✓ $cluster completed" | tee -a {log}
                    else
                        echo "  ✗ $cluster FAILED" | tee -a {log}
                        exit 1
                    fi
                ) &
                pids+=($!)
            done
            
            failed=0
            for pid in "${{pids[@]}}"; do
                wait $pid || failed=$((failed + 1))
            done
            
            if [ $failed -eq 0 ]; then
                echo "✓ Batch {wildcards.batch_id} completed successfully" | tee -a {log}
            else
                echo "✗ Batch {wildcards.batch_id}: $failed clusters failed" | tee -a {log}
                exit 1
            fi
            """
    
    # Spinup phase - parallel batches
    rule spinup_batch_parallel:
        input:
            LARGE_SCALE_DIR + "/.parallel/build_batch_{batch_id}.done"
        output:
            touch(LARGE_SCALE_DIR + "/.parallel/spinup_batch_{batch_id}.done")
        params:
            clusters=lambda wildcards: get_parallel_batches()[int(wildcards.batch_id)][1]
        threads: 4 * PARALLEL_BATCH_SIZE
        resources:
            mem_mb=60000 * PARALLEL_BATCH_SIZE,
            runtime=11520,
            slurm_partition="defq",
            partition_arg="",
            slurm_account="ivm"
        log:
            LARGE_SCALE_DIR + "/.parallel/logs/spinup_batch_{batch_id}.log"
        shell:
            """
            mkdir -p $(dirname {log})
            echo "🚀 Parallel spinup batch {wildcards.batch_id}: {params.clusters}" | tee {log}
            
            pids=()
            for cluster in {params.clusters}; do
                cluster_dir="{LARGE_SCALE_DIR}/$cluster/base"
                echo "  Starting $cluster..." | tee -a {log}
                (
                    cd "$cluster_dir"
                    geb spinup &> logs/spinup.log
                    if [ $? -eq 0 ]; then
                        touch spinup.done
                        echo "  ✓ $cluster completed" | tee -a {log}
                    else
                        echo "  ✗ $cluster FAILED" | tee -a {log}
                        exit 1
                    fi
                ) &
                pids+=($!)
            done
            
            failed=0
            for pid in "${{pids[@]}}"; do
                wait $pid || failed=$((failed + 1))
            done
            
            [ $failed -eq 0 ] || (echo "✗ $failed clusters failed" | tee -a {log} && exit 1)
            echo "✓ Batch {wildcards.batch_id} completed" | tee -a {log}
            """
    
    # Run phase - parallel batches
    rule run_batch_parallel:
        input:
            LARGE_SCALE_DIR + "/.parallel/spinup_batch_{batch_id}.done"
        output:
            touch(LARGE_SCALE_DIR + "/.parallel/run_batch_{batch_id}.done")
        params:
            clusters=lambda wildcards: get_parallel_batches()[int(wildcards.batch_id)][1]
        threads: 6 * PARALLEL_BATCH_SIZE
        resources:
            mem_mb=60000 * PARALLEL_BATCH_SIZE,
            runtime=11520,
            slurm_partition="defq",
            partition_arg="",
            slurm_account="ivm"
        log:
            LARGE_SCALE_DIR + "/.parallel/logs/run_batch_{batch_id}.log"
        shell:
            """
            mkdir -p $(dirname {log})
            echo "🚀 Parallel run batch {wildcards.batch_id}: {params.clusters}" | tee {log}
            
            pids=()
            for cluster in {params.clusters}; do
                cluster_dir="{LARGE_SCALE_DIR}/$cluster/base"
                echo "  Starting $cluster..." | tee -a {log}
                (
                    cd "$cluster_dir"
                    export TMPDIR=/scratch/$SLURM_JOB_ID/$cluster
                    mkdir -p $TMPDIR
                    geb run &> logs/run.log
                    rm -rf $TMPDIR
                    if [ $? -eq 0 ]; then
                        touch run.done
                        echo "  ✓ $cluster completed" | tee -a {log}
                    else
                        echo "  ✗ $cluster FAILED" | tee -a {log}
                        exit 1
                    fi
                ) &
                pids+=($!)
            done
            
            failed=0
            for pid in "${{pids[@]}}"; do
                wait $pid || failed=$((failed + 1))
            done
            
            [ $failed -eq 0 ] || (echo "✗ $failed clusters failed" | tee -a {log} && exit 1)
            echo "✓ Batch {wildcards.batch_id} completed" | tee -a {log}
            """
    
    # Evaluate phase - parallel batches
    rule evaluate_batch_parallel:
        input:
            LARGE_SCALE_DIR + "/.parallel/run_batch_{batch_id}.done"
        output:
            touch(LARGE_SCALE_DIR + "/.parallel/evaluate_batch_{batch_id}.done")
        params:
            clusters=lambda wildcards: get_parallel_batches()[int(wildcards.batch_id)][1]
        threads: 2 * PARALLEL_BATCH_SIZE
        resources:
            mem_mb=16000 * PARALLEL_BATCH_SIZE,
            runtime=11520,
            slurm_partition="ivm",
            partition_arg="--partition=ivm",
            slurm_account="ivm"
        log:
            LARGE_SCALE_DIR + "/.parallel/logs/evaluate_batch_{batch_id}.log"
        shell:
            """
            mkdir -p $(dirname {log})
            echo "🚀 Parallel evaluate batch {wildcards.batch_id}: {params.clusters}" | tee {log}
            
            pids=()
            for cluster in {params.clusters}; do
                cluster_dir="{LARGE_SCALE_DIR}/$cluster/base"
                echo "  Starting $cluster..." | tee -a {log}
                (
                    cd "$cluster_dir"
                    geb evaluate --methods {EVALUATION_METHODS} &> logs/evaluate.log
                    if [ $? -eq 0 ]; then
                        touch evaluate.done
                        echo "  ✓ $cluster completed" | tee -a {log}
                    else
                        echo "  ✗ $cluster FAILED" | tee -a {log}
                        exit 1
                    fi
                ) &
                pids+=($!)
            done
            
            failed=0
            for pid in "${{pids[@]}}"; do
                wait $pid || failed=$((failed + 1))
            done
            
            [ $failed -eq 0 ] || (echo "✗ $failed clusters failed" | tee -a {log} && exit 1)
            echo "✓ Batch {wildcards.batch_id} completed" | tee -a {log}
            """
    
    # Override convenience rules to use parallel batches
    rule build_all_parallel:
        input:
            [f"{LARGE_SCALE_DIR}/.parallel/build_batch_{i}.done" 
             for i, _ in get_parallel_batches()]
    
    rule spinup_all_parallel:
        input:
            [f"{LARGE_SCALE_DIR}/.parallel/spinup_batch_{i}.done" 
             for i, _ in get_parallel_batches()]
    
    rule run_all_parallel:
        input:
            [f"{LARGE_SCALE_DIR}/.parallel/run_batch_{i}.done" 
             for i, _ in get_parallel_batches()]
    
    rule evaluate_all_parallel:
        input:
            [f"{LARGE_SCALE_DIR}/.parallel/evaluate_batch_{i}.done" 
             for i, _ in get_parallel_batches()]
    
    rule all_parallel:
        input:
            [f"{LARGE_SCALE_DIR}/.parallel/evaluate_batch_{i}.done" 
             for i, _ in get_parallel_batches()]
        shell:
            f"""
            echo "✓ All {len(CLUSTER_NAMES)} clusters completed in parallel mode!"
            """
