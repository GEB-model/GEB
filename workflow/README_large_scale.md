# Large-Scale Multi-Basin Cluster Workflow

This Snakemake workflow efficiently runs the complete GEB pipeline on multiple basin clusters created by the `geb init_multiple` command using SLURM for parallel execution.

## Quick Start

1. **Create clusters** using `geb init_multiple`:
   ```bash
   geb init_multiple --geometry-bounds="5.0,50.0,15.0,55.0" --cluster-prefix="test"
   ```

2. **Submit to SLURM cluster** (recommended):
   ```bash
   cd /scistor/ivm/tbr910/GEB/sh_scripts
   sbatch run_large_scale_slurm.sh
   ```

## Usage Options

### SLURM Cluster Execution (Recommended)

```bash
cd /scistor/ivm/tbr910/GEB/sh_scripts

# Complete pipeline with default settings
sbatch run_large_scale_slurm.sh

# Build phase only with 20 parallel jobs
sbatch run_large_scale_slurm.sh build cluster 20

# Run specific phase with custom parameters
sbatch run_large_scale_slurm.sh [COMMAND] [EXECUTION_MODE] [MAX_JOBS] [LARGE_SCALE_DIR] [PREFIX]
```

**Parameters:**
- `COMMAND`: `all`, `build`, `spinup`, `run`, `evaluate`
- `EXECUTION_MODE`: `cluster` (parallel across nodes) or `local` (single node)
- `MAX_JOBS`: Number of parallel jobs (default: 10)
- `LARGE_SCALE_DIR`: Directory with cluster models (default: `/scistor/ivm/tbr910/GEB/models/large_scale`)
- `PREFIX`: Cluster prefix (default: `test`)

### Direct Snakemake Usage (Advanced)

```bash
cd /scistor/ivm/tbr910/GEB/GEB/workflow

# Run with cluster-generic executor
snakemake --snakefile Snakefile_large_scale \
          --configfile config/large_scale.yml \
          --executor cluster-generic \
          --cluster-generic-submit-cmd "sbatch --parsable ..." \
          --jobs 10
```

## Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# View controller job output
tail -f /net/sys/pscst201/BETA-IVM-HPC@ada-nodes/tbr910/GEB/sh_scripts/logs/large_scale_cluster-<job_id>.out

# View individual cluster job logs  
tail -f /net/sys/pscst201/BETA-IVM-HPC@ada-nodes/tbr910/GEB/sh_scripts/slurm_logs/build_cluster-cluster=test_000-<job_id>.out

# View cluster-specific logs
tail -f /scistor/ivm/tbr910/GEB/models/large_scale/test_000/base/logs/build.log
```

## Configuration

Edit `/scistor/ivm/tbr910/GEB/GEB/workflow/config/large_scale.yml` to customize:

- **LARGE_SCALE_DIR**: Directory containing cluster models
- **CLUSTER_PREFIX**: Prefix used for cluster directories (e.g., "test" for test_000, test_001...)
- **HIGH_MEM_CLUSTERS**: Clusters requiring >60GB RAM (uses ivm-fat partition)
- **EVALUATION_METHODS**: Which evaluation methods to run

### Memory Configuration by Basin Type

Large basins like Danube automatically get high-memory allocation:
```yaml
HIGH_MEM_CLUSTERS: ["danube_001", "mississippi_002"]  # >60GB RAM, ivm-fat partition
```

Standard clusters use regular ivm partition with configurable memory.

## Resource Management

The workflow automatically:
- **Distributes jobs across multiple nodes** using SLURM
- **Manages memory allocation** by basin type:
  - Build: 32GB (large basins: 65GB)
  - Spinup: 48GB (large basins: 98GB)  
  - Run: 56GB (large basins: 131GB on ivm-fat)
  - Evaluate: 16GB (large basins: 32GB)
- **Limits concurrent jobs** based on `MAX_JOBS` parameter
- **Handles failures** gracefully with restart capability
- **Tracks progress** with `.done` files

## Job Architecture

**Cluster Mode (Recommended):**
```
Controller Job (8GB, ivm)
├── Build Jobs (32-65GB each, distributed across nodes)
├── Spinup Jobs (48-98GB each, after builds complete)  
├── Run Jobs (56-131GB each, after spinups complete)
└── Evaluate Jobs (16-32GB each, after runs complete)
```

Each cluster runs as a **separate SLURM job** for maximum parallelism.

## Advanced Usage

## Examples

```bash
# Complete pipeline for test clusters with 15 parallel jobs
sbatch run_large_scale_slurm.sh all cluster 15

# Build Danube clusters (automatically uses ivm-fat partition)  
sbatch run_large_scale_slurm.sh build cluster 5 /path/to/large_scale danube

# Run evaluation only for specific clusters
sbatch run_large_scale_slurm.sh evaluate cluster 10

# Local mode (single node, testing)
sbatch run_large_scale_slurm.sh build local 4
```

## Output Structure

```
large_scale/
├── test_000/
│   ├── base/
│   │   ├── build.done
│   │   ├── spinup.done
│   │   ├── run.done
│   │   ├── evaluate.done
│   │   ├── logs/
│   │   ├── input/
│   │   └── output/
│   └── complete.done
├── test_001/
│   └── ...
├── all_builds.done
├── all_spinups.done
├── all_runs.done
├── all_evaluations.done
└── all_complete.done
```

## Log Locations

```
/net/sys/pscst201/BETA-IVM-HPC@ada-nodes/tbr910/GEB/sh_scripts/
├── logs/                                    # Controller job logs
│   └── large_scale_cluster-<job_id>.out
├── slurm_logs/                             # Individual cluster job logs
│   ├── build_cluster-cluster=test_000-<job_id>.out
│   └── run_cluster-cluster=danube_001-<job_id>.out
└── .snakemake/                             # Snakemake metadata
```

Individual cluster logs:
```
/scistor/ivm/tbr910/GEB/models/large_scale/test_000/base/logs/
├── build.log
├── spinup.log  
├── run.log
└── evaluate.log
```

## Performance Tips

1. **Memory Management**: Large basins automatically use high-memory nodes
2. **Parallel Jobs**: Use `MAX_JOBS=20` for faster execution with many clusters
3. **Monitoring**: Use `squeue -u $USER` to track job progress
4. **Resume**: Workflow automatically resumes from last completed phase
5. **Node Distribution**: Jobs automatically spread across available compute nodes

## Troubleshooting

### Common Issues
- **"Directory cannot be locked"**: Previous run didn't clean up properly
  ```bash
  cd /scistor/ivm/tbr910/GEB/GEB/workflow
  snakemake --snakefile Snakefile_large_scale --configfile config/large_scale.yml --unlock
  ```

- **Memory errors**: Add cluster to `HIGH_MEM_CLUSTERS` in config or check if it's a large basin

- **Jobs not appearing**: Ensure you're using `cluster` mode, not `local` mode

### Log Analysis
- **Controller issues**: Check `/net/sys/.../logs/large_scale_cluster-*.out`
- **Individual cluster issues**: Check `/net/sys/.../slurm_logs/build_cluster-cluster=*-*.out`
- **GEB model issues**: Check `/scistor/.../models/large_scale/*/base/logs/*.log`
