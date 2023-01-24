#!/bin/bash
#SBATCH --job-name=all
#SBATCH --output=all.out
#SBATCH --gres=gpu:2
#SBATCH --ntasks=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=20G
#SBATCH --time=100:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jens.de.bruijn@vu.nl

source ~/.bashrc

SCRIPT_DIR="$HOME/GEB/GEB_private/"
cd $SCRIPT_DIR

module load cuda10.2/toolkit/10.2.89  # load cuda environment
conda activate GEB  # activate conda environment

python run.py --scenario spinup --config $1 --GPU
srun --exclusive --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python run.py --scenario sprinkler --config $1 --GPU &
srun --exclusive --ntasks=1 --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python run.py --scenario sugarcane --config $1 --GPU &
wait
python run.py --scenario base --config $1 --GPU