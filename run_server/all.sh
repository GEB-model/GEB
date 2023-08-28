#!/bin/bash

#SBATCH --job-name=all
#SBATCH --output=all.out
#SBATCH --gres=gpu:2
#SBATCH --ntasks=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1G
#SBATCH --time=0-100:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jens.de.bruijn@vu.nl

source ~/.bashrc

SCRIPT_DIR="$HOME/GEB/GEB_private/"
cd $SCRIPT_DIR

module load cuda10.2/toolkit/10.2.89  # load cuda environment
conda activate GEB  # activate conda environment

python run.py --scenario spinup --config $1 --GPU
python run.py --scenario noadaptation --config $1 --GPU --gpu_device 0 &
python run.py --scenario sprinkler --config $1 --GPU --gpu_device 1 &
wait
python run.py --scenario base --config $1 --GPU --gpu_device 0 &
wait