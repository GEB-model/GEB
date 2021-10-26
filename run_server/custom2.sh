#!/bin/bash
#SBATCH --job-name=custom
#SBATCH --output=custom.out
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=100:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jens.de.bruijn@vu.nl

source ~/.bashrc

SCRIPT_DIR="/scistor/ivm/jbn271/Packages/GEB/"
cd $SCRIPT_DIR

module load cuda10.2/toolkit/10.2.89  # load cuda environment
conda activate abm  # activate conda environment

python run.py --GPU --headless --scenario government_subsidies --config GEB_switch.yml --gpu_device 1