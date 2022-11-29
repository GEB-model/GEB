#!/bin/bash
#SBATCH --partition=ivm
#SBATCH --job-name=ivm_calibrate
#SBATCH --output=calibrate_ivm.out
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=16G
#SBATCH --time=336:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jens.de.bruijn@vu.nl

source ~/.bashrc

SCRIPT_DIR="$HOME/GEB/GEB_private/"
cd $SCRIPT_DIR

conda activate GEB  # activate conda environment

python calibration/calibrate.py --config krishna.yml