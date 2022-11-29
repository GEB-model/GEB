#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=calibrate
#SBATCH --output=calibrate.out
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=480G
#SBATCH --time=336:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jens.de.bruijn@vu.nl

source ~/.bashrc

SCRIPT_DIR="$HOME/GEB/GEB_private/"
cd $SCRIPT_DIR

conda activate GEB  # activate conda environment

python calibration/calibrate.py --config krishna.yml
