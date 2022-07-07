#!/bin/bash
#SBATCH --job-name=custom
#SBATCH --output=custom.out
#SBATCH --ntasks=1
#SBATCH --partition=ivm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --time=200:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jens.de.bruijn@vu.nl

source ~/.bashrc

SCRIPT_DIR="/scistor/ivm/jbn271/Packages/GEB/"
cd $SCRIPT_DIR

conda activate a  # activate conda environment

python run.py --headless --scenario spinup
python run.py --headless --scenario base
python run.py --headless --scenario ngo_training
python run.py --headless --scenario government_subsidies
python run.py --headless --scenario base --switch_crops
python run.py --headless --scenario ngo_training --switch_crops
python run.py --headless --scenario government_subsidies --switch_crops