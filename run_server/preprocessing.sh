#!/bin/bash
#SBATCH --job-name=preprocessing
#SBATCH --output=preprocessing.out
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=100:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jens.de.bruijn@vu.nl

source ~/.bashrc

SCRIPT_DIR="$HOME/GEB/GEB_private/"
cd $SCRIPT_DIR

conda activate GEB  # activate conda environment

python preprocessing/10_iterative_proportional_fitting.py --config bhima.yml
python preprocessing/11_create_farmers.py --config bhima.yml
python preprocessing/12_preprocess_modflow.py --config bhima.yml
python run.py --scenario spinup --config bhima.yml
python run.py --scenario sugarcane --config bhima.yml
