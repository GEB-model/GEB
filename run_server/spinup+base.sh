#!/bin/bash
#SBATCH --job-name=spinup+base
#SBATCH --output=spinup+base.out
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=100:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jens.de.bruijn@vu.nl

source ~/.bashrc

SCRIPT_DIR="$HOME/GEB/GEB_private/"
cd $SCRIPT_DIR

# module load cuda10.2/toolkit/10.2.89  # load cuda environment
conda activate GEB  # activate conda environment

python run.py --scenario spinup --config krishna.yml
python run.py --scenario base --config krishna.yml
