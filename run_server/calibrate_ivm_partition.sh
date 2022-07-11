#!/bin/bash
#SBATCH --partition=ivm
#SBATCH --job-name=ivm_calibrate
#SBATCH --output=calibrate_ivm.out
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=16G
#SBATCH --time=200:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jens.de.bruijn@vu.nl

source ~/.bashrc

SCRIPT_DIR="/scistor/ivm/jbn271/Packages/GEB/"
cd $SCRIPT_DIR

module load cuda10.2/toolkit/10.2.89  # load cuda environment

conda activate a  # activate conda environment

python calibration/calibrate.py --config calibration/config_ivm_partition.yml