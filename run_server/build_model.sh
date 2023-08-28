#!/bin/bash
#SBATCH --job-name=build_model
#SBATCH --output=build_model.out
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --time=100:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jens.de.bruijn@vu.nl

source ~/.bashrc

SCRIPT_DIR="$HOME/GEB/GEB_private/"
cd $SCRIPT_DIR

conda activate GEB  # activate conda environment

python build_model.py --config $1 && \
python preprocessing/krishna/2_create_tehsil_data_table.py --config $1 && \
python "preprocessing/krishna/3_parse_IHDS I.py" --config $1 && \
python preprocessing/krishna/4_iterative_proportional_fitting.py --config $1 && \
python preprocessing/krishna/5_create_farmers.py --config $1 && \
python build_model_farmers.py --config $1
