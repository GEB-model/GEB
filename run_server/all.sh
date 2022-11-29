#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=all
#SBATCH --output=all.out
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=12G
#SBATCH --time=100:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jens.de.bruijn@vu.nl
#SBATCH --exclusive

source ~/.bashrc

SCRIPT_DIR="/scistor/ivm/jbn271/GEB/GEB_private/"
cd $SCRIPT_DIR

module load cuda10.2/toolkit/10.2.89  # load cuda environment
conda activate abm  # activate conda environment

for scenario in base self_investment ngo_training government_subsidies;
do
    echo "running $scenario"
    srun --export=ALL -N1 -n1 --gres=gpu:1 --exclusive python run.py --GPU --headless --scenario $scenario &
done
wait