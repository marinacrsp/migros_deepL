#!/bin/bash
#SBATCH --output=/work/scratch/mcrespo/output/%j.out
#SBATCH --account=dl
#SBATCH --partition=gpu
#SBATCH --nodes=1

# Load Conda

module add cuda/12.1

source /home/mcrespo/miniconda3/etc/profile.d/conda.sh
conda activate /home/mcrespo/miniconda3/envs/sel_py11
nvcc --version

# python selora_training.py
# python selora_finetuning_luca.py
python classifier_train.py
# python compute_fid.py