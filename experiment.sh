#!/bin/bash
#SBATCH  --output=/work/scratch/mcrespo/output/%j.out
#SBATCH  --account=dl_jobs

# Load Conda

module add cuda/12.1

source /home/mcrespo/miniconda3/etc/profile.d/conda.sh
conda activate /home/mcrespo/miniconda3/envs/sel_py11
nvcc --version

# python test_loras.py
python train_lora.py