#!/bin/bash
#SBATCH  --output=output/logs/%j.out
#SBATCH  --account=dl_jobs

# Load Conda

module add cuda/12.1

source /home/mcrespo/miniconda3/etc/profile.d/conda.sh
conda activate /home/mcrespo/miniconda3/envs/sel_py11
nvcc --version

# python selora_finetuning
python selora_finetuning.py