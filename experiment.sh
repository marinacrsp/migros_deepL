#!/bin/bash
#SBATCH  --output=output/logs/%j.out
#SBATCH  --ntasks=1
#SBATCH  --mem-per-cpu=8G
#SBATCH  --cpus-per-task=1
#SBATCH  --gpus-per-node=1

# Load Conda
source /cluster/home/mcrespo/project_deepL/selora_env/bin/activate

# python selora_finetuning
python selora_finetuning