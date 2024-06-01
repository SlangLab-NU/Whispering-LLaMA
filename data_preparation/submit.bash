#!/bin/bash -l
#SBATCH -N 1
#SBATCH -c 3
#SBATCH -p gpu

#SBATCH --gres=gpu:v100-sxm2:1   # --gres=gpu:t4:1
#SBATCH --time=08:00:00
#SBATCH --output=%j.output
#SBATCH --error=%j.error

module load anaconda3/2022.05

source activate /work/van-speech-nlp/jindaznb/mmenv/

