#!/bin/bash -l
#SBATCH -N 1
#SBATCH -c 3
#SBATCH -p gpu
#SBATCH --gres=gpu:1   # --gres=gpu:t4:1
#SBATCH --time=08:00:00
#SBATCH --output=log/%j.output
#SBATCH --error=log/%j.error

module load anaconda3/2022.05

source activate /work/van-speech-nlp/jindaznb/mmenv/

# 42795999
cd ..
python 3_prepare_for_torgo_baseline.py --speaker_id M03