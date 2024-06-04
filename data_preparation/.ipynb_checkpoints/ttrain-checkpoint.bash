#!/bin/bash -l
#SBATCH -N 1
#SBATCH -c 3
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1   # --gres=gpu:t4:1
#SBATCH --constraint=a100@80g  # Constraint for A100 GPU with 80GB memory
#SBATCH --time=08:00:00
#SBATCH --output=log/%j.output
#SBATCH --error=log/%j.error

module load anaconda3/2022.05

source activate /work/van-speech-nlp/jindaznb/mmenv/

# 42763640
cd ..
speaker_id='F01' && python training/WL-S-Copy1.py --lr 1e-3 \
    --d 1 \
    --pretrained_path 'weights/alpaca.pth' \
    --tokenizer_path 'weights/tokenizer.model' \
    --data "Inference/gs_inferences/torgo_${speaker_id}"
    --dataset_name "torgo"