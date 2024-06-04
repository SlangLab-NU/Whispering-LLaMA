#!/bin/bash -l
#SBATCH -N 1
#SBATCH -c 3
#SBATCH -p gpu
#SBATCH --gres=gpu:v100-sxm2:1   # --gres=gpu:t4:1
#SBATCH --time=08:00:00
#SBATCH --output=log/%j.output
#SBATCH --error=log/%j.error

module load anaconda3/2022.05

source activate /work/van-speech-nlp/jindaznb/mmenv/


# 42730670
# python 2_finetune_whisper_on_torgo.py --speaker_id F01


# finetuning with TORGO datasset
# cd ..
# python 3_prepare_for_torgo_baseline.py --speaker_id F01



# general inference, without training on TORGO, but is on Gigaspeech
# 42755979
# cd ..
# speaker_id='F01' && python Inference/WL-S_inference.py \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data "Inference/gs_inferences/torgo_${speaker_id}_test.pt" \
#     --save_dir 'runs/Inference' \
#     --root 'runs/WL_S_0.001'



# finetuning on TORGO
cd .. && dataset_name='torgo' && speaker_id='F01' && python Inference/WL-S_inference.py \
    --pretrained_path 'weights/alpaca.pth' \
    --tokenizer_path 'weights/tokenizer.model' \
    --data "Inference/gs_inferences/torgo_${speaker_id}_test.pt" \
    --save_dir "runs/Inference_${dataset_name}" \
    --root "runs/WL_S_0.001_${dataset_name}" 