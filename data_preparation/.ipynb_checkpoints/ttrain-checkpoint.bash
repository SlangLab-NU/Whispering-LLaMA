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
module load ffmpeg/20190305 

source activate /work/van-speech-nlp/jindaznb/mmenv/
cd ..


# train adpater with giga_17 dataset
# 42827940
# python training/WL-S_train-Copy1.py --lr 1e-3 \
#     --d 1 \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data_path "Inference/gs_inferences/17" \
#     --dataset_name "giga17"

# 42832434
# speaker_id='F01' && python training/WL-S_train-Copy1.py --lr 1e-3 \
#     --d 1 \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data_path "Inference/gs_inferences/torgo_${speaker_id}" \ 
#     --dataset_name "torgo_F01"


# 42835408
# python 4_output_feature_pt.py --speaker_id M03
# speaker_id='M03' && python training/WL-S_train-Copy1.py --lr 1e-3 \
#     --d 1 \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data_path "Inference/gs_inferences/torgo_${speaker_id}" --dataset_name "torgo_${speaker_id}"


# 42839821
# python 4_output_feature_pt.py --speaker_id M03
# python 4_output_feature_pt.py --speaker_id F03
# python 4_output_feature_pt.py --speaker_id F04
# python 4_output_feature_pt.py --speaker_id M01
# python 4_output_feature_pt.py --speaker_id M02
# python 4_output_feature_pt.py --speaker_id M04
# python 4_output_feature_pt.py --speaker_id M05