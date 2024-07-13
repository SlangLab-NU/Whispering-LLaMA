#!/bin/bash -l
#SBATCH -N 1
#SBATCH -c 12
#SBATCH -p gpu
#SBATCH --gres=gpu:v100-sxm2:1   # --gres=gpu:t4:1
#SBATCH --time=08:00:00
#SBATCH --output=log/%j.output
#SBATCH --error=log/%j.error

nvidia-smi
module load anaconda3/2022.05
module load ffmpeg/20190305 

source activate /work/van-speech-nlp/jindaznb/visenv/
cd /work/van-speech-nlp/jindaznb/jslpnb/mllm_expriments/Whispering-LLaMA


speaker_id='M02' && option='M'  \
    && base_path='/work/van-speech-nlp/jindaznb/jslpnb/mllm_expriments/Whispering-LLaMA' \
    && python -m debugpy --listen 5678 --wait-for-client training/WL-S_M_train.py --lr 1e-3 \
    --option $option \
    --d 1 \
    --pretrained_path "${base_path}/weights/alpaca.pth" \
    --tokenizer_path "${base_path}/weights/tokenizer.model" \
    --data_path "${base_path}/Inference/gs_inferences/single_phoneme/torgo_${speaker_id}_large-v2_phoneme" \
    --dataset_name "torgo_${speaker_id}_large-v2_phoneme" \
    --num_epochs 10