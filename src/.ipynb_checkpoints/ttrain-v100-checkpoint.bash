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
export CUDA_LAUNCH_BLOCKING=1



# 43468159
speaker_id='M04'
option='M'
datafolder='n_best_phoneme_comma'
base_path='/work/van-speech-nlp/jindaznb/jslpnb/mllm_expriments/Whispering-LLaMA'
python training/WL-S_M_train.py --lr 1e-3 \
    --option $option \
    --d 1 \
    --pretrained_path "${base_path}/weights/alpaca.pth" \
    --tokenizer_path "${base_path}/weights/tokenizer.model" \
    --data_path "${base_path}/Inference/gs_inferences/${datafolder}/torgo_${speaker_id}_large-v2" \
    --dataset_name "torgo_${speaker_id}_${datafolder}" \
    --num_epochs 10 \
    --input_batch_size 32 \
    --micro_batch_size 1 \
    --adapter_path "${base_path}/runs/WL_${option}_0.001_torgo_${speaker_id}_${datafolder}"

# 43468038
speaker_id='M05'
option='M'
datafolder='large-v2_hypo'
base_path='/work/van-speech-nlp/jindaznb/jslpnb/mllm_expriments/Whispering-LLaMA'
python training/WL-S_M_train.py --lr 1e-3 \
    --option $option \
    --d 1 \
    --pretrained_path "${base_path}/weights/alpaca.pth" \
    --tokenizer_path "${base_path}/weights/tokenizer.model" \
    --data_path "${base_path}/Inference/gs_inferences/${datafolder}/torgo_${speaker_id}_large-v2" \
    --dataset_name "torgo_${speaker_id}_${datafolder}" \
    --num_epochs 10 \
    --input_batch_size 32 \
    --adapter_path "${base_path}/runs/WL_${option}_0.001_torgo_${speaker_id}_${datafolder}"