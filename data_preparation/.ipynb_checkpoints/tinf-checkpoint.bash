#!/bin/bash -l
#SBATCH -N 1
#SBATCH -c 3
#SBATCH -p gpu
#SBATCH --gres=gpu:v100-sxm2:1   # --gres=gpu:t4:1
#SBATCH --time=08:00:00
#SBATCH --output=log/%j.output
#SBATCH --error=log/%j.error

module load anaconda3/2022.05
module load ffmpeg/20190305 

source activate /work/van-speech-nlp/jindaznb/mmenv/
cd ..

# 42730670
# python 2_finetune_whisper_on_torgo.py --speaker_id F01


# finetuning with TORGO datasset 

# python 3_prepare_for_torgo_baseline.py --speaker_id M03
# 42813082
# python 3_prepare_for_torgo_baseline.py --speaker_id F03
# python 3_prepare_for_torgo_baseline.py --speaker_id F04
# python 3_prepare_for_torgo_baseline.py --speaker_id M01

# 42813212
# python 3_prepare_for_torgo_baseline.py --speaker_id M02 --generate_json
# python 3_prepare_for_torgo_baseline.py --speaker_id M04 --generate_json
# python 3_prepare_for_torgo_baseline.py --speaker_id M05



# general inference, without training on TORGO, but is on Gigaspeech
# 42755979
# cd ..
# speaker_id='F01' && python Inference/WL-S_inference.py \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data "Inference/gs_inferences/torgo_${speaker_id}_test.pt" \
#     --save_dir 'runs/Inference' \
#     --root 'runs/WL_S_0.001'


# 42852605
# speaker_id='M03' && python Inference/WL-S_inference.py \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data "Inference/gs_inferences/torgo_${speaker_id}_test.pt" \
#     --save_dir 'runs/Inference' \
#     --root 'runs/WL_S_0.001_giga17'

# 
speaker_id='F04' && python Inference/WL-S_inference.py \
    --pretrained_path 'weights/alpaca.pth' \
    --tokenizer_path 'weights/tokenizer.model' \
    --data "Inference/gs_inferences/torgo_${speaker_id}_test.pt" \
    --save_dir "runs/Inference/${speaker_id}" \
    --root 'runs/WL_S_0.001_giga17'

# 42783602
# finetuning on TORGO
# cd .. && dataset_name='torgo' && speaker_id='F01' && python Inference/WL-S_inference.py \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data "Inference/gs_inferences/torgo_${speaker_id}_test.pt" \
#     --save_dir "runs/Inference" \
#     --root "runs/WL_S_0.001_${dataset_name}" 