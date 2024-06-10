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
export CUDA_LAUNCH_BLOCKING=1


# 42906246
# python 2_finetune_whisper_on_torgo.py --speaker_id F03
# 42906251
# python 2_finetune_whisper_on_torgo.py --speaker_id F04

# 42906253
# python 2_finetune_whisper_on_torgo.py --speaker_id M05


cd ..


# finetuning with TORGO datasset 
# python 3_prepare_for_torgo_baseline.py --speaker_id M03
# 42813082
# python 3_prepare_for_torgo_baseline.py --speaker_id F03
# python 3_prepare_for_torgo_baseline.py --speaker_id F04
# python 3_prepare_for_torgo_baseline.py --speaker_id M01

# 42813212
# python 3_prepare_for_torgo_baseline.py --speaker_id M02 --generate_json
# python 3_prepare_for_torgo_baseline.py --speaker_id M04 --generate_json




# general inference, without training on TORGO, but is on Gigaspeech

# 42861388
# speaker_id='F04' && python Inference/WL-S_inference.py \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data "Inference/gs_inferences/torgo_${speaker_id}_test.pt" \
#     --save_dir "runs/Inference/${speaker_id}" \
#     --root 'runs/WL_S_0.001_giga17'

# speaker_id='M01' && python Inference/WL-S_inference.py \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data "Inference/gs_inferences/torgo_${speaker_id}_test.pt" \
#     --save_dir "runs/Inference/${speaker_id}" \
#     --root 'runs/WL_S_0.001_giga17'


# 42861389
# speaker_id='M02' && python Inference/WL-S_inference.py \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data "Inference/gs_inferences/torgo_${speaker_id}_test.pt" \
#     --save_dir "runs/Inference/${speaker_id}" \
#     --root 'runs/WL_S_0.001_giga17'

# speaker_id='M04' && python Inference/WL-S_inference.py \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data "Inference/gs_inferences/torgo_${speaker_id}_test.pt" \
#     --save_dir "runs/Inference/${speaker_id}" \
#     --root 'runs/WL_S_0.001_giga17'



# debug M05 running with whisper-en
# python 3_prepare_for_torgo_baseline.py --speaker_id M05
# python 4_output_feature_pt.py --speaker_id M05
# speaker_id='M05' && python Inference/WL-S_inference.py \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data "Inference/gs_inferences/torgo_${speaker_id}_test.pt" \
#     --save_dir "runs/Inference/${speaker_id}" \
#     --root 'runs/WL_S_0.001_giga17'



# finetuning on TORGO
# cd .. && dataset_name='torgo' && speaker_id='F01' && python Inference/WL-S_inference.py \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data "Inference/gs_inferences/torgo_${speaker_id}_test.pt" \
#     --save_dir "runs/Inference" \
#     --root "runs/WL_S_0.001_${dataset_name}" 


# speaker_id='F01' && python Inference/WL-S_inference.py \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data "Inference/gs_inferences/torgo_${speaker_id}_test.pt" \
#     --save_dir "runs/Inference/${speaker_id}" \
#     --root "runs/WL_S_0.001_torgo_${speaker_id}"



# 42906379
# speaker_id='M05' && python Inference/WL-S_inference.py \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data "Inference/gs_inferences/torgo_${speaker_id}_test.pt" \
#     --save_dir "runs/Inference/${speaker_id}" \
#     --root "runs/WL_S_0.001_torgo_${speaker_id}"





python 1_finetuned_gen_json_torch_pt_train.py --speaker_id M02