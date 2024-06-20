#!/bin/bash -l
#SBATCH -N 1
#SBATCH -c 12
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1   
#SBATCH --constraint=a100@80g
#SBATCH --time=08:00:00
#SBATCH --output=log/%j.output
#SBATCH --error=log/%j.error

nvidia-smi
module load anaconda3/2022.05
module load ffmpeg/20190305 

source activate /work/van-speech-nlp/jindaznb/mmenv/
cd ..





# 
# python 1_finetuned_whisper_gen_json_torch_pt_train.py --speaker_id M02
# speaker_id='M02' && python training/WL-S_train-Copy1.py --lr 1e-3 \
#     --d 1 \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data_path "Inference/gs_inferences/finetuned_torgo_${speaker_id}" --dataset_name "torgo_${speaker_id}_finetuned_whisper"


# 42977426
# python 1_finetuned_whisper_gen_json_torch_pt_train.py --speaker_id M03
# speaker_id='M03' && python training/WL-S_train-Copy1.py --lr 1e-3 \
#     --d 1 \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data_path "Inference/gs_inferences/finetuned_torgo_${speaker_id}" --dataset_name "torgo_${speaker_id}_finetuned_whisper"

# 42981874
# python 1_finetuned_whisper_gen_json_torch_pt_train.py --speaker_id M04
# speaker_id='M04' && python training/WL-S_train-Copy1.py --lr 1e-3 \
#     --d 1 \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data_path "Inference/gs_inferences/finetuned_torgo_${speaker_id}" --dataset_name "torgo_${speaker_id}_finetuned_whisper"


# 42981906
# python 1_finetuned_whisper_gen_json_torch_pt_train.py --speaker_id M05 && speaker_id='M05' && python training/WL-S_train-Copy1.py --lr 1e-3 \
#     --d 1 \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data_path "Inference/gs_inferences/finetuned_torgo_${speaker_id}" --dataset_name "torgo_${speaker_id}_finetuned_whisper"


# train adpater with giga_17 dataset
# python training/WL-S_train-Copy1.py --lr 1e-3 \
#     --d 1 \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data_path "Inference/gs_inferences/17" \
#     --dataset_name "giga17"

# 42867478
# speaker_id='F01' && python training/WL-S_train-Copy1.py --lr 1e-3 \
#     --d 1 \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data_path "Inference/gs_inferences/torgo_${speaker_id}" --dataset_name "torgo_${speaker_id}"


# speaker_id='F03' && python training/WL-S_train-Copy1.py --lr 1e-3 \
#     --d 1 \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data_path "Inference/gs_inferences/torgo_${speaker_id}" --dataset_name "torgo_${speaker_id}"


# 42867479
# speaker_id='F04' && python training/WL-S_train-Copy1.py --lr 1e-3 \
#     --d 1 \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data_path "Inference/gs_inferences/torgo_${speaker_id}" --dataset_name "torgo_${speaker_id}"


# 42884604
# speaker_id='M01' && python training/WL-S_train-Copy1.py --lr 1e-3 \
#     --d 1 \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data_path "Inference/gs_inferences/torgo_${speaker_id}" --dataset_name "torgo_${speaker_id}"


# 42867483
# speaker_id='M02' && python training/WL-S_train-Copy1.py --lr 1e-3 \
#     --d 1 \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data_path "Inference/gs_inferences/torgo_${speaker_id}" --dataset_name "torgo_${speaker_id}"



# 42884605
# speaker_id='M03' && python training/WL-S_train-Copy1.py --lr 1e-3 \
#     --d 1 \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data_path "Inference/gs_inferences/torgo_${speaker_id}" --dataset_name "torgo_${speaker_id}"



# 42867486
# speaker_id='M04' && python training/WL-S_train-Copy1.py --lr 1e-3 \
#     --d 1 \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data_path "Inference/gs_inferences/torgo_${speaker_id}" --dataset_name "torgo_${speaker_id}"


# 42896850
# speaker_id='M05' && python training/WL-S_train-Copy1.py --lr 1e-3 \
#     --d 1 \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data_path "Inference/gs_inferences/torgo_${speaker_id}" --dataset_name "torgo_${speaker_id}"



# 42892049
# debug M05 running with whisper-en
# python 3_prepare_for_torgo_baseline.py --speaker_id M05
# python 4_output_feature_pt.py --speaker_id M05
# speaker_id='M05' && python Inference/WL-S_inference.py \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data "Inference/gs_inferences/torgo_${speaker_id}_test.pt" \
#     --save_dir "runs/Inference/${speaker_id}" \
#     --root 'runs/WL_S_0.001_giga17'


# python 3_hypothesis_whisper_finetuned.py --speaker_id M01 --model_name "finetuned_whisper_output/model/torgo_tiny_finetune_F01"
