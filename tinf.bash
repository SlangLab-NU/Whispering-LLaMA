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
export CUDA_LAUNCH_BLOCKING=1



# model_id='large-v2' && option='M' && speaker_id='M03' && dataset_name="torgo_${speaker_id}_${model_id}_phoneme" && \
#     python Inference/WL-S_M_inference.py \
#     --option $option \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data_path "Inference/gs_inferences/${model_id}_hypo/torgo_${speaker_id}_${model_id}_test.pt" \
#     --save_dir "runs/Inference/${speaker_id}" \
#     --root "runs/WL_${option}_0.001_${dataset_name}" 


# 43281918
model_id='large-v2' && option='M' && speaker_id='M04' && dataset_name="torgo_${speaker_id}_${model_id}_phoneme" && \
    python Inference/WL-S_M_inference.py \
    --option $option \
    --pretrained_path 'weights/alpaca.pth' \
    --tokenizer_path 'weights/tokenizer.model' \
    --data_path "Inference/gs_inferences/${model_id}_hypo/torgo_${speaker_id}_${model_id}_test.pt" \
    --save_dir "runs/Inference/${speaker_id}" \
    --root "runs/WL_${option}_0.001_${dataset_name}" 


model_id='large-v2' && option='M' && speaker_id='M05' && dataset_name="torgo_${speaker_id}_${model_id}_phoneme" && \
    python Inference/WL-S_M_inference.py \
    --option $option \
    --pretrained_path 'weights/alpaca.pth' \
    --tokenizer_path 'weights/tokenizer.model' \
    --data_path "Inference/gs_inferences/${model_id}_hypo/torgo_${speaker_id}_${model_id}_test.pt" \
    --save_dir "runs/Inference/${speaker_id}" \
    --root "runs/WL_${option}_0.001_${dataset_name}" 


# 43158210
# python 4_2_whisper_inf_baseline-TORGO.py --model_name "large-v2"


# analyze M03 43079353
# python 4_1_whisper_inf_baseline.py --speaker_id M03 --model_name "large-v2"

# 43119402
# python 4_1_whisper_inf_baseline.py --speaker_id M01 --model_name "large-v2"
# python 4_1_whisper_inf_baseline.py --speaker_id M02 --model_name "large-v2"

# python 4_1_whisper_inf_baseline.py --speaker_id M04 --model_name "large-v2"
# python 4_1_whisper_inf_baseline.py --speaker_id M05 --model_name "large-v2"

# 43119431
# python 4_1_whisper_inf_baseline.py --speaker_id F01 --model_name "large-v2"
# python 4_1_whisper_inf_baseline.py --speaker_id F03 --model_name "large-v2"
# python 4_1_whisper_inf_baseline.py --speaker_id F04 --model_name "large-v2"


# python 2_finetune_whisper_on_torgo_frozen_encoder --speaker_id M03 --model_name "whisper-large-v2"


# model_id='large-v2' && option='M' && speaker_id='F01' && dataset_name="torgo_${speaker_id}_${model_id}" && \
#     python Inference/WL-S_M_inference.py \
#     --option $option \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data_path "Inference/gs_inferences/${model_id}_hypo/torgo_${speaker_id}_${model_id}_test.pt" \
#     --save_dir "runs/Inference/${speaker_id}" \
#     --root "runs/WL_${option}_0.001_${dataset_name}" 


# model_id='large-v2' && option='M' && speaker_id='F03' && dataset_name="torgo_${speaker_id}_${model_id}" && \
#     python Inference/WL-S_M_inference.py \
#     --option $option \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data_path "Inference/gs_inferences/${model_id}_hypo/torgo_${speaker_id}_${model_id}_test.pt" \
#     --save_dir "runs/Inference/${speaker_id}" \
#     --root "runs/WL_${option}_0.001_${dataset_name}" 


# model_id='large-v2' && option='M' && speaker_id='F04' && dataset_name="torgo_${speaker_id}_${model_id}" && \
#     python Inference/WL-S_M_inference.py \
#     --option $option \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data_path "Inference/gs_inferences/${model_id}_hypo/torgo_${speaker_id}_${model_id}_test.pt" \
#     --save_dir "runs/Inference/${speaker_id}" \
#     --root "runs/WL_${option}_0.001_${dataset_name}" 



# model_id='large-v2' && option='M' && speaker_id='M04' && dataset_name="torgo_${speaker_id}_${model_id}" && \
#     python Inference/WL-S_M_inference.py \
#     --option $option \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data_path "Inference/gs_inferences/${model_id}_hypo/torgo_${speaker_id}_${model_id}_test.pt" \
#     --save_dir "runs/Inference/${speaker_id}" \
#     --root "runs/WL_${option}_0.001_${dataset_name}" 


# model_id='large-v2' && option='M' && speaker_id='M05' && dataset_name="torgo_${speaker_id}_${model_id}" && \
#     python Inference/WL-S_M_inference.py \
#     --option $option \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data_path "Inference/gs_inferences/${model_id}_hypo/torgo_${speaker_id}_${model_id}_test.pt" \
#     --save_dir "runs/Inference/${speaker_id}" \
#     --root "runs/WL_${option}_0.001_${dataset_name}" 




# 43103724
# model_id='large-v2' && option='S' && speaker_id='M02' && dataset_name="torgo_${speaker_id}_${model_id}" && \
#     python Inference/WL-S_M_inference.py \
#     --option $option \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data_path "Inference/gs_inferences/${model_id}_hypo/torgo_${speaker_id}_${model_id}_test.pt" \
#     --save_dir "runs/Inference/${speaker_id}" \
#     --root "runs/WL_${option}_0.001_${dataset_name}" 



# model_id='large-v2' && option='S' && speaker_id='M03' && dataset_name="torgo_${speaker_id}_${model_id}" && \
#     python Inference/WL-S_M_inference.py \
#     --option $option \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data_path "Inference/gs_inferences/${model_id}_hypo/torgo_${speaker_id}_${model_id}_test.pt" \
#     --save_dir "runs/Inference/${speaker_id}" \
#     --root "runs/WL_${option}_0.001_${dataset_name}" 


# model_id='large-v2' && option='S' && speaker_id='M03' && dataset_name="torgo_${speaker_id}_${model_id}" && \
#     python Inference/WL-S_M_inference.py \
#     --option $option \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data_path "Inference/gs_inferences/${model_id}_hypo/torgo_${speaker_id}_${model_id}_test.pt" \
#     --save_dir "runs/Inference/${speaker_id}" \
#     --root "runs/WL_${option}_0.001_${dataset_name}" 


# 43097411
# model_id='large-v2' && option='S' && speaker_id='M02' && dataset_name="torgo_${speaker_id}_${model_id}" && \
#     python Inference/WL-S_M_inference.py \
#     --option $option \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data_path "Inference/gs_inferences/${model_id}_hypo/torgo_${speaker_id}_${model_id}_test.pt" \
#     --save_dir "runs/Inference/${speaker_id}" \
#     --root "runs/WL_${option}_0.001_${dataset_name}" 



# model_id='large-v2' && option='S' && speaker_id='M03' && dataset_name="torgo_${speaker_id}_${model_id}" && \
#     python Inference/WL-S_M_inference.py \
#     --option $option \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data_path "Inference/gs_inferences/${model_id}_hypo/torgo_${speaker_id}_${model_id}_test.pt" \
#     --save_dir "runs/Inference/${speaker_id}" \
#     --root "runs/WL_${option}_0.001_${dataset_name}" 





# model_id='large-v2' && speaker_id='F03' && dataset_name="torgo_${speaker_id}_${model_id}" && python Inference/WL-S_M_inference.py \
#     --option 'S' \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data "Inference/gs_inferences/${model_id}_hypo/torgo_${speaker_id}_${model_id}_test.pt" \
#     --save_dir "runs/Inference" \
#     --root "runs/WL_S_0.001_${dataset_name}" 





# speaker_id='M03' && option='S' && dataset_name="torgo_${speaker_id}" && \
#     python Inference/WL-S_M_inference.py \
#     --option $option \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data "Inference/gs_inferences/tiny_hypo/torgo_${speaker_id}_test.pt" \
#     --save_dir "runs/Inference/${speaker_id}" \
#     --root "runs/WL_${option}_0.001_${dataset_name}" 



 
# speaker_id='M03' && dataset_name="torgo_${speaker_id}" && python Inference/WL-S_inference.py \
#     --option 'S' \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data "Inference/gs_inferences/baseline_data_tiny_hypo_v2/torgo_${speaker_id}_test.pt" \
#     --save_dir "runs/Inference" \
#     --root "runs/WL_S_0.001_${dataset_name}" 


# 43072614
# speaker_id='M01' && dataset_name="torgo_${speaker_id}_large-v2" && python Inference/WL-M_inference.py \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data "Inference/gs_inferences/torgo_${speaker_id}_large-v2_test.pt" \
#     --save_dir "runs/Inference" \
#     --root "runs/WL_M_0.001_${dataset_name}" 


# speaker_id='M02' && dataset_name="torgo_${speaker_id}_large-v2" && python Inference/WL-M_inference.py \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data "Inference/gs_inferences/torgo_${speaker_id}_large-v2_test.pt" \
#     --save_dir "runs/Inference" \
#     --root "runs/WL_M_0.001_${dataset_name}" 




# python 01_output_feature_pt.py --speaker_id "F01"

# 43003263
# python 0_prepare_torgo_json.py --model_name "large-v2" --speaker_id "M03" --best_of 50
# python 01_output_feature_pt.py  --model_name "large_v2" --speaker_id "M03"

# python 0_prepare_torgo_json.py --model_name "large-v2" --speaker_id "M04" --best_of 50
# python 01_output_feature_pt.py  --model_name "large_v2" --speaker_id "M04"



# python 0_prepare_torgo_json.py --model_name "large-v2" --speaker_id "M05" --best_of 50
# python 01_output_feature_pt.py  --model_name "large_v2" --speaker_id "M05"

# python 0_prepare_torgo_json.py --model_name "large-v2" --speaker_id "M01" --best_of 50
# python 01_output_feature_pt.py  --model_name "large_v2" --speaker_id "M01"





# 43055755
# python 0_prepare_torgo_json.py --model_name "large-v2" --speaker_id "M02" --best_of 50
# python 01_output_feature_pt.py  --model_name "large_v2" --speaker_id "M02"

# 43064594
# python 0_prepare_torgo_json.py --model_name "large-v2" --speaker_id "F01" --best_of 50
# python 01_output_feature_pt.py --speaker_id "F01"



# 43055756
# python 0_prepare_torgo_json.py --model_name "large-v2" --speaker_id "F03"  --best_of 50
# python 01_output_feature_pt.py  --model_name "large_v2" --speaker_id "F03"

# 43055757
# python 0_prepare_torgo_json.py --model_name "large-v2" --speaker_id "F04"  --best_of 50
# python 01_output_feature_pt.py  --model_name "large_v2" --speaker_id "F04"


 






# 42951175
# python 2_finetune_whisper_on_torgo.py --speaker_id M03
# 42950955
# python 2_finetune_whisper_on_torgo.py --speaker_id M04

# 42951173
# python 2_finetune_whisper_on_torgo.py --speaker_id M05

# 42951178
# python 2_finetune_whisper_on_torgo.py --speaker_id M02



# 42966792
# python 2_finetune_whisper_on_torgo.py --speaker_id F01
# 42966794
# python 2_finetune_whisper_on_torgo.py --speaker_id F03
# 42966797
# python 2_finetune_whisper_on_torgo.py --speaker_id F04
# 42966799
# python 2_finetune_whisper_on_torgo.py --speaker_id M01





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





# python 1_finetuned_gen_json_torch_pt_train.py --speaker_id M02


# 42935867
# python 2_whisper_tiny_baseline.py --speaker_id F01
# python 2_whisper_tiny_baseline.py --speaker_id F03
# python 2_whisper_tiny_baseline.py --speaker_id F04
# python 2_whisper_tiny_baseline.py --speaker_id M02
# python 2_whisper_tiny_baseline.py --speaker_id M03
# python 2_whisper_tiny_baseline.py --speaker_id M04
# python 2_whisper_tiny_baseline.py --speaker_id M05


# 42938168
# python 2_whisper_tiny_baseline.py --speaker_id F01 --model_name "large-v2"
# python 2_whisper_tiny_baseline.py --speaker_id F03 --model_name "large-v2"
# python 2_whisper_tiny_baseline.py --speaker_id F04 --model_name "large-v2"
# python 2_whisper_tiny_baseline.py --speaker_id M02 --model_name "large-v2"
# python 2_whisper_tiny_baseline.py --speaker_id M03 --model_name "large-v2"
# python 2_whisper_tiny_baseline.py --speaker_id M04 --model_name "large-v2"
# python 2_whisper_tiny_baseline.py --speaker_id M05 --model_name "large-v2"