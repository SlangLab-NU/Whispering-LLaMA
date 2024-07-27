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

source activate /work/van-speech-nlp/jindaznb/visenv/
cd /work/van-speech-nlp/jindaznb/jslpnb/mllm_expriments/Whispering-LLaMA


# 43468159
# speaker_id='M05'
# option='M'
# datafolder='n_best_phoneme_comma'
# base_path='/work/van-speech-nlp/jindaznb/jslpnb/mllm_expriments/Whispering-LLaMA'
# python training/WL-S_M_train.py --lr 1e-3 \
#     --option $option \
#     --d 1 \
#     --pretrained_path "${base_path}/weights/alpaca.pth" \
#     --tokenizer_path "${base_path}/weights/tokenizer.model" \
#     --data_path "${base_path}/Inference/gs_inferences/${datafolder}/torgo_${speaker_id}_large-v2" \
#     --dataset_name "torgo_${speaker_id}_${datafolder}" \
#     --num_epochs 10 \
#     --input_batch_size 32 \
#     --adapter_path "${base_path}/runs/WL_${option}_0.001_torgo_${speaker_id}_${datafolder}"



# speaker_id='M04'
# option='M'
# datafolder='n_best_phoneme_comma'
# base_path='/work/van-speech-nlp/jindaznb/jslpnb/mllm_expriments/Whispering-LLaMA'
# python training/WL-S_M_train.py --lr 1e-3 \
#     --option $option \
#     --d 1 \
#     --pretrained_path "${base_path}/weights/alpaca.pth" \
#     --tokenizer_path "${base_path}/weights/tokenizer.model" \
#     --data_path "${base_path}/Inference/gs_inferences/${datafolder}/torgo_${speaker_id}_large-v2" \
#     --dataset_name "torgo_${speaker_id}_${datafolder}" \
#     --num_epochs 10 \
#     --input_batch_size 32 \
#     --adapter_path "${base_path}/runs/WL_${option}_0.001_torgo_${speaker_id}_${datafolder}"






# 43446247
# speaker_id='M04'
# option='M'
# datafolder='large-v2_hypo'
# base_path='/work/van-speech-nlp/jindaznb/jslpnb/mllm_expriments/Whispering-LLaMA'
# python training/WL-S_M_train.py --lr 1e-3 \
#     --option $option \
#     --d 1 \
#     --pretrained_path "${base_path}/weights/alpaca.pth" \
#     --tokenizer_path "${base_path}/weights/tokenizer.model" \
#     --data_path "${base_path}/Inference/gs_inferences/${datafolder}/torgo_${speaker_id}_large-v2" \
#     --dataset_name "torgo_${speaker_id}_${datafolder}" \
#     --num_epochs 10 \
#     --input_batch_size 32

    
# 43446236
# speaker_id='M04'
# option='M'
# datafolder='n_best_phoneme_comma'
# base_path='/work/van-speech-nlp/jindaznb/jslpnb/mllm_expriments/Whispering-LLaMA'
# python training/WL-S_M_train.py --lr 1e-3 \
#     --option $option \
#     --d 1 \
#     --pretrained_path "${base_path}/weights/alpaca.pth" \
#     --tokenizer_path "${base_path}/weights/tokenizer.model" \
#     --data_path "${base_path}/Inference/gs_inferences/${datafolder}/torgo_${speaker_id}_large-v2" \
#     --dataset_name "torgo_${speaker_id}_${datafolder}" \
#     --num_epochs 10 \
#     --input_batch_size 32
    


# 43448382
speaker_id='M05'
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
    --input_batch_size 32





# 43358034
# speaker_id='M04' && option='M' && datafolder='nbest_phoneme' \
#     && base_path='/work/van-speech-nlp/jindaznb/jslpnb/mllm_expriments/Whispering-LLaMA' \
#     && python training/WL-S_M_train.py --lr 1e-3 \
#     --option $option \
#     --d 1 \
#     --pretrained_path "${base_path}/weights/alpaca.pth" \
#     --tokenizer_path "${base_path}/weights/tokenizer.model" \
#     --data_path "${base_path}/Inference/gs_inferences/${datafolder}/torgo_${speaker_id}_large-v2_n_best_phoneme" \
#     --dataset_name "torgo_${speaker_id}_large-v2_phoneme_nbest" \
#     --num_epochs 10 \
#     --input_batch_size 32


# 43358035
# speaker_id='M04' && option='M'  \
#     && base_path='/work/van-speech-nlp/jindaznb/jslpnb/mllm_expriments/Whispering-LLaMA' \
#     && python training/WL-S_M_train.py --lr 1e-3 \
#     --option $option \
#     --d 1 \
#     --pretrained_path "${base_path}/weights/alpaca.pth" \
#     --tokenizer_path "${base_path}/weights/tokenizer.model" \
#     --data_path "${base_path}/Inference/gs_inferences/single_phoneme/torgo_${speaker_id}_large-v2_phoneme" \
#     --dataset_name "torgo_${speaker_id}_large-v2_phoneme" \
#     --num_epochs 10 \
#     --input_batch_size 32



# 43358069
# speaker_id='M04' && option='M'  \
#     && base_path='/work/van-speech-nlp/jindaznb/jslpnb/mllm_expriments/Whispering-LLaMA' \
#     && python training/WL-S_M_train.py --lr 1e-3 \
#     --option $option \
#     --d 1 \
#     --pretrained_path "${base_path}/weights/alpaca.pth" \
#     --tokenizer_path "${base_path}/weights/tokenizer.model" \
#     --data_path "${base_path}/Inference/gs_inferences/large-v2_hypo/torgo_${speaker_id}_large-v2" \
#     --dataset_name "torgo_${speaker_id}_large-v2" \
#     --num_epochs 10








# 
# speaker_id='M04' && option='M'  \
#     && base_path='/work/van-speech-nlp/jindaznb/jslpnb/mllm_expriments/Whispering-LLaMA' \
#     && python training/WL-S_M_train.py --lr 1e-3 \
#     --option $option \
#     --d 1 \
#     --pretrained_path "${base_path}/weights/alpaca.pth" \
#     --tokenizer_path "${base_path}/weights/tokenizer.model" \
#     --data_path "${base_path}/Inference/gs_inferences/single_phoneme/torgo_${speaker_id}_large-v2_phoneme" \
#     --adapter_path "${base_path}/runs/WL_${option}_0.001_torgo_${speaker_id}_large-v2_phoneme" \
#     --dataset_name "torgo_${speaker_id}_large-v2_phoneme" \
#     --num_epochs 20 \
#     --input_batch_size 32

# 43281905
# speaker_id='M03' && option='M'  \
#     && base_path='/work/van-speech-nlp/jindaznb/jslpnb/mllm_expriments/Whispering-LLaMA' \
#     && python training/WL-S_M_train.py --lr 1e-3 \
#     --option $option \
#     --d 1 \
#     --pretrained_path "${base_path}/weights/alpaca.pth" \
#     --tokenizer_path "${base_path}/weights/tokenizer.model" \
#     --data_path "${base_path}/Inference/gs_inferences/single_phoneme/torgo_${speaker_id}_large-v2_phoneme" \
#     --adapter_path "${base_path}/runs/WL_${option}_0.001_torgo_${speaker_id}_large-v2_phoneme" \
#     --dataset_name "torgo_${speaker_id}_large-v2_phoneme" \
#     --num_epochs 20 \
#     --input_batch_size 32





# 43163037
# speaker_id='F01' && option='M'  \
#     && base_path='/work/van-speech-nlp/jindaznb/jslpnb/mllm_expriments/Whispering-LLaMA' \
#     && python training/WL-S_M_train.py --lr 1e-3 \
#     --option $option \
#     --d 1 \
#     --pretrained_path "${base_path}/weights/alpaca.pth" \
#     --tokenizer_path "${base_path}/weights/tokenizer.model" \
#     --data_path "${base_path}/Inference/gs_inferences/large-v2_hypo/torgo_${speaker_id}_large-v2" \
#     --dataset_name "torgo_${speaker_id}_large-v2" \
#     --adapter_path "${base_path}/runs/WL_${option}_0.001_torgo_${speaker_id}_large-v2" \
#     --num_epochs 20


# 43163038
# speaker_id='F03' && option='M'  \
#     && base_path='/work/van-speech-nlp/jindaznb/jslpnb/mllm_expriments/Whispering-LLaMA' \
#     && python training/WL-S_M_train.py --lr 1e-3 \
#     --option $option \
#     --d 1 \
#     --pretrained_path "${base_path}/weights/alpaca.pth" \
#     --tokenizer_path "${base_path}/weights/tokenizer.model" \
#     --data_path "${base_path}/Inference/gs_inferences/large-v2_hypo/torgo_${speaker_id}_large-v2" \
#     --dataset_name "torgo_${speaker_id}_large-v2" \
#     --adapter_path "${base_path}/runs/WL_${option}_0.001_torgo_${speaker_id}_large-v2" \
#     --num_epochs 20

# 43163040
# speaker_id='F04' && option='M'  \
#     && base_path='/work/van-speech-nlp/jindaznb/jslpnb/mllm_expriments/Whispering-LLaMA' \
#     && python training/WL-S_M_train.py --lr 1e-3 \
#     --option $option \
#     --d 1 \
#     --pretrained_path "${base_path}/weights/alpaca.pth" \
#     --tokenizer_path "${base_path}/weights/tokenizer.model" \
#     --data_path "${base_path}/Inference/gs_inferences/large-v2_hypo/torgo_${speaker_id}_large-v2" \
#     --dataset_name "torgo_${speaker_id}_large-v2" \
#     --adapter_path "${base_path}/runs/WL_${option}_0.001_torgo_${speaker_id}_large-v2" \
#     --num_epochs 20




    

# 43111952
# speaker_id='F03' && python training/WL-S_M_train.py --lr 1e-3 \
#     --option 'M' \
#     --d 1 \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data_path "Inference/gs_inferences/large-v2_hypo/torgo_${speaker_id}_large-v2" --dataset_name "torgo_${speaker_id}_large-v2"



# 43110608
# speaker_id='F01' && python training/WL-S_M_train.py --lr 1e-3 \
#     --option 'M' \
#     --d 1 \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data_path "Inference/gs_inferences/large-v2_hypo/torgo_${speaker_id}_large-v2" --dataset_name "torgo_${speaker_id}_large-v2"

## 43110607
# speaker_id='M05' && python training/WL-S_M_train.py --lr 1e-3 \
#     --option 'M' \
#     --d 1 \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data_path "Inference/gs_inferences/large-v2_hypo/torgo_${speaker_id}_large-v2" --dataset_name "torgo_${speaker_id}_large-v2"



# speaker_id='M04' && python training/WL-S_M_train.py --lr 1e-3 \
#     --option 'M' \
#     --d 1 \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data_path "Inference/gs_inferences/large-v2_hypo/torgo_${speaker_id}_large-v2" --dataset_name "torgo_${speaker_id}_large-v2" 


    #\ --adapter_path "/work/van-speech-nlp/jindaznb/jslpnb/mllm_expriments/Whispering-LLaMA/runs/WL_M_0.001_torgo_M05_large-v2"






# 43097452
# speaker_id='M01' && python training/WL-S_M_train-Copy1.py --lr 1e-3 \
#     --option 'S' \
#     --d 1 \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data_path "Inference/gs_inferences/large-v2_hypo/torgo_${speaker_id}_large-v2" --dataset_name "torgo_${speaker_id}_large-v2"


# 43100449
# speaker_id='M02' && python training/WL-S_M_train-Copy1.py --lr 1e-3 \
#     --option 'S' \
#     --d 1 \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data_path "Inference/gs_inferences/large-v2_hypo/torgo_${speaker_id}_large-v2" --dataset_name "torgo_${speaker_id}_large-v2"


# 43100450
# speaker_id='M03' && python training/WL-S_M_train-Copy1.py --lr 1e-3 \
#     --option 'S' \
#     --d 1 \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data_path "Inference/gs_inferences/large-v2_hypo/torgo_${speaker_id}_large-v2" --dataset_name "torgo_${speaker_id}_large-v2"





    
# 43095348
# speaker_id='F01' && python training/WL-S_M_train-Copy1.py --lr 1e-3 \
#     --option 'M' \
#     --d 1 \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data_path "Inference/gs_inferences/large-v2_hypo/torgo_${speaker_id}_large-v2" --dataset_name "torgo_${speaker_id}_large-v2"

# S: 43094923





# 43077610
# speaker_id='M01' && python training/WL-S_M_train.py --lr 1e-3 \
#     --d 1 \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data_path "Inference/gs_inferences/torgo_${speaker_id}_large-v2" --dataset_name "torgo_${speaker_id}_large-v2"


# 43077614
# speaker_id='M02' && python training/WL-S_train-Copy1.py --lr 1e-3 \
#     --d 1 \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data_path "Inference/gs_inferences/torgo_${speaker_id}_large-v2" --dataset_name "torgo_${speaker_id}_large-v2"

# 43077615
# speaker_id='M03' && python training/WL-S_train-Copy1.py --lr 1e-3 \
#     --d 1 \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data_path "Inference/gs_inferences/torgo_${speaker_id}_large-v2" --dataset_name "torgo_${speaker_id}_large-v2"






# 43074028
# speaker_id='M04' && python training/WL-S_train-Copy1.py --lr 1e-3 \
#     --d 1 \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data_path "Inference/gs_inferences/torgo_${speaker_id}_large-v2" --dataset_name "torgo_${speaker_id}_large-v2"

# 43074027
# speaker_id='M05' && python training/WL-S_train-Copy1.py --lr 1e-3 \
#     --d 1 \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data_path "Inference/gs_inferences/torgo_${speaker_id}_large-v2" --dataset_name "torgo_${speaker_id}_large-v2"


# 43074029
# speaker_id='F04' && python training/WL-S_train-Copy1.py --lr 1e-3 \
#     --d 1 \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data_path "Inference/gs_inferences/torgo_${speaker_id}_large-v2" --dataset_name "torgo_${speaker_id}_large-v2"

#  43074030
# speaker_id='F03' && python training/WL-S_train-Copy1.py --lr 1e-3 \
#     --d 1 \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data_path "Inference/gs_inferences/torgo_${speaker_id}_large-v2" --dataset_name "torgo_${speaker_id}_large-v2"









# 43066918
# speaker_id='M01' && python training/WL-S_train-Copy1.py --lr 1e-3 \
#     --d 1 \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data_path "Inference/gs_inferences/torgo_${speaker_id}_large-v2" --dataset_name "torgo_${speaker_id}_large-v2"

# 43066919
# speaker_id='M02' && python training/WL-S_train-Copy1.py --lr 1e-3 \
#     --d 1 \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data_path "Inference/gs_inferences/torgo_${speaker_id}_large-v2" --dataset_name "torgo_${speaker_id}_large-v2"

# 43066920
# speaker_id='M03' && python training/WL-S_train-Copy1.py --lr 1e-3 \
#     --d 1 \
#     --pretrained_path 'weights/alpaca.pth' \
#     --tokenizer_path 'weights/tokenizer.model' \
#     --data_path "Inference/gs_inferences/torgo_${speaker_id}_large-v2" --dataset_name "torgo_${speaker_id}_large-v2"









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
