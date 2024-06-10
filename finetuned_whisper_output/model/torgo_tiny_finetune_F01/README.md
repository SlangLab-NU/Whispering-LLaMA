---
license: apache-2.0
base_model: openai/whisper-tiny
tags:
- generated_from_trainer
metrics:
- wer
model-index:
- name: torgo_tiny_finetune_F01
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# torgo_tiny_finetune_F01

This model is a fine-tuned version of [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 0.2909
- Wer: 24.6180

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 16
- eval_batch_size: 1
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 1000
- num_epochs: 20

### Training results

| Training Loss | Epoch | Step  | Validation Loss | Wer     |
|:-------------:|:-----:|:-----:|:---------------:|:-------:|
| 0.6321        | 0.83  | 500   | 0.3242          | 51.0187 |
| 0.1046        | 1.66  | 1000  | 0.4511          | 55.3480 |
| 0.099         | 2.49  | 1500  | 0.3225          | 31.5789 |
| 0.0656        | 3.32  | 2000  | 0.3007          | 50.6791 |
| 0.0506        | 4.15  | 2500  | 0.2984          | 27.6740 |
| 0.0383        | 4.98  | 3000  | 0.2853          | 23.6842 |
| 0.0296        | 5.8   | 3500  | 0.3449          | 32.3430 |
| 0.0198        | 6.63  | 4000  | 0.2730          | 26.6553 |
| 0.0192        | 7.46  | 4500  | 0.3049          | 49.2360 |
| 0.0136        | 8.29  | 5000  | 0.3279          | 25.8065 |
| 0.0121        | 9.12  | 5500  | 0.3082          | 23.8540 |
| 0.0101        | 9.95  | 6000  | 0.2722          | 25.5518 |
| 0.0065        | 10.78 | 6500  | 0.3414          | 32.0883 |
| 0.0062        | 11.61 | 7000  | 0.3140          | 22.9202 |
| 0.0053        | 12.44 | 7500  | 0.2601          | 24.7029 |
| 0.002         | 13.27 | 8000  | 0.2978          | 33.8710 |
| 0.0021        | 14.1  | 8500  | 0.2798          | 31.1545 |
| 0.0011        | 14.93 | 9000  | 0.3137          | 25.1273 |
| 0.0006        | 15.75 | 9500  | 0.2926          | 22.2411 |
| 0.0003        | 16.58 | 10000 | 0.2891          | 23.4295 |
| 0.0001        | 17.41 | 10500 | 0.2930          | 25.2122 |
| 0.0001        | 18.24 | 11000 | 0.2906          | 24.7878 |
| 0.0001        | 19.07 | 11500 | 0.2906          | 24.6180 |
| 0.0           | 19.9  | 12000 | 0.2909          | 24.6180 |


### Framework versions

- Transformers 4.32.0
- Pytorch 2.1.0+cu121
- Datasets 2.14.7
- Tokenizers 0.13.3
