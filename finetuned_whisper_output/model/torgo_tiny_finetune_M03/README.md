---
license: apache-2.0
base_model: openai/whisper-tiny
tags:
- generated_from_trainer
metrics:
- wer
model-index:
- name: torgo_tiny_finetune_M03
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# torgo_tiny_finetune_M03

This model is a fine-tuned version of [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 0.3315
- Wer: 28.3531

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
| 0.6365        | 0.85  | 500   | 0.3228          | 25.3820 |
| 0.1068        | 1.71  | 1000  | 0.3417          | 52.2920 |
| 0.1009        | 2.56  | 1500  | 0.3318          | 48.2173 |
| 0.0686        | 3.41  | 2000  | 0.2947          | 30.3056 |
| 0.0516        | 4.27  | 2500  | 0.3396          | 26.0611 |
| 0.0353        | 5.12  | 3000  | 0.3153          | 28.4380 |
| 0.0255        | 5.97  | 3500  | 0.2689          | 24.7878 |
| 0.0207        | 6.83  | 4000  | 0.4144          | 35.1443 |
| 0.0148        | 7.68  | 4500  | 0.2552          | 31.8336 |
| 0.015         | 8.53  | 5000  | 0.3356          | 32.5127 |
| 0.0114        | 9.39  | 5500  | 0.3311          | 32.9372 |
| 0.0098        | 10.24 | 6000  | 0.3318          | 24.4482 |
| 0.0067        | 11.09 | 6500  | 0.2942          | 29.5416 |
| 0.0031        | 11.95 | 7000  | 0.3945          | 60.8659 |
| 0.0044        | 12.8  | 7500  | 0.3343          | 28.5229 |
| 0.0026        | 13.65 | 8000  | 0.3204          | 23.3447 |
| 0.0019        | 14.51 | 8500  | 0.3103          | 24.6180 |
| 0.001         | 15.36 | 9000  | 0.3257          | 29.6265 |
| 0.0004        | 16.21 | 9500  | 0.3615          | 28.1834 |
| 0.0001        | 17.06 | 10000 | 0.3410          | 26.7402 |
| 0.0002        | 17.92 | 10500 | 0.3327          | 28.1834 |
| 0.0001        | 18.77 | 11000 | 0.3314          | 28.6078 |
| 0.0001        | 19.62 | 11500 | 0.3315          | 28.3531 |


### Framework versions

- Transformers 4.32.0
- Pytorch 2.1.0+cu121
- Datasets 2.14.7
- Tokenizers 0.13.3
