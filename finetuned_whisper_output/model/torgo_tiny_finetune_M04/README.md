---
license: apache-2.0
base_model: openai/whisper-tiny
tags:
- generated_from_trainer
metrics:
- wer
model-index:
- name: torgo_tiny_finetune_M04
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# torgo_tiny_finetune_M04

This model is a fine-tuned version of [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 0.3396
- Wer: 32.3430

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
| 0.6227        | 0.84  | 500   | 0.3265          | 33.1070 |
| 0.1027        | 1.69  | 1000  | 0.3360          | 32.6825 |
| 0.1001        | 2.53  | 1500  | 0.4014          | 60.0170 |
| 0.0686        | 3.37  | 2000  | 0.3220          | 40.3226 |
| 0.048         | 4.22  | 2500  | 0.3146          | 31.3243 |
| 0.0366        | 5.06  | 3000  | 0.3477          | 57.8098 |
| 0.0262        | 5.9   | 3500  | 0.3054          | 21.8166 |
| 0.0237        | 6.75  | 4000  | 0.3007          | 43.4635 |
| 0.0153        | 7.59  | 4500  | 0.2969          | 24.8727 |
| 0.0149        | 8.43  | 5000  | 0.3628          | 52.8014 |
| 0.0112        | 9.27  | 5500  | 0.3670          | 29.7963 |
| 0.0096        | 10.12 | 6000  | 0.3354          | 24.5331 |
| 0.007         | 10.96 | 6500  | 0.3464          | 57.0458 |
| 0.0052        | 11.8  | 7000  | 0.3246          | 30.1358 |
| 0.0037        | 12.65 | 7500  | 0.3677          | 50.7640 |
| 0.0021        | 13.49 | 8000  | 0.3359          | 34.0407 |
| 0.002         | 14.33 | 8500  | 0.3406          | 41.7657 |
| 0.0011        | 15.18 | 9000  | 0.3296          | 36.3328 |
| 0.0004        | 16.02 | 9500  | 0.3359          | 33.5314 |
| 0.0           | 16.86 | 10000 | 0.3381          | 40.2377 |
| 0.0003        | 17.71 | 10500 | 0.3388          | 35.1443 |
| 0.0           | 18.55 | 11000 | 0.3410          | 33.4465 |
| 0.0001        | 19.39 | 11500 | 0.3396          | 32.3430 |


### Framework versions

- Transformers 4.32.0
- Pytorch 2.1.0+cu121
- Datasets 2.14.7
- Tokenizers 0.13.3
