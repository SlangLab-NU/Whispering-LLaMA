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
- Loss: 3.4295
- Wer: 96.0

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

| Training Loss | Epoch | Step  | Validation Loss | Wer      |
|:-------------:|:-----:|:-----:|:---------------:|:--------:|
| 0.1254        | 1.66  | 1000  | 1.8418          | 92.7273  |
| 0.0684        | 3.32  | 2000  | 2.5681          | 96.7273  |
| 0.0381        | 4.98  | 3000  | 2.7969          | 177.8182 |
| 0.0223        | 6.63  | 4000  | 2.9649          | 174.9091 |
| 0.0139        | 8.29  | 5000  | 3.3330          | 99.2727  |
| 0.0095        | 9.95  | 6000  | 3.4180          | 97.8182  |
| 0.0049        | 11.61 | 7000  | 3.5189          | 98.5455  |
| 0.0017        | 13.27 | 8000  | 3.4448          | 97.0909  |
| 0.001         | 14.93 | 9000  | 3.4596          | 98.5455  |
| 0.0004        | 16.58 | 10000 | 3.4080          | 96.3636  |
| 0.0003        | 18.24 | 11000 | 3.4101          | 96.3636  |
| 0.0           | 19.9  | 12000 | 3.4295          | 96.0     |


### Framework versions

- Transformers 4.32.0
- Pytorch 2.1.0+cu121
- Datasets 2.14.7
- Tokenizers 0.13.3
