---
license: apache-2.0
base_model: openai/whisper-tiny
tags:
- generated_from_trainer
metrics:
- wer
model-index:
- name: torgo_tiny_finetune_F04
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# torgo_tiny_finetune_F04

This model is a fine-tuned version of [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 0.3499
- Wer: 26.6553

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
| 0.6368        | 0.85  | 500   | 0.3637          | 28.9474 |
| 0.11          | 1.69  | 1000  | 0.3521          | 36.5025 |
| 0.0969        | 2.54  | 1500  | 0.2911          | 46.3497 |
| 0.0679        | 3.39  | 2000  | 0.2895          | 27.0798 |
| 0.053         | 4.24  | 2500  | 0.3115          | 26.9949 |
| 0.0361        | 5.08  | 3000  | 0.2972          | 28.8625 |
| 0.0278        | 5.93  | 3500  | 0.3036          | 26.9100 |
| 0.0233        | 6.78  | 4000  | 0.3311          | 59.0832 |
| 0.0148        | 7.63  | 4500  | 0.3000          | 27.6740 |
| 0.0149        | 8.47  | 5000  | 0.3317          | 37.6061 |
| 0.0105        | 9.32  | 5500  | 0.2975          | 29.4567 |
| 0.0087        | 10.17 | 6000  | 0.3593          | 27.1647 |
| 0.0075        | 11.02 | 6500  | 0.2840          | 28.0985 |
| 0.004         | 11.86 | 7000  | 0.3760          | 26.7402 |
| 0.0039        | 12.71 | 7500  | 0.3477          | 33.4465 |
| 0.0029        | 13.56 | 8000  | 0.3595          | 26.0611 |
| 0.0022        | 14.41 | 8500  | 0.3429          | 29.5416 |
| 0.0013        | 15.25 | 9000  | 0.2967          | 24.0238 |
| 0.0004        | 16.1  | 9500  | 0.3539          | 28.4380 |
| 0.0003        | 16.95 | 10000 | 0.3646          | 25.1273 |
| 0.0001        | 17.8  | 10500 | 0.3638          | 25.4669 |
| 0.0001        | 18.64 | 11000 | 0.3502          | 26.3158 |
| 0.0001        | 19.49 | 11500 | 0.3499          | 26.6553 |


### Framework versions

- Transformers 4.32.0
- Pytorch 2.1.0+cu121
- Datasets 2.14.7
- Tokenizers 0.13.3
