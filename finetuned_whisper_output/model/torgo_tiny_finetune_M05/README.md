---
license: apache-2.0
base_model: openai/whisper-tiny
tags:
- generated_from_trainer
metrics:
- wer
model-index:
- name: torgo_tiny_finetune_M05
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# torgo_tiny_finetune_M05

This model is a fine-tuned version of [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 0.3072
- Wer: 24.1935

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
| 0.6262        | 0.84  | 500   | 0.3083          | 50.8489 |
| 0.1041        | 1.68  | 1000  | 0.3578          | 41.3413 |
| 0.0997        | 2.53  | 1500  | 0.3539          | 51.6978 |
| 0.0694        | 3.37  | 2000  | 0.3092          | 83.9559 |
| 0.0489        | 4.21  | 2500  | 0.3775          | 64.3463 |
| 0.0382        | 5.05  | 3000  | 0.3589          | 67.7419 |
| 0.0268        | 5.89  | 3500  | 0.3005          | 29.7114 |
| 0.0209        | 6.73  | 4000  | 0.3221          | 21.5620 |
| 0.0173        | 7.58  | 4500  | 0.3337          | 42.9542 |
| 0.0128        | 8.42  | 5000  | 0.3374          | 27.0798 |
| 0.011         | 9.26  | 5500  | 0.3639          | 20.7131 |
| 0.0083        | 10.1  | 6000  | 0.3622          | 24.9576 |
| 0.0066        | 10.94 | 6500  | 0.2958          | 21.5620 |
| 0.005         | 11.78 | 7000  | 0.3478          | 46.4346 |
| 0.0023        | 12.63 | 7500  | 0.3206          | 33.1919 |
| 0.0026        | 13.47 | 8000  | 0.3023          | 27.8438 |
| 0.0017        | 14.31 | 8500  | 0.2990          | 19.9491 |
| 0.0008        | 15.15 | 9000  | 0.2862          | 17.6570 |
| 0.0007        | 15.99 | 9500  | 0.2924          | 20.5433 |
| 0.0002        | 16.84 | 10000 | 0.2935          | 23.1749 |
| 0.0001        | 17.68 | 10500 | 0.3048          | 23.5993 |
| 0.0001        | 18.52 | 11000 | 0.3061          | 24.5331 |
| 0.0001        | 19.36 | 11500 | 0.3072          | 24.1935 |


### Framework versions

- Transformers 4.32.0
- Pytorch 2.1.0+cu121
- Datasets 2.14.7
- Tokenizers 0.13.3
