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
- Loss: 1.8882
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
- learning_rate: 1e-05
- train_batch_size: 16
- eval_batch_size: 1
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 500
- training_steps: 15000

### Training results

| Training Loss | Epoch | Step  | Validation Loss | Wer     |
|:-------------:|:-----:|:-----:|:---------------:|:-------:|
| 1.0961        | 0.83  | 500   | 1.1711          | 78.5455 |
| 0.0636        | 1.66  | 1000  | 1.2730          | 80.7273 |
| 0.0271        | 2.49  | 1500  | 1.3743          | 83.2727 |
| 0.0126        | 3.32  | 2000  | 1.3312          | 84.0    |
| 0.0084        | 4.15  | 2500  | 1.4455          | 85.0909 |
| 0.0056        | 4.98  | 3000  | 1.4981          | 89.8182 |
| 0.0038        | 5.8   | 3500  | 1.5312          | 87.6364 |
| 0.0033        | 6.63  | 4000  | 1.5521          | 83.6364 |
| 0.0023        | 7.46  | 4500  | 1.5808          | 83.6364 |
| 0.0016        | 8.29  | 5000  | 1.5671          | 87.6364 |
| 0.0012        | 9.12  | 5500  | 1.6538          | 89.8182 |
| 0.001         | 9.95  | 6000  | 1.6387          | 92.0    |
| 0.0004        | 10.78 | 6500  | 1.7502          | 94.5455 |
| 0.0006        | 11.61 | 7000  | 1.7028          | 95.6364 |
| 0.0011        | 12.44 | 7500  | 1.7247          | 91.6364 |
| 0.0002        | 13.27 | 8000  | 1.7171          | 94.1818 |
| 0.0002        | 14.1  | 8500  | 1.7441          | 92.0    |
| 0.0003        | 14.93 | 9000  | 1.7491          | 93.0909 |
| 0.0001        | 15.75 | 9500  | 1.7794          | 94.5455 |
| 0.0001        | 16.58 | 10000 | 1.8263          | 96.0    |
| 0.0002        | 17.41 | 10500 | 1.8542          | 97.0909 |
| 0.0002        | 18.24 | 11000 | 1.8260          | 94.5455 |
| 0.0001        | 19.07 | 11500 | 1.8537          | 95.6364 |
| 0.0001        | 19.9  | 12000 | 1.8420          | 96.3636 |
| 0.0001        | 20.73 | 12500 | 1.8646          | 96.7273 |
| 0.0           | 21.56 | 13000 | 1.8655          | 94.9091 |
| 0.0001        | 22.39 | 13500 | 1.8908          | 96.7273 |
| 0.0           | 23.22 | 14000 | 1.8906          | 97.8182 |
| 0.0001        | 24.05 | 14500 | 1.8865          | 96.0    |
| 0.0           | 24.88 | 15000 | 1.8882          | 96.0    |


### Framework versions

- Transformers 4.32.0
- Pytorch 2.1.0+cu121
- Datasets 2.14.7
- Tokenizers 0.13.3
