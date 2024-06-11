---
license: apache-2.0
base_model: openai/whisper-tiny
tags:
- generated_from_trainer
metrics:
- wer
model-index:
- name: torgo_tiny_finetune_F03
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# torgo_tiny_finetune_F03

This model is a fine-tuned version of [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0640
- Wer: 15.0892

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
| 0.6368        | 0.85  | 500   | 0.1136          | 7.8189  |
| 0.11          | 1.69  | 1000  | 0.0872          | 9.1907  |
| 0.0969        | 2.54  | 1500  | 0.0843          | 9.3278  |
| 0.0679        | 3.39  | 2000  | 0.0980          | 7.1331  |
| 0.053         | 4.24  | 2500  | 0.0756          | 7.1331  |
| 0.0361        | 5.08  | 3000  | 0.0637          | 9.1907  |
| 0.0278        | 5.93  | 3500  | 0.0491          | 8.3676  |
| 0.0233        | 6.78  | 4000  | 0.0446          | 27.8464 |
| 0.0148        | 7.63  | 4500  | 0.0403          | 12.8944 |
| 0.0149        | 8.47  | 5000  | 0.0748          | 28.6694 |
| 0.0105        | 9.32  | 5500  | 0.0631          | 17.6955 |
| 0.0087        | 10.17 | 6000  | 0.0619          | 12.0713 |
| 0.0075        | 11.02 | 6500  | 0.0525          | 18.6557 |
| 0.004         | 11.86 | 7000  | 0.0588          | 19.7531 |
| 0.0039        | 12.71 | 7500  | 0.0618          | 24.5542 |
| 0.0029        | 13.56 | 8000  | 0.0915          | 13.7174 |
| 0.0022        | 14.41 | 8500  | 0.0638          | 20.4390 |
| 0.0013        | 15.25 | 9000  | 0.0946          | 14.5405 |
| 0.0004        | 16.1  | 9500  | 0.0746          | 15.7750 |
| 0.0003        | 16.95 | 10000 | 0.0633          | 11.2483 |
| 0.0001        | 17.8  | 10500 | 0.0645          | 12.7572 |
| 0.0001        | 18.64 | 11000 | 0.0631          | 14.4033 |
| 0.0001        | 19.49 | 11500 | 0.0640          | 15.0892 |


### Framework versions

- Transformers 4.32.0
- Pytorch 2.1.0+cu121
- Datasets 2.14.7
- Tokenizers 0.13.3
