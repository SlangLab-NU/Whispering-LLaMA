---
license: apache-2.0
base_model: openai/whisper-tiny
tags:
- generated_from_trainer
metrics:
- wer
model-index:
- name: torgo_tiny_finetune_M02
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# torgo_tiny_finetune_M02

This model is a fine-tuned version of [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 0.3391
- Wer: 30.4754

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
| 0.6178        | 0.85  | 500   | 0.3570          | 29.2869 |
| 0.105         | 1.7   | 1000  | 0.3471          | 35.7385 |
| 0.1006        | 2.55  | 1500  | 0.3797          | 35.1443 |
| 0.0661        | 3.4   | 2000  | 0.3132          | 49.8302 |
| 0.0483        | 4.25  | 2500  | 0.3368          | 62.6486 |
| 0.0335        | 5.1   | 3000  | 0.2921          | 39.7284 |
| 0.0271        | 5.95  | 3500  | 0.3178          | 31.8336 |
| 0.0222        | 6.8   | 4000  | 0.3214          | 56.6214 |
| 0.0188        | 7.65  | 4500  | 0.3255          | 29.3718 |
| 0.0135        | 8.5   | 5000  | 0.3525          | 40.3226 |
| 0.0098        | 9.35  | 5500  | 0.3004          | 31.3243 |
| 0.0094        | 10.2  | 6000  | 0.3255          | 29.5416 |
| 0.0063        | 11.05 | 6500  | 0.3111          | 32.3430 |
| 0.0042        | 11.9  | 7000  | 0.3198          | 42.1053 |
| 0.0027        | 12.76 | 7500  | 0.2946          | 26.9100 |
| 0.0028        | 13.61 | 8000  | 0.3201          | 32.0034 |
| 0.0015        | 14.46 | 8500  | 0.3236          | 31.0696 |
| 0.0008        | 15.31 | 9000  | 0.3244          | 29.9660 |
| 0.0004        | 16.16 | 9500  | 0.3332          | 31.8336 |
| 0.0004        | 17.01 | 10000 | 0.3586          | 30.3905 |
| 0.0001        | 17.86 | 10500 | 0.3415          | 29.6265 |
| 0.0           | 18.71 | 11000 | 0.3403          | 29.7963 |
| 0.0001        | 19.56 | 11500 | 0.3391          | 30.4754 |


### Framework versions

- Transformers 4.32.0
- Pytorch 2.1.0+cu121
- Datasets 2.14.7
- Tokenizers 0.13.3
