---
license: apache-2.0
base_model: openai/whisper-tiny
tags:
- generated_from_trainer
metrics:
- wer
model-index:
- name: torgo_tiny_finetune_M01
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# torgo_tiny_finetune_M01

This model is a fine-tuned version of [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 0.3526
- Wer: 96.6044

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
| 0.6272        | 0.85  | 500   | 0.2872          | 24.1087  |
| 0.1055        | 1.7   | 1000  | 0.3364          | 77.5042  |
| 0.0998        | 2.55  | 1500  | 0.3646          | 65.8744  |
| 0.0635        | 3.4   | 2000  | 0.3276          | 34.9745  |
| 0.0521        | 4.24  | 2500  | 0.3619          | 31.8336  |
| 0.0368        | 5.09  | 3000  | 0.3158          | 43.0390  |
| 0.0269        | 5.94  | 3500  | 0.3424          | 53.7351  |
| 0.0215        | 6.79  | 4000  | 0.2886          | 48.8964  |
| 0.0182        | 7.64  | 4500  | 0.3331          | 31.0696  |
| 0.0135        | 8.49  | 5000  | 0.3308          | 45.0764  |
| 0.0092        | 9.34  | 5500  | 0.2825          | 28.9474  |
| 0.0088        | 10.19 | 6000  | 0.3169          | 32.3430  |
| 0.0056        | 11.04 | 6500  | 0.3223          | 55.7725  |
| 0.0034        | 11.88 | 7000  | 0.3396          | 30.2207  |
| 0.0041        | 12.73 | 7500  | 0.3403          | 31.8336  |
| 0.0031        | 13.58 | 8000  | 0.3544          | 138.4550 |
| 0.0023        | 14.43 | 8500  | 0.3357          | 54.8387  |
| 0.0004        | 15.28 | 9000  | 0.3618          | 53.6503  |
| 0.0003        | 16.13 | 9500  | 0.3598          | 74.3633  |
| 0.0002        | 16.98 | 10000 | 0.3536          | 98.8964  |
| 0.0003        | 17.83 | 10500 | 0.3529          | 95.8404  |
| 0.0001        | 18.68 | 11000 | 0.3505          | 98.0475  |
| 0.0001        | 19.52 | 11500 | 0.3526          | 96.6044  |


### Framework versions

- Transformers 4.32.0
- Pytorch 2.1.0+cu121
- Datasets 2.14.7
- Tokenizers 0.13.3
