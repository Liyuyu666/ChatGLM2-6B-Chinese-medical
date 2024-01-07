---
license: other
library_name: peft
tags:
- llama-factory
- lora
- generated_from_trainer
base_model: /home/cent/lyq/code/chatglm2-6b
model-index:
- name: train_2024-01-06-15-33-58
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# train_2024-01-06-15-33-58

This model is a fine-tuned version of [/home/cent/lyq/code/chatglm2-6b](https://huggingface.co//home/cent/lyq/code/chatglm2-6b) on the alpaca_zh dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 4
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 16
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- num_epochs: 1.0
- mixed_precision_training: Native AMP

### Training results



### Framework versions

- PEFT 0.7.1
- Transformers 4.36.2
- Pytorch 2.0.0
- Datasets 2.16.1
- Tokenizers 0.15.0