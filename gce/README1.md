```bash
Loading base model and tokenizer for 'google/gemma-3-4b-it'...
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:02<00:00,  1.43s/it]
Model and tokenizer loaded successfully.
Loading and preparing dataset 'lemon-mint/korean_parallel_sentences_v1.1'...
Dataset formatted.
Sample formatted prompt:
### Instruction:
Translate the following text from English to Korean.

### Input:
The uterus is located below the belly button, and its size and shape vary from woman to woman. The uterus is usually pear-shaped and about 7~8 cm in length.

### Response:
자궁은 배꼽 아래에 위치하며, 크기와 모양은 여성마다 다릅니다. 자궁은 일반적으로 배 모양이며, 길이는 약 7~8cm입니다.
Configuring LoRA...
trainable params: 16,394,240 || all params: 4,316,473,712 || trainable%: 0.3798
Converting train dataset to ChatML: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 492564/492564 [00:10<00:00, 46265.12 examples/s]
Applying chat template to train dataset: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 492564/492564 [00:10<00:00, 46850.46 examples/s]
Tokenizing train dataset: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 492564/492564 [03:29<00:00, 2350.26 examples/s]
Truncating train dataset: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 492564/492564 [01:44<00:00, 4719.18 examples/s]
Starting training...
  0%|                                                                                                                                                                                                              | 18/30786 [01:04<30:29:55,  3.57s/it


```

```bash
Loading base model and tokenizer for 'google/gemma-3-1b-it'...
config.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 899/899 [00:00<00:00, 6.78MB/s]
model.safetensors: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2.00G/2.00G [00:03<00:00, 508MB/s]
generation_config.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [00:00<00:00, 1.83MB/s]
Model and tokenizer loaded successfully.
Loading and preparing dataset 'lemon-mint/korean_parallel_sentences_v1.1'...
Dataset formatted.
Sample formatted prompt:
### Instruction:
Translate the following text from English to Korean.

### Input:
The uterus is located below the belly button, and its size and shape vary from woman to woman. The uterus is usually pear-shaped and about 7~8 cm in length.

### Response:
자궁은 배꼽 아래에 위치하며, 크기와 모양은 여성마다 다릅니다. 자궁은 일반적으로 배 모양이며, 길이는 약 7~8cm입니다.
Configuring LoRA...
trainable params: 6,522,880 || all params: 1,006,408,832 || trainable%: 0.6481
Applying chat template to train dataset: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 492564/492564 [00:11<00:00, 43302.48 examples/s]
Tokenizing train dataset: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 492564/492564 [03:30<00:00, 2339.69 examples/s]
Truncating train dataset:  84%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                          | 415285/492564 [01:30<00:17, 4525.53 examples/s]Truncating train dataset: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 492564/492564 [01:47<00:00, 4567.44 examples/s]
Starting training...
 0%|                                                                                                                                                                                                              | 17/30786 [00:45<22:20:43,  2.61s/it]{'loss': 1.2875, 'grad_norm': 0.9678901433944702, 'learning_rate': 0.0001998765672708374, 'mean_token_accuracy': 0.7258369915187359, 'epoch': 0.0}


```

```bash
INSTANCE_NAME=gemma3-g2s8-l4-test
ZONE_NAME=us-central1-c
gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE_NAME
```