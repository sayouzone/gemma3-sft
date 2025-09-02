#

##

####

```bash
export PROJECT_ID=sayouzone-ai
export TPU_NAME=gemma3-tpu-v5-test
export ZONE=us-central1-a
export ACCELERATOR_TYPE=v5litepod-4
export RUNTIME_VERSION=v2-alpha-tpuv5-lite
```

**TPU VM 인스턴스 생성**

```bash
gcloud compute tpus tpu-vm create $TPU_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --accelerator-type=$ACCELERATOR_TYPE \
    --version=$RUNTIME_VERSION
```

**SSH으로 TPU VM 연결**

```bash
gcloud compute tpus tpu-vm ssh $TPU_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE
```

**TPU VM 인스턴스 삭제**

```bash
gcloud compute tpus tpu-vm delete $TPU_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE
```

**TPU VM 목록 확인**

```bash
gcloud compute tpus tpu-vm list \
    --project=$PROJECT_ID \
    --zone=$ZONE
```

TPU 인스턴스 생성

```bash
Reauthentication required.
Please enter your password:
Reauthentication successful.
Create request issued for: [gemma3-tpu-v5-test]
Waiting for operation [projects/sayouzone-ai/locations/us-central1-a/operations/operation-1756167217248-63d398b2678
f4-66260447-49f5fb78] to complete...

Waiting for operation [projects/sayouzone-ai/locations/us-central1-a/operations/operation-1756167217248-63d398b2678
f4-66260447-49f5fb78] to complete...done.                                                                          
Created tpu [gemma3-tpu-v5-test].
```

####

```bash
sudo apt-get update
sudo apt-get install libopenblas-dev screen -y
pip install numpy
pip install torch torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html
```

```bash
sudo apt-get update
sudo apt-get install libopenblas-dev screen -y
pip install numpy
pip install torch torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html

# Hugging Face 생태계 라이브러리 설치
pip install transformers datasets accelerate evaluate peft trl
pip install python-dotenv
#pip install optimum-tpu
pip install optimum-tpu -f https://storage.googleapis.com/libtpu-releases/index.html
#pip install datasets trl==0.9.4
```

```
optimum-tpu==0.2.3
torch==2.5.1
torch-xla[tpu]==2.5.1
sentencepiece==0.2.0
transformers==4.46.3

```

**PyTorch가 TPU에 액세스할 수 있는지 확인**

```bash
PJRT_DEVICE=TPU python3 -c "import torch_xla.core.xla_model as xm; print(xm.get_xla_supported_devices(\"TPU\"))"
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/home/seongjungkim/.local/lib/python3.10/site-packages/torch_xla/core/xla_model.py", line 82, in get_xla_supported_devices
    for i, _ in enumerate(devices[:max_devices] if max_devices else devices)
TypeError: slice indices must be integers or None or have an __index__ method
```

```bash
PJRT_DEVICE=TPU python3 -c "import torch_xla; print(torch_xla.devices())"
[device(type='xla', index=0), device(type='xla', index=1), device(type='xla', index=2), device(type='xla', index=3)]
```

**기본 계산 수행**

[기본 계산 수행](https://cloud.google.com/tpu/docs/run-calculation-pytorch?hl=ko#perform_a_basic_calculation)

```python
# tpu-test_xla_model.py
import torch
import torch_xla.core.xla_model as xm

dev = xm.xla_device()
print('dev', dev)
t1 = torch.randn(3,3,device=dev)
t2 = torch.randn(3,3,device=dev)
print(t1 + t2)
```

```bash
python3 tpu-test_xla_model.py

WARNING:root:libtpu.so and TPU device found. Setting PJRT_DEVICE=TPU.
/home/seongjungkim/test_xla_model.py:4: DeprecationWarning: Use torch_xla.device instead
  dev = xm.xla_device()
dev xla:0
tensor([[ 0.3355, -1.4628, -3.2610],
        [-1.4656,  0.3196, -2.8766],
        [ 0.8667, -1.5060,  0.7125]], device='xla:0')
```

```bash
PJRT_DEVICE=TPU python3 tpu-test_xla_model.py

/home/seongjungkim/test_xla_model.py:4: DeprecationWarning: Use torch_xla.device instead
  dev = xm.xla_device()
tensor([[ 0.3355, -1.4628, -3.2610],
        [-1.4656,  0.3196, -2.8766],
        [ 0.8667, -1.5060,  0.7125]], device='xla:0')
```

```python
# tpu-test_xla_model1.py
import torch
#import torch_xla.core.xla_model as xm
import torch_xla

#dev = xm.xla_device()
dev = torch_xla.device()
t1 = torch.randn(3,3,device=dev)
t2 = torch.randn(3,3,device=dev)
print(t1 + t2)
```

```bash
python3 test_xla_model1.py

WARNING:root:libtpu.so and TPU device found. Setting PJRT_DEVICE=TPU.
tensor([[ 0.3355, -1.4628, -3.2610],
        [-1.4656,  0.3196, -2.8766],
        [ 0.8667, -1.5060,  0.7125]], device='xla:0')
```

```bash
PJRT_DEVICE=TPU python3 test_xla_model1.py

tensor([[ 0.3355, -1.4628, -3.2610],
        [-1.4656,  0.3196, -2.8766],
        [ 0.8667, -1.5060,  0.7125]], device='xla:0')
```

optimum-tpu 패키지 설치 오류

```bash
INFO: pip is looking at multiple versions of transformers to determine which version is compatible with other requirements. This could take a while.
ERROR: Cannot install torch-xla[tpu]==2.3.0 and torch-xla[tpu]==2.4.0 because these package versions have conflicting dependencies.

The conflict is caused by:
    torch-xla[tpu] 2.4.0 depends on libtpu-nightly==0.1.dev20240612; extra == "tpu"
    torch-xla[tpu] 2.3.0 depends on libtpu-nightly==0.1.dev20240322; extra == "tpu"

To fix this you could try to:
1. loosen the range of package versions you've specified
2. remove package versions to allow pip attempt to solve the dependency conflict

ERROR: ResolutionImpossible: for help visit https://pip.pypa.io/en/latest/topics/dependency-resolution/#dealing-with-dependency-conflicts
```

optimum-tpu 패키지 설치 성공

```bash
pip install optimum-tpu -f https://storage.googleapis.com/libtpu-releases/index.html
```

```bash
python3 --version

Python 3.10.12
```

```bash
huggingface-cli login
```

.env 생성

```text
HF_TOKEN=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
```

load_dataset.py

```bash
README.md: 2.48kB [00:00, 14.5MB/s]
data/train-00000-of-00001.parquet: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 134M/134M [00:02<00:00, 54.3MB/s]
Generating train split: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 492564/492564 [00:00<00:00, 1205594.36 examples/s]
Map: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200000/200000 [00:14<00:00, 13813.75 examples/s]
Filter: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200000/200000 [00:01<00:00, 112585.68 examples/s]
```

```bash
PJRT_DEVICE=TPU python3 load_dataset.py
```

```bash
python3 load_model.py

WARNING:root:libtpu.so and TPU device found. Setting PJRT_DEVICE=TPU.
tokenizer_config.json: 100%|███████████████████████████████████████████████████████████████| 1.16M/1.16M [00:00<00:00, 12.8MB/s]
tokenizer.model: 100%|█████████████████████████████████████████████████████████████████████| 4.69M/4.69M [00:00<00:00, 9.19MB/s]
tokenizer.json: 100%|███████████████████████████████████████████████████████████████████████| 33.4M/33.4M [00:00<00:00, 130MB/s]
added_tokens.json: 100%|██████████████████████████████████████████████████████████████████████| 35.0/35.0 [00:00<00:00, 622kB/s]
special_tokens_map.json: 100%|█████████████████████████████████████████████████████████████████| 662/662 [00:00<00:00, 11.8MB/s]
config.json: 100%|█████████████████████████████████████████████████████████████████████████████| 855/855 [00:00<00:00, 12.3MB/s]
model.safetensors.index.json: 100%|████████████████████████████████████████████████████████| 90.6k/90.6k [00:00<00:00, 4.02MB/s]
model-00002-of-00002.safetensors: 100%|████████████████████████████████████████████████████| 3.64G/3.64G [00:50<00:00, 72.4MB/s]
model-00001-of-00002.safetensors: 100%|████████████████████████████████████████████████████| 4.96G/4.96G [00:55<00:00, 89.9MB/s]
Fetching 2 files: 100%|███████████████████████████████████████████████████████████████████████████| 2/2 [00:55<00:00, 27.69s/it]
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 10.07it/s]
generation_config.json: 100%|██████████████████████████████████████████████████████████████████| 215/215 [00:00<00:00, 3.59MB/s]
```

```bash
PJRT_DEVICE=TPU python3 load_model.py

tokenizer_config.json: 100%|███████████████████████████████████████████████████████████████| 1.16M/1.16M [00:00<00:00, 12.8MB/s]
tokenizer.model: 100%|█████████████████████████████████████████████████████████████████████| 4.69M/4.69M [00:00<00:00, 9.19MB/s]
tokenizer.json: 100%|███████████████████████████████████████████████████████████████████████| 33.4M/33.4M [00:00<00:00, 130MB/s]
added_tokens.json: 100%|██████████████████████████████████████████████████████████████████████| 35.0/35.0 [00:00<00:00, 622kB/s]
special_tokens_map.json: 100%|█████████████████████████████████████████████████████████████████| 662/662 [00:00<00:00, 11.8MB/s]
config.json: 100%|█████████████████████████████████████████████████████████████████████████████| 855/855 [00:00<00:00, 12.3MB/s]
model.safetensors.index.json: 100%|████████████████████████████████████████████████████████| 90.6k/90.6k [00:00<00:00, 4.02MB/s]
model-00002-of-00002.safetensors: 100%|████████████████████████████████████████████████████| 3.64G/3.64G [00:50<00:00, 72.4MB/s]
model-00001-of-00002.safetensors: 100%|████████████████████████████████████████████████████| 4.96G/4.96G [00:55<00:00, 89.9MB/s]
Fetching 2 files: 100%|███████████████████████████████████████████████████████████████████████████| 2/2 [00:55<00:00, 27.69s/it]
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 10.07it/s]
generation_config.json: 100%|██████████████████████████████████████████████████████████████████| 215/215 [00:00<00:00, 3.59MB/s]
```

```bash
PJRT_DEVICE=TPU python3 finetune_translation.py

Map: 100%|█████████████████████████████████████████████████████████████████████| 200000/200000 [00:24<00:00, 8017.93 examples/s]
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00,  9.95it/s]
/home/sjkim/.local/lib/python3.10/site-packages/trl/trainer/sft_trainer.py:453: UserWarning: Padding-free training is enabled, but the attention implementation is not set to 'flash_attention_2'. Padding-free training flattens batches into a single sequence, and 'flash_attention_2' is the only known attention mechanism that reliably supports this. Using other implementations may lead to unexpected behavior. To ensure compatibility, set `attn_implementation='flash_attention_2'` in the model configuration, or verify that your attention mechanism can handle flattened sequences.
  warnings.warn(
/home/sjkim/.local/lib/python3.10/site-packages/trl/trainer/sft_trainer.py:495: UserWarning: You are using packing, but the attention implementation is not set to 'flash_attention_2' or 'kernels-community/vllm-flash-attn3'. Packing flattens batches into a single sequence, and Flash Attention is the only known attention mechanisms that reliably support this. Using other implementations may lead to cross-contamination between batches. To avoid this, either disable packing by setting `packing=False`, or set `attn_implementation='flash_attention_2'` or `attn_implementation='kernels-community/vllm-flash-attn3'` in the model configuration.
  warnings.warn(
Adding EOS to train dataset: 100%|████████████████████████████████████████████| 180000/180000 [00:14<00:00, 12368.09 examples/s]
Tokenizing train dataset: 100%|████████████████████████████████████████████████| 180000/180000 [00:50<00:00, 3548.26 examples/s]
Packing train dataset: 100%|██████████████████████████████████████████████████| 180000/180000 [00:03<00:00, 59552.40 examples/s
```

`packing=False`
`attn_implementation='flash_attention_2'`
`attn_implementation='kernels-community/vllm-flash-attn3'`


```json
{"accelerator_type":"v5litepod-4", "consumer_project":"sayouzone-ai", "consumer_project_num":"1037372895180", "evententry_timestamp":"2025-08-28T00:38:47.641756185Z", "stage":"", "system_available_memory_GiB":178.68426132202148, "uid":"8378495935069930559", "verb":"memoryanalyzer"}
```

- accelerator_type: "v5litepod-4"

This is the most important detail. You are using a TPU v5 Lite Pod with 4 chips. This hardware is designed for training and inference of AI models.

Each TPU v5e chip (which v5 Lite is based on) has 16 GiB of high-bandwidth memory (HBM). With 4 chips, your total available TPU memory is 64 GiB (4 chips x 16 GiB/chip).

- system_available_memory_GiB: 178.68

This shows you have about 179 GiB of available system RAM (Host CPU memory).

This is not the TPU's memory. It's the memory available to the host machine that manages the TPU. It's used for tasks like data loading, preprocessing, and general system operations.

- verb: "memoryanalyzer"

This simply indicates that the log entry was generated by a memory analysis or monitoring process.

```
dataset={
  'korean': [
    '이러한 기본 기동 외에도 서핑에는 수많은 다른 기동이 있습니다. 서퍼는 자신의 스타일과 파도의 조건에 따라 다양한 기동을 조합하여 독특한 서핑 경험을 만듭니다. 서핑 기동을 익히는 것은 시간과 연습이 필요하지만, 숙련되면 파도를 타는 즐거움을 한층 더 높일 수 있습니다.', 
    '시간을 잘 지키는 사람은 약속을 지키고, 시간을 낭비하지 않으며, 신뢰할 수 있습니다.', 
    '데스페라도는 종종 범죄자나 폭도로 묘사됩니다. 그들은 사회의 규범과 법률을 무시하고 자신의 목 표를 달성하기 위해 무엇이든 할 의향이 있습니다.', 
    '목관악기는 나무로 만들어진 악기로, 공기 기둥을 진동시켜 소리를 냅니다. 플루트, 클라리넷, 오보에, 바순 등이 대표적인 목관악기입니다. 목관악기는 관현악단에서 중요한 역할을 하며, 멜로디 연주, 하모니 구성, 리듬 유지 등 다양한 역할을 합니다.', 
    '최근 몇 년 동안 크 리스찬 뮤직은 대중 음악 장르와 점점 더 융합되고 있습니다. 예를 들어, 일부 크리스찬 아티스트들은 팝, 록, 힙합과 같은 세속적인 음악 스타일을 채택하고 있습니 다. 이러한 융합은 크리스찬 뮤직의 접근성과 영향력을 확대하는 데 기여했습니다.'
  ], 
  'english': [
    'In addition to these basic maneuvers, there are numerous other surfing maneuvers. Surfers combine different maneuvers, depending on their style and the wave conditions, to create a unique surfing experience. Mastering surfing maneuvers takes time and practice, but once proficient, it can greatly enhance the enjoyment of riding waves.', 
    "Punctual people keep appointments, don't waste time, and are reliable.", 
    'Desperados are often portrayed as criminals or outlaws. They disregard the norms and laws of society and are willing to do whatever it takes to achieve their goals.', 
    'Woodwind instruments are musical instruments made of wood that produce sound by vibrating a column of air. Common woodwind instruments include the flute, clarinet, oboe, and bassoon. Woodwinds play an essential role in the orchestra, performing various functions such as playing melodies, providing harmony, and maintaining rhythm.', 
    'In recent years, Christian music has increasingly converged with popular music genres. For instance, some Christian artists have adopted secular musical styles such as pop, rock, and hip-hop. This convergence has contributed to the widening accessibility and influence of Christian music.'
  ], 
  'text': [
    '### Instruction:\nTranslate the following English text to Korean.\n\n### English:\nIn addition to these basic maneuvers, there are numerous other surfing maneuvers. Surfers combine different maneuvers, depending on their style and the wave conditions, to create a unique surfing experience. Mastering surfing maneuvers takes time and practice, but once proficient, it can greatly enhance the enjoyment of riding waves.\n\n### Korean:\n이러한 기본 기동 외에도 서핑에는 수많은 다른 기동이 있습니다. 서퍼는 자신의 스타일과 파도의 조건에 따라 다양한 기동을 조합하여 독특한 서핑 경험을 만듭니다. 서핑 기동을 익히는 것은 시간과 연습이 필요하지만, 숙련되면 파도를 타는 즐거움을 한층 더 높일 수 있습니다.', 
    "### Instruction:\nTranslate the following English text to Korean.\n\n### English:\nPunctual people keep appointments, don't waste time, and are reliable.\n\n### Korean:\n시간을 잘 지키는 사람은 약속을 지키고, 시간을 낭비하지 않으며, 신뢰할 수 있습니다.", 
    '### Instruction:\nTranslate the following English text to Korean.\n\n### English:\nDesperados are often portrayed as criminals or outlaws. They disregard the norms and laws of society and are willing to do whatever it takes to achieve their goals.\n\n### Korean:\n데스페라도는 종종 범죄자나 폭도로 묘사됩니다. 그들은 사회의 규범과 법률을 무시하고 자신의 목표를 달성하기 위해 무엇이든 할 의향이 있습니다.', 
    '### Instruction:\nTranslate the following English text to Korean.\n\n### English:\nWoodwind instruments are musical instruments made of wood that produce sound by vibrating a column of air. Common woodwind instruments include the flute, clarinet, oboe, and bassoon. Woodwinds play an essential role in the orchestra, performing various functions such as playing melodies, providing harmony, and maintaining rhythm.\n\n### Korean:\n목관악기는 나무로 만들어진 악기로, 공기 기둥을 진동시켜 소리를 냅니다. 플루트, 클라리넷, 오보에, 바순 등이 대표적인 목관악기입니다. 목관악기는 관현악단에서 중요한 역할을 하며, 멜로디 연주, 하모니 구성, 리듬 유지 등 다양한 역할을 합니다.', 
    '### Instruction:\nTranslate the following English text to Korean.\n\n### English:\nIn recent years, Christian music has increasingly converged with popular music genres. For instance, some Christian artists have adopted secular musical styles such as pop, rock, and hip-hop. This convergence has contributed to the widening accessibility and influence of Christian music.\n\n### Korean:\n최근 몇 년 동안 크리스찬 뮤직은 대중 음악 장르와 점점 더 융합되고 있습니다. 예를 들어, 일부 크리스찬 아티스트들은 팝, 록, 힙합과 같은 세속적 인 음악 스타일을 채택하고 있습니다. 이러한 융합은 크리스찬 뮤직의 접근성과 영향력을 확대하는 데 기여했습니다.']
}
```

## Datasets

**표 2: 추천 영-한 병렬 코퍼스 조사**
| 데이터셋 이름 (Hugging Face ID) | 크기 (쌍의 수) | 도메인 | 비고/품질 |
|-----------------------------|-------------|-------|---------|
| nayohan/aihub-en-ko-translation-1.2m | 1,190,000 | 뉴스, 대화, 위키 | AI Hub에서 제공하는 대규모 고품질 데이터셋. 다양한 도메인을 포함. |
| bongsoo/news_talk_en_ko | 1,300,000 | 뉴스, 구어체 | 뉴스 기사와 대화체 텍스트를 포함하여 공식 및 비공식 번역에 유용. |
| lemon-mint/korean_english_parallel_wiki_augmented_v1 | 503,000 | 위키피디아 | 위키피디아 문서 기반으로, 정보성 텍스트 번역에 적합. |
| Moo/korean-parallel-corpora | 99,000 | 다양함 | 크기는 작지만, 빠른 프로토타이핑이나 특정 스타일 학습에 사용될 수 있음. |


## Errors

```bash
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00,  9.86it/s]
/home/sjkim/.local/lib/python3.10/site-packages/huggingface_hub/utils/_deprecation.py:100: FutureWarning: Deprecated argument(s) used in '__init__': dataset_text_field. Will not be supported from version '1.0.0'.

Deprecated positional argument(s) used in SFTTrainer, please use the SFTConfig to set these arguments instead.
  warnings.warn(message, FutureWarning)
/home/sjkim/.local/lib/python3.10/site-packages/trl/trainer/sft_trainer.py:278: UserWarning: You didn't pass a `max_seq_length` argument to the SFTTrainer, this will default to 1024
  warnings.warn(
/home/sjkim/.local/lib/python3.10/site-packages/trl/trainer/sft_trainer.py:307: UserWarning: You passed a `dataset_text_field` argument to the SFTTrainer, the value you passed will override the one in the `SFTConfig`.
  warnings.warn(
Traceback (most recent call last):
  File "/home/sjkim/finetune_translation.py", line 87, in <module>
    trainer = SFTTrainer(
  File "/home/sjkim/.local/lib/python3.10/site-packages/huggingface_hub/utils/_deprecation.py", line 101, in inner_f
    return f(*args, **kwargs)
  File "/home/sjkim/.local/lib/python3.10/site-packages/trl/trainer/sft_trainer.py", line 362, in __init__
    train_dataset = self._prepare_dataset(
  File "/home/sjkim/.local/lib/python3.10/site-packages/trl/trainer/sft_trainer.py", line 519, in _prepare_dataset
    return self._prepare_packed_dataloader(
  File "/home/sjkim/.local/lib/python3.10/site-packages/trl/trainer/sft_trainer.py", line 602, in _prepare_packed_dataloader
    constant_length_iterator = ConstantLengthDataset(
  File "/home/sjkim/.local/lib/python3.10/site-packages/trl/trainer/utils.py", line 475, in __init__
    self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
TypeError: unsupported operand type(s) for *: 'NoneType' and 'float'
```

```bash
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 10.00it/s]
/home/sjkim/.local/lib/python3.10/site-packages/huggingface_hub/utils/_deprecation.py:100: FutureWarning: Deprecated argument(s) used in '__init__': max_seq_length. Will not be supported from version '1.0.0'.

Deprecated positional argument(s) used in SFTTrainer, please use the SFTConfig to set these arguments instead.
  warnings.warn(message, FutureWarning)
/home/sjkim/.local/lib/python3.10/site-packages/trl/trainer/sft_trainer.py:269: UserWarning: You passed a `max_seq_length` argument to the SFTTrainer, the value you passed will override the one in the `SFTConfig`.
  warnings.warn(
Generating train split: 40851 examples [00:21, 1906.95 examples/s]
Generating train split: 4525 examples [00:02, 1916.33 examples/s]
```