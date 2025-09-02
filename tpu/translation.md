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

```bash
gcloud compute tpus tpu-vm create $TPU_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --accelerator-type=$ACCELERATOR_TYPE \
    --version=$RUNTIME_VERSION
```

```bash
gcloud compute tpus tpu-vm ssh $TPU_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE
```

```bash
gcloud compute tpus tpu-vm delete $TPU_NAME \
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

load_dataset.py

```bash
README.md: 2.48kB [00:00, 14.5MB/s]
data/train-00000-of-00001.parquet: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 134M/134M [00:02<00:00, 54.3MB/s]
Generating train split: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 492564/492564 [00:00<00:00, 1205594.36 examples/s]
Map: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200000/200000 [00:14<00:00, 13813.75 examples/s]
Filter: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200000/200000 [00:01<00:00, 112585.68 examples/s]
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