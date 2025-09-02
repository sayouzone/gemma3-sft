from datasets import load_dataset

# Hugging Face Hub에서 데이터셋 로드
dataset_name = "lemon-mint/korean_parallel_sentences_v1.1"
raw_dataset = load_dataset(dataset_name)

# 데이터 구조 확인
print(raw_dataset)

# 첫 번째 샘플 확인
print(raw_dataset['train'])

"""
결과:
README.md: 2.48kB [00:00, 5.95MB/s]
data/train-00000-of-00001.parquet: 100%|█| 134M/134M [00:09<00:00, 13.
Generating train split: 100%|█| 492564/492564 [00:00<00:00, 2671159.82
DatasetDict({
    train: Dataset({
        features: ['korean', 'english'],
        num_rows: 492564
    })
})
Dataset({
    features: ['korean', 'english'],
    num_rows: 492564
})
"""