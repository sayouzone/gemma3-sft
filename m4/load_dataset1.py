from datasets import load_dataset

def format_translation_prompt(example):
    """
    번역 데이터셋의 각 샘플을 SFTTrainer에 적합한 프롬프트 형식으로 변환합니다.
    """
    prompt = f"<s> 다음 영문 텍스트를 한국어로 번역하세요: {example['english']} {example['korean']}</s>"
    return {"text": prompt}

# 데이터셋 로드
dataset_name = "lemon-mint/korean_english_parallel_wiki_augmented_v1"
dataset = load_dataset(dataset_name, split="train")

# 데이터셋 형식 변환
formatted_dataset = dataset.map(format_translation_prompt, remove_columns=dataset.column_names)

# 데이터셋 섞기 및 분할 (95% 훈련, 5% 검증)
shuffled_dataset = formatted_dataset.shuffle(seed=42)
split_dataset = shuffled_dataset.train_test_split(test_size=0.05)

train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# 결과 확인
print(f"훈련 데이터셋 크기: {len(train_dataset)}")
print(f"검증 데이터셋 크기: {len(eval_dataset)}")
print("\n변환된 데이터 샘플:")
print(train_dataset['text'])
