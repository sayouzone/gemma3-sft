# load_dataset.py
# 데이터셋 로드 및 전처리 스크립트
from datasets import load_dataset

import os
from dotenv import load_dotenv

# --- 1. 기본 설정 및 인증 ---
# .env 파일 로드
load_dotenv()

# 환경 변수 설정
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# 데이터셋 로드 (예: aihub 데이터셋)
#dataset_name = "nayohan/aihub-en-ko-translation-1.2m"
#dataset_name = "nayohan/parallel_translation"
dataset_name = "lemon-mint/korean_parallel_sentences_v1.1"
dataset = load_dataset(dataset_name, split="train")

# 데이터셋 셔플 및 일부 선택 (빠른 실험을 위해)
dataset = dataset.shuffle(seed=42).select(range(200000))

# 데이터 클리닝 (간단한 예)
def clean_examples(examples):
    # 'en'과 'ko' 키가 모두 존재하는지 확인
    return [
        #(en, ko) for en, ko in zip(examples['en'], examples['ko'])
        (en, ko) for en, ko in zip(examples['english'], examples['korean'])
        if en is not None and ko is not None and len(en.strip()) > 0 and len(ko.strip()) > 0
    ]

def format_prompt(example):
    # Gemma의 템플릿은 <start_of_turn>과 <end_of_turn>을 사용하지만,
    # SFTTrainer는 이를 자동으로 처리해줍니다.
    # 여기서는 데이터셋의 'text' 필드를 생성하는 데 집중합니다.
    # 이 형식은 모델이 역할을 명확히 이해하도록 돕습니다.
    #prompt = f"### Instruction:\nTranslate the following English text to Korean.\n\n### English:\n{example['en']}\n\n### Korean:\n{example['ko']}"
    prompt = f"### Instruction:\nTranslate the following English text to Korean.\n\n### English:\n{example['english']}\n\n### Korean:\n{example['korean']}"
    return {"text": prompt}

formatted_dataset = dataset.map(format_prompt)

# 실제 클리닝 로직은 데이터셋에 따라 더 복잡해질 수 있습니다.
# 이 예제에서는 간단히 null 값과 빈 문자열만 제거합니다.
#dataset = dataset.filter(lambda example: example['en'] is not None and example['ko'] is not None)
dataset = dataset.filter(lambda example: example['english'] is not None and example['korean'] is not None)