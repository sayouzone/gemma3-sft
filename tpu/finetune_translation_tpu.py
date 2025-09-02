# finetune_translation_tpu.py
# Gemma-3 모델 로드 및 SFTTrainer 설정 스크립트
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer, SFTConfig
from optimum.tpu import TpuTrainer

import os
from dotenv import load_dotenv

# --- 1. 기본 설정 및 인증 ---
# .env 파일 로드
load_dotenv()

# 환경 변수 설정
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["PJRT_DEVICE"] = "TPU"

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

model_id = "google/gemma-3-4b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16, # TPU에 최적화된 bfloat16 사용
    device_map="auto", # 자동으로 장치에 모델을 분산
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

"""
training_args = SFTConfig(
    output_dir="./gemma-3-4b-en-ko-translation",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4, # 유효 배치 크기 = 8(코어) * 4 * 4 = 128
    learning_rate=2e-5,
    num_train_epochs=1,
    logging_steps=10,
    save_strategy="epoch",
    report_to="tensorboard",
    push_to_hub=True,
    hub_model_id="your-username/gemma-3-4b-en-ko-translation", # Hugging Face 사용자 이름으로 변경
    # FSDP v2 설정
    fsdp="full_shard",
    # TPU에 최적화된 옵티마이저 사용
    optim="adamw_torch_xla",
    # 데이터셋 관련 설정
    max_seq_length=512,
    #max_length=512,
    packing=True, # 여러 짧은 샘플을 하나의 시퀀스로 묶어 효율성 증대
    dataset_text_field="text",
)
"""
# 3. TrainingArguments 설정
# TPU 학습을 위한 몇 가지 주요 인자가 있습니다.
training_args = TrainingArguments(
    #output_dir="./tpu_results",       # 결과 저장 디렉토리
    output_dir="./gemma-3-4b-en-ko-translation",
    per_device_train_batch_size=16,   # TPU 코어 당 배치 사이즈
    num_train_epochs=3,               # 학습 에포크
    logging_dir='./logs',             # 로그 디렉토리
    logging_steps=10,
    # 'steps'를 기반으로 평가 및 저장을 설정하는 것이 좋습니다.
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    load_best_model_at_end=True,
)

# 데이터셋을 훈련 및 평가 세트로 분할
train_test_split = formatted_dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

"""
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    #dataset_text_field="text",
    #max_seq_length=512,
)
"""
# 4. TpuTrainer 생성
# 기존 transformers의 Trainer 대신 TpuTrainer를 사용합니다.
trainer = TpuTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# 훈련 시작
trainer.train()

# 훈련이 완료되면 모델을 Hub에 푸시합니다.
trainer.push_to_hub()