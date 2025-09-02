# load_model.py
# Gemma-3 모델 로드 및 SFTTrainer 설정 스크립트
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

import os
from dotenv import load_dotenv

# --- 1. 기본 설정 및 인증 ---
# .env 파일 로드
load_dotenv()

# 환경 변수 설정
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["PJRT_DEVICE"] = "TPU"

model_id = "google/gemma-3-4b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16, # TPU에 최적화된 bfloat16 사용
    device_map="auto", # 자동으로 장치에 모델을 분산
)

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
    #max_seq_length=512,
    max_length=512,
    packing=True, # 여러 짧은 샘플을 하나의 시퀀스로 묶어 효율성 증대
)