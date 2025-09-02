import os
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
#from optimum.quanto import QuantoConfig
from transformers import QuantoConfig

"""
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
"""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 1부: 라이브러리 임포트 및 기본 설정
# 모델 및 데이터셋 설정
#model_id = "google/gemma-3-4b-it"
#new_model_name = "gemma-3-4b-it-en-ko-translation"
model_id = "google/gemma-3-1b-it"
new_model_name = "gemma-3-1b-it-en-ko-trans"
dataset_name = "lemon-mint/korean_english_parallel_wiki_augmented_v1"

def format_translation_prompt(example):
    return {"text": f"<s> 다음 영문 텍스트를 한국어로 번역하세요: {example['english']} {example['korean']}</s>"}

# 2부: 데이터 준비
# 데이터셋 로드 및 전처리
dataset = load_dataset(dataset_name, split="train")
formatted_dataset = dataset.map(format_translation_prompt, remove_columns=dataset.column_names)
split_dataset = formatted_dataset.shuffle(seed=42).train_test_split(test_size=0.05)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# 3부: 4비트 양자화 및 모델 로드
# Quanto를 사용한 4비트 양자화 설정
quantization_config = QuantoConfig(weights="int4")

# 토크나이저 및 모델 로드
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    attn_implementation="eager",
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16, # bfloat16 사용으로 성능 향상 기대
    device_map="mps"
)

"""
Traceback (most recent call last):
  File "/Users/seongjungkim/Development/python/agentspace/base-framework/src/sayou/sft/gemma3/m4/load_model.py", line 6, in <module>
    from optimum.quanto import QuantoConfig
ImportError: cannot import name 'QuantoConfig' from 'optimum.quanto' (/Users/seongjungkim/Development/python/agentspace/gemma3-env/lib/python3.11/site-packages/optimum/quanto/__init__.py)

오류 해결:
#from optimum.quanto import QuantoConfig
from transformers import QuantoConfig
"""

# 4부: PEFT LoRA 구성
# LoRA 구성
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

# 5부: SFTConfig를 이용한 훈련 인자 설정
# SFTTrainer 훈련 인자 설정
training_arguments = SFTConfig(
    output_dir=new_model_name,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    logging_steps=10,
    max_steps=500,
    save_strategy="steps",
    save_steps=50,
    #bf16=True, # bfloat16 활성화
    bf16=False, # bfloat16 비활성화
    #fp16=True, # fp16 활성화
    report_to="none",
    packing=True,
    dataset_text_field="text",
)

# 6부: SFTTrainer 초기화 및 훈련 실행
# SFTTrainer 초기화
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    #tokenizer=tokenizer,
    args=training_arguments,
)

# 훈련 시작
trainer.train()

# 훈련된 모델(어댑터) 저장
trainer.save_model(f"{new_model_name}-final")

