# finetune_translation.py
# Gemma-3 모델 로드 및 SFTTrainer 설정 스크립트
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model
from trl import SFTTrainer, SFTConfig

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

# The input 'examples' is a batch from the dataset (a dictionary of lists)
def formatting_func(examples):
    # The trainer expects a list of strings, so we return the 'text' column directly.
    return examples["text"]

formatted_dataset = dataset.map(format_prompt)

# 실제 클리닝 로직은 데이터셋에 따라 더 복잡해질 수 있습니다.
# 이 예제에서는 간단히 null 값과 빈 문자열만 제거합니다.
#dataset = dataset.filter(lambda example: example['en'] is not None and example['ko'] is not None)
dataset = dataset.filter(lambda example: example['english'] is not None and example['korean'] is not None)

model_name = "gemma-3-4b"
model_id = f"google/{model_name}-it"
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
    hub_model_id="java2core/gemma-3-4b-en-ko-translation", # Hugging Face 사용자 이름으로 변경
    # FSDP v2 설정
    fsdp="full_shard",
    # TPU에 최적화된 옵티마이저 사용
    optim="adamw_torch_xla",
    # 데이터셋 관련 설정
    #max_seq_length=512,
    #max_length=512,
    #packing=True, # 여러 짧은 샘플을 하나의 시퀀스로 묶어 효율성 증대
    packing=False,
    #attn_implementation="flash_attention_2",
    #attn_implementation="kernels-community/vllm-flash-attn3",
    dataset_text_field="text",
)

# 데이터셋을 훈련 및 평가 세트로 분할
train_test_split = formatted_dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']
print(f"dataset={train_dataset[:5]}")

trainer = SFTTrainer(
    model=model,
    #tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    #dataset_text_field="text",
    #formatting_func=formatting_func,
    #max_seq_length=512,
)

# 훈련 시작
print("Starting training...", flush=True)
trainer.train()
print("Training completed.", flush=True)

# --- 6. 모델 저장 및 병합 ---
print("Saving final adapter and merging...")
#final_adapter_path = os.path.join(output_dir, "final_adapter")
final_adapter_path = f"{model_name}-en-ko-trans_final_adapter"
model.save_pretrained(final_adapter_path)
tokenizer.save_pretrained(final_adapter_path)

# 병합을 위해 모델을 CPU로 이동 (메모리 문제 방지)
merged_model = model.merge_and_unload()

#merged_model_path = os.path.join(output_dir, f"{model_name}-trans_merged_model")
merged_model_path = f"{model_name}-en-ko-trans_merged_model"
merged_model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)
print(f"Adapter saved to '{final_adapter_path}'")
print(f"Merged model saved to '{merged_model_path}'")

# 훈련이 완료되면 모델을 Hub에 푸시합니다.
trainer.push_to_hub()