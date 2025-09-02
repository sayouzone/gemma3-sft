import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, TaskType
from huggingface_hub import login

dataset_name = "lemon-mint/korean_parallel_sentences_v1.1"
print(f"Loading and preparing dataset '{dataset_name}'...")

raw_dataset = load_dataset(dataset_name, split="train")

# Gemma-3-it 모델의 공식 프롬프트 형식을 따르는 것이 좋습니다.
# 그러나 번역과 같은 단순한 작업에서는 아래와 같은 커스텀 템플릿도 효과적입니다.
def create_translation_prompt(sample):
    """
    데이터셋 샘플을 받아 명령어 형식의 프롬프트 텍스트를 생성합니다.
    """
    source_lang = "English"
    target_lang = "Korean"
    prompt = f"""### Instruction:

Translate the following text from {source_lang} to {target_lang}.

Input:

{sample['english']}

Response:

{sample['korean']}"""

    return {"text": prompt}

# 데이터셋에 프롬프트 형식 적용
# num_proc를 설정하여 다중 프로세싱으로 속도를 높일 수 있습니다.
formatted_dataset = raw_dataset.map(create_translation_prompt, num_proc=4)
print(formatted_dataset)

"""
Dataset({
    features: ['korean', 'english', 'text'],
    num_rows: 492564
})
"""

# 변환된 결과 확인
#print(formatted_dataset['train']['text'])
print(formatted_dataset['text'])

# 여기에 자신의 Hugging Face 액세스 토큰을 입력하세요.
login(token="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") 

model_id = "google/gemma-3-1b-it" # 또는 다른 Gemma 3 모델 (예: 4b-it)

# bfloat16은 최신 GPU에서 float16보다 수치 안정성과 성능 면에서 유리합니다.
# device_map="mps"는 모델을 Apple Silicon GPU에 올리도록 명시합니다.
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    #torch_dtype=torch.bfloat16,
    torch_dtype=torch.float16,
    device_map="mps"
)


tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

# 훈련 가능한 파라미터 수 출력 (LoRA의 효율성 확인)
model.print_trainable_parameters()
# 예시 출력: trainable params: 4,718,592 |

training_args = TrainingArguments(
    output_dir="./gemma-en-ko-translator",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    #bf16=True, # MPS에서는 bf16 사용
    fp16=True, # MPS에서는 bf16 사용
    bf16=False, # MPS에서는 bf16 사용
    logging_steps=10,
    num_train_epochs=1, # 전체 데이터셋으로 1 에포크 훈련
    save_strategy="epoch", # 에포크 단위로 저장
    # MPS 관련 필수 설정
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
    # 기타 설정
    report_to="none", # wandb 등 로깅 서비스 사용 시 "wandb"로 변경
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    #train_dataset=formatted_dataset["train"], # 프롬프트 형식으로 변환된 데이터셋 사용
    train_dataset=formatted_dataset, # 프롬프트 형식으로 변환된 데이터셋 사용
    #dataset_text_field="text", # 데이터셋에서 텍스트 필드 이름 지정
    #tokenizer=tokenizer,
    #max_seq_length=512, # 모델이 처리할 최대 시퀀스 길이
    #packing=False, # 여러 짧은 샘플을 하나의 시퀀스로 묶지 않음
)

# 훈련 시작
trainer.train()
