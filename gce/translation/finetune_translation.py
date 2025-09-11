# ==============================================================================
# GCE Nvidia L4 & Gemma 3 기반 영어-한국어 번역 모델 파인튜닝 전체 스크립트
# ==============================================================================

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from trl import SFTTrainer

import os
from dotenv import load_dotenv
from huggingface_hub import login

import google.auth
from google.cloud import secretmanager

# --- 1. 기본 설정 및 인증 ---
# .env 파일 로드
load_dotenv()

# 환경 변수 설정
#os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
hf_token_secret_id = os.getenv("hf_token_secret_id")

def get_hf_token(hf_token_secret_id):
    # 기본 인증 정보 로드
    # GCE 인스턴스는 서비스 계정을 통해 자동으로 인증됩니다.
    credentials, project = google.auth.default()
    print('project', project)
    print('credentials', credentials)
    print('credentials.service_account_email', credentials.service_account_email)
    #project = '1037372895180'

    # Secret Manager 클라이언트 생성
    client = secretmanager.SecretManagerServiceClient()

    # 시크릿의 이름 (버전 포함)
    # 'YOUR_SECRET_ID'와 'LATEST_VERSION'을 실제 시크릿 정보로 바꿔주세요.
    # 'LATEST_VERSION'은 가장 최신 버전을 의미합니다.
    secret_name = f"projects/{project}/secrets/{hf_token_secret_id}/versions/latest"

    payload = None
    try:
        # 시크릿 버전 접근 요청
        response = client.access_secret_version(name=secret_name)
        
        # 페이로드 데이터 디코딩
        payload = response.payload.data.decode("UTF-8")
        
        # 시크릿 값 출력
        print(f"Secret value: {payload}")
        
    except Exception as e:
        print(f"Error accessing secret: {e}")

    return payload

def login_huggingface(hf_token):
    #os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
    if hf_token:
        # Hugging Face Hub에 로그인합니다.
        login(token=hf_token)
        print("Successfully logged in to Hugging Face Hub!")
    else:
        print("HF_TOKEN environment variable not found.")

def translation_prompt(data):
    """데이터셋 샘플을 명령어 형식의 프롬프트로 변환하는 함수"""
    user_prompt = """### Instruction:
Translate the following text from English to Korean.

### Input:
{english_sentence}

### Response:
{korean_sentence}"""

    return {
        "text": user_prompt.format(english_sentence=data['english'], korean_sentence=data['korean'])
    }

def translation_genre_prompt(data):
    """데이터셋 샘플을 명령어 형식의 프롬프트로 변환하는 함수"""
    user_prompt = """### Instruction:
Translate the following text from English to Korean as {genre} genre.

### Input:
{english_sentence}

### Response:
{korean_sentence}"""

    return {
        "text": user_prompt.format(
            genre=data['genre'],
            english_sentence=data['english'],
            korean_sentence=data['korean']
        )
    }

def translate_en_to_ko(model, tokenizer, text):
    prompt = f"""### Instruction:
Translate the following text from English to Korean.

### Input:
{text}

### Response:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    #print("inputs", inputs)
    outputs = model.generate(**inputs, max_new_tokens=len(text) * 3, eos_token_id=tokenizer.eos_token_id)
    #print("outputs", outputs)
    #result = tokenizer.decode(outputs, skip_special_tokens=True)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_part = result.split("### Response:")
    return response_part

hf_token = get_hf_token(hf_token_secret_id)
print("HF_TOKEN", hf_token)
os.environ["HF_TOKEN"] = hf_token
#login_huggingface(hf_token)

model_name = "gemma-3-4b"
# --- 2. 모델 및 토크나이저 로드 ---
model_id = f"google/{model_name}-it"
print(f"Loading base model and tokenizer for '{model_id}'...")

# Nvidia GPU 최적화된 bfloat16 데이터 타입 사용
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    attn_implementation="eager", # Use "flash_attention_2" when running on Ampere or newer GPU
    torch_dtype=torch.bfloat16,
    #torch_dtype=torch.float16,
    device_map="auto"  # 모델을 자동 로드
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
# 패딩 토큰 설정 (Causal LM의 일반적인 관행)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Model and tokenizer loaded successfully.")

# --- 3. 데이터셋 준비 및 형식화 ---
dataset_name = "lemon-mint/korean_parallel_sentences_v1.1"
print(f"Loading and preparing dataset '{dataset_name}'...")

raw_dataset = load_dataset(dataset_name, split="train")

# 훈련 시간을 줄이기 위해 작은 서브셋으로 테스트 (실제 훈련 시에는 전체 데이터 사용)
# raw_dataset = raw_dataset.shuffle(seed=42).select(range(10000))

#.map()을 사용하여 전체 데이터셋에 프롬프트 형식 적용
formatted_dataset = raw_dataset.map(translation_prompt, num_proc=4, remove_columns=raw_dataset.column_names)
#formatted_dataset = raw_dataset.map(translation_genre_prompt, num_proc=4, remove_columns=raw_dataset.column_names)
#formatted_dataset = formatted_dataset.shuffle().select(range(1500))
print("Dataset formatted.")
#print(f"Sample formatted prompt:\n{formatted_dataset['text']}")
print(f"Sample formatted prompt:\n{formatted_dataset['text'][5]}")

# --- 4. LoRA 구성 및 모델 준비 ---
print("Configuring LoRA...")
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

# PEFT 모델 생성
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()

# --- 5. SFTTrainer 구성 및 훈련 ---
output_dir = f"./{model_name}-en-ko-trans"

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    bf16=True,  # Mbf16 사용 여부 (GPU에서만 가능)
    #fp16=True,  # Use fp16 for mixed-precision on MPS
    logging_steps=20,
    num_train_epochs=1,
    save_strategy="epoch",
    report_to="none"
)

trainer = SFTTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=formatted_dataset,
    #dataset_text_field="text",
    #tokenizer=tokenizer,
    #max_seq_length=512,
    #packing=False,
)

print("Starting training...")
trainer.train()
print("Training completed.")

# Save the final model again to the Hugging Face Hub
#trainer.save_model()

# --- 6. 모델 저장 및 병합 ---
print("Saving final adapter and merging...")
#final_adapter_path = os.path.join(output_dir, "final_adapter")
final_adapter_path = f"{model_name}-en-ko-trans_final_adapter"
peft_model.save_pretrained(final_adapter_path)
tokenizer.save_pretrained(final_adapter_path)

# 병합을 위해 모델을 CPU로 이동 (메모리 문제 방지)
merged_model = peft_model.merge_and_unload()

#merged_model_path = os.path.join(output_dir, f"{model_name}-trans_merged_model")
merged_model_path = f"{model_name}-en-ko-trans_merged_model"
merged_model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)
print(f"Adapter saved to '{final_adapter_path}'")
print(f"Merged model saved to '{merged_model_path}'")

# 메모리 정리
del model, peft_model, merged_model, trainer
torch.cuda.empty_cache() # MPS에서는 torch.mps.empty_cache()

# --- 7. 병합된 모델로 추론 테스트 ---
print("\n--- Inference Test with Merged Model ---")

# 병합된 모델 로드
loaded_model = AutoModelForCausalLM.from_pretrained(
    merged_model_path,
    attn_implementation="eager", # Use "flash_attention_2" when running on Ampere or newer GPU
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
loaded_tokenizer = AutoTokenizer.from_pretrained(merged_model_path)

# 테스트 문장
test_sentence = "The ability to fine-tune powerful language models on consumer hardware is a significant breakthrough for the AI community."
translation = translate_en_to_ko(loaded_model, loaded_tokenizer, test_sentence)

print(f"English Input:\n{test_sentence}")
print(f"\nKorean Translation:\n{translation}")
