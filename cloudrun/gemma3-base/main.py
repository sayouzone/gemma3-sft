# main.py
import os
from dotenv import load_dotenv
from huggingface_hub import login

import torch
from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM

# .env 파일 로드
load_dotenv()

# 환경 변수 설정
#os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
# Cloud Run에서 설정한 환경 변수 이름(HF_TOKEN)으로 토큰을 가져옵니다.
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    # Hugging Face Hub에 로그인합니다.
    login(token=hf_token)
    print("Successfully logged in to Hugging Face Hub!")
else:
    print("HF_TOKEN environment variable not found.")

model_type = os.getenv("model_type")
model_id = os.getenv("model_id")

print("model_type", model_type)
print("model_id", model_id)

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# FastAPI 앱 초기화
app = FastAPI()

# 모델과 토크나이저를 전역적으로 로드 (앱 시작 시 한 번만 실행)
#model_id = "google/gemma-3-1b-it"  # 예: "google/gemma-3-9b-it" 또는 GCS 경로
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16 # 메모리 사용량 감소를 위해 bfloat16 사용
).to(device)

@app.post("/generate")
async def generate(request: Request):
    """텍스트 생성을 위한 엔드포인트"""
    json_body = await request.json()
    prompt = json_body.get("prompt")
    max_tokens = json_body.get("max_tokens", 100)

    if not prompt:
        return {"error": "Prompt is required"}, 400

    # Hugging Face Chat Template 사용
    chat = [{"role": "user", "content": prompt}]
    prompt_template = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(prompt_template, return_tensors="pt").to(device)

    # 텍스트 생성
    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"response": generated_text}

@app.get("/")
def health_check():
    """서비스 상태 확인을 위한 엔드포인트"""
    return {"status": "ok"}