# main.py
import os
from dotenv import load_dotenv
from huggingface_hub import login

from fastapi import FastAPI, Request

# .env 파일 로드
load_dotenv()

# Cloud Run에서 설정한 환경 변수 이름(HF_TOKEN)으로 토큰을 가져옵니다.
hf_token = os.environ.get("HF_TOKEN")
#os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
if hf_token:
    # Hugging Face Hub에 로그인합니다.
    login(token=hf_token)
    print("Successfully logged in to Hugging Face Hub!")
else:
    print("HF_TOKEN environment variable not found.")

# 환경 변수 설정
model_type = os.getenv("model_type")
model_id = os.getenv("model_id")

print("model_type", model_type)
print("model_id", model_id)

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# FastAPI 앱 초기화
app = FastAPI()

@app.post("/generate")
async def generate(request: Request):
    """텍스트 생성을 위한 엔드포인트"""
    json_body = await request.json()
    prompt = json_body.get("prompt")
    max_tokens = json_body.get("max_tokens", 100)

    return {"response": hf_token}

@app.get("/")
def health_check():
    """서비스 상태 확인을 위한 엔드포인트"""
    return {"status": "ok"}