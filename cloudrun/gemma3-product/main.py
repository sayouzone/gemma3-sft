# main.py
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from contextlib import asynccontextmanager

from routers import auth, apis, views
from utils import storageapi, gemma3api

# .env 파일 로드
load_dotenv()

# 환경 변수 설정
#os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
model_type = os.getenv("model_type")
ft_local_path = os.getenv("ft_local_path")
ft_gcs_bucket = os.getenv("ft_gcs_bucket")
ft_gcs_path = os.getenv("ft_gcs_path")
print("model_type", model_type)
print("ft_local_path", ft_local_path)
print("ft_gcs_bucket", ft_gcs_bucket)
print("ft_gcs_path", ft_gcs_path)

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("sub startup")

    # Load the ML model
    #ml_models["answer_to_everything"] = fake_answer_to_everything_ml_model

    current_dir = os.path.dirname(os.path.realpath(__file__))
    print("current_dir", current_dir)

    #app_home = os.environ.get("APP_HOME")
    #print("APP_HOME", app_home)

    #if not app_home:
    #    os.environ["APP_HOME"] = current_dir

    storage_api = storageapi.StorageAPI(ft_gcs_bucket)
    #storage_api.download_folder(ft_gcs_path, ft_local_path)
    storage_api.download_gcs_folder(ft_gcs_path, ft_local_path)

    dir_list = os.listdir(ft_local_path)
    for item in dir_list:
        print(item)

    #config_name = os.environ.get("config", "prd_extern")
    #prop = properties.Properties()
    #prop.load_config(config_name, ['postgresql', 'openai','model','azure_openai'])
    #print('prop', prop)

    #args = common.make_args("hyd")
    #print('args', args, type(args), flush=True)

    #get_retriever(args, prop)

    # 1. model_id를 GCS 경로로 설정
    # 모델과 토크나이저를 전역적으로 로드 (앱 시작 시 한 번만 실행)
    #model_id = "google/gemma-3-1b-it"  # 예: "google/gemma-3-9b-it" 또는 GCS 경로
    #model_id = "gs://sayouzone-ai-gemma3/gce-us-central1/gemma-3-1b_merged_model"
    model_id = ft_local_path
    print("model_id", model_id)
    app.state.model_id = model_id
    app.state.gemma3_api = gemma3api.Gemma3API(model_id, model_type=model_type)

    yield

    # Clean up the ML models and release the resources
    #ml_models.clear()
    
    #clear_retriever()
    print("sub shutdown")

# FastAPI 앱 초기화
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

app.mount("/resource", StaticFiles(directory="static"), name="static")

app.include_router(apis.router)
app.include_router(auth.router)
app.include_router(views.router)

@app.get("/")
def health_check():
    """서비스 상태 확인을 위한 엔드포인트"""
    return {"status": "ok"}