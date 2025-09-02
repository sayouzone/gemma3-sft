# Dockerfile
# 기본 이미지로 Python 3.10 슬림 버전을 사용합니다.
FROM python:3.12-slim

# 작업 디렉터리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 git 설치 (Hugging Face Hub에서 private 모델 로드 시 필요)
RUN apt-get update && apt-get install -y build-essential git
# Install gcloud CLI for GCS access
#RUN apt-get update && \
#    apt-get install -y --no-install-recommends curl gnupg && \
#    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
#    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
#    apt-get update && \
#    apt-get install -y google-cloud-sdk build-essential git

RUN pip install --upgrade pip

# Download the model from GCS into the container's /app/model directory
# Replace with your actual GCS path
#RUN gcloud storage cp -r gs://sayouzone-ai-gemma3/gce-us-central1/gemma-3-1b_merged_model /app/model

# requirements.txt 파일을 컨테이너에 복사하고 라이브러리를 설치합니다.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드를 컨테이너에 복사합니다.
COPY . .
#RUN ls -al /app/model

# 컨테이너가 8080 포트를 사용하도록 설정합니다.
EXPOSE 8080

# Uvicorn을 사용하여 FastAPI 앱을 실행합니다.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]