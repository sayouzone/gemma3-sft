# Dockerfile
# 기본 이미지로 Python 3.10 슬림 버전을 사용합니다.
FROM python:3.12-slim

# 작업 디렉터리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 git 설치 (Hugging Face Hub에서 private 모델 로드 시 필요)
RUN apt-get update && apt-get install -y build-essential git
RUN pip install --upgrade pip

# requirements.txt 파일을 컨테이너에 복사하고 라이브러리를 설치합니다.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드를 컨테이너에 복사합니다.
COPY . .

# 컨테이너가 8080 포트를 사용하도록 설정합니다.
EXPOSE 8080

# Uvicorn을 사용하여 FastAPI 앱을 실행합니다.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]