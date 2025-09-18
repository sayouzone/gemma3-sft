from datasets import load_dataset
from huggingface_hub import HfApi, login

import os
from dotenv import load_dotenv

import google.auth
from google.cloud import secretmanager

"""
데이터 로딩: load_dataset 함수를 사용하여 Hugging Face Hub에서 데이터셋을 로드합니다.23
전처리 워크플로우:
1. 셔플(Shuffle): 데이터셋을 무작위로 섞어 학습 데이터의 편향을 방지합니다 (dataset.shuffle()).
2. 분할(Split): 과적합(overfitting)을 모니터링하기 위해 전체 데이터셋을 학습(train) 및 검증(validation) 세트로 분할합니다 (dataset.train_test_split()).
https://colab.research.google.com/github/google-health/medgemma/blob/main/notebooks/fine_tune_with_hugging_face.ipynb
3. 포맷팅 적용: .map() 함수를 사용하여 위에서 정의한 create_translation_prompt 함수를 데이터셋의 모든 샘플에 일괄적으로 적용하여, 모델이 요구하는 형식으로 변환합니다.
https://colab.research.google.com/github/google-gemini/gemma-cookbook/blob/main/Gemma/[Gemma_2]Finetune_with_Unsloth.ipynb
4. 토크나이징: 실제 토크나이징 과정은 이후 단계의 SFTTrainer가 내부적으로 처리하지만, 이 단계에서 포맷팅된 텍스트가 모델의 입력 ID로 변환된다는 점을 이해하는 것이 중요합니다.
"""

"""
# --- 3. 데이터셋 준비 및 형식화 ---
dataset_name = "lemon-mint/korean_parallel_sentences_v1.1"
print(f"Loading and preparing dataset '{dataset_name}'...")

raw_dataset = load_dataset(dataset_name, split="train")

# 훈련 시간을 줄이기 위해 작은 서브셋으로 테스트 (실제 훈련 시에는 전체 데이터 사용)
# raw_dataset = raw_dataset.shuffle(seed=42).select(range(10000))
"""

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

def load_dataset_file(file_paths):
    # 단일 파일 로드
    raw_dataset_dict = load_dataset('csv', data_files=file_paths)
    raw_dataset = raw_dataset_dict['train']

    # 여러 파일 한번에 로드 (train/test 분리)
    #data_files = {"train": "train.csv", "test": "test.csv"}
    #dataset = load_dataset('csv', data_files=data_files)

    print(raw_dataset)
    return raw_dataset

def create_translation_prompt(data):
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


if __name__ == "__main__":
    raw_dataset = load_dataset_file("datasets/The_Wonderful_Wizard_of_Oz.csv")

    print("raw_dataset", type(raw_dataset))
    print("raw_dataset", raw_dataset[:1])
    #.map()을 사용하여 전체 데이터셋에 프롬프트 형식 적용
    
    formatted_dataset = raw_dataset.map(create_translation_prompt, num_proc=4, remove_columns=raw_dataset.column_names)
    #formatted_dataset = formatted_dataset.shuffle().select(range(1500))
    print("Dataset formatted.")
    #print(f"Sample formatted prompt:\n{formatted_dataset['text']}")
    #print(f"Sample formatted prompt:\n{formatted_dataset['text'][0]}")

    #print(formatted_dataset[0])
    print(formatted_dataset)

    # --- 1. 기본 설정 및 인증 ---
    # .env 파일 로드
    load_dotenv()

    # 환경 변수 설정
    #os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
    hf_token_secret_id = os.getenv("hf_token_secret_id")

    hf_token =get_hf_token(hf_token_secret_id)
    #print("HF_TOKEN", hf_token)
    login_huggingface(hf_token)
    #login_huggingface("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")

    api = HfApi()
    repo_id = "sayouzone25/translation-dataset-v1" # Replace with your username and desired dataset name

    # Create a private repository
    #api.create_repo(repo_id=repo_id, repo_type="dataset", private=True)

    raw_dataset.push_to_hub(repo_id)