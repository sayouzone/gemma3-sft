import argparse
import os
import torch
from random import randint
import re

from datasets import load_dataset
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText

from dotenv import load_dotenv
from huggingface_hub import login

import google.auth
from google.cloud import secretmanager

# 모델 추론 테스트 및 SQL 쿼리 생성
"""
학습이 완료되면 모델을 평가하고 테스트해야 한다.
테스트 데이터 세트에서 다양한 샘플을 로드하고 이러한 샘플에서 모델을 평가할 수 있다.
테스트 데이터 세트에서 무작위 샘플을 로드하고 SQL 명령어를 생성한다.

https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora?hl=ko#test-model-inference-and-generate-sql-queries
"""

# --- 1. 기본 설정 및 인증 ---
# .env 파일 로드
load_dotenv()

# 환경 변수 설정
#os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
hf_token_secret_id = os.getenv("hf_token_secret_id")

#model_name = "gemma-3-4b"
model_name = "gemma-3-12b"
local_csv_file = "datasets/The_Wonderful_Wizard_of_Oz_1.csv"
dataset_name ="philschmid/gretel-synthetic-text-to-sql"


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

def sql_prompt(data):
    # System message for the assistant
    system_message = """You are a text to SQL query translator. 
Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA."""

    # User prompt that combines the user query and the schema
    user_prompt = """Given the <USER_QUERY> and the <SCHEMA>, 
generate the corresponding SQL command to retrieve the desired data, considering the query's syntax, semantics, and schema constraints.

<SCHEMA>
{context}
</SCHEMA>

<USER_QUERY>
{question}
</USER_QUERY>
"""

    return {
        "messages": [
            # {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt.format(question=data["sql_prompt"], context=data["sql_context"])},
            {"role": "assistant", "content": data["sql"]}
        ]
    }

def translation_prompt(data):
    """
    데이터셋 데이터를 명령어 형식의 프롬프트로 변환하는 함수
    A function that converts a dataset sample into a instruction prompt.
    """

    user_prompt = """### Instruction:
Translate the following text from English to Korean.

### Input:
{english_sentence}

### Response:
{korean_sentence}"""

    return {
        "text": user_prompt.format(
            english_sentence=data['english'], 
            korean_sentence=data['korean'])
    }

def translation_genre_prompt(data):
    """
    데이터셋 데이터를 장르 포함해서 명령어 형식의 프롬프트로 변환하는 함수
    A function that converts a dataset data into a instruction prompt including the genre.
    """

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

def inference_genre_prompt(genre, sentence):
    """
    데이터셋 데이터를 장르 포함해서 명령어 형식의 프롬프트로 변환하는 함수
    A function that converts a dataset data into a instruction prompt including the genre.
    """

    user_prompt = """### Instruction:
Translate the following text from English to Korean as {genre} genre.

### Input:
{english_sentence}

### Response:
"""

    #return {
    #    "role": "user",
    #    "text": user_prompt.format(
    #        genre=genre,
    #        english_sentence=sentence
    #    )
    #}
    return {
        "messages": [
            # {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt.format(genre=genre, english_sentence=sentence)},
            {"role": "assistant", "content": sentence}
        ]
    }
    #{"role": "user", "content": user_prompt.format(question=data["sql_prompt"], context=data["sql_context"])}

def load_dataset_file(file_paths):
    # 단일 파일 로드
    raw_dataset_dict = load_dataset('csv', data_files=file_paths)
    raw_dataset = raw_dataset_dict['train']

    # 여러 파일 한번에 로드 (train/test 분리)
    #data_files = {"train": "train.csv", "test": "test.csv"}
    #dataset = load_dataset('csv', data_files=data_files)

    print(raw_dataset)
    return raw_dataset

def load_dataset_sql(dataset_name):
    # Load dataset from the hub
    dataset = load_dataset(dataset_name, split="train")
    dataset = dataset.shuffle().select(range(12500))

    # Convert dataset to OAI messages
    dataset = dataset.map(sql_prompt, remove_columns=dataset.features,batched=False)
    # split dataset into 10,000 training samples and 2,500 test samples
    dataset = dataset.train_test_split(test_size=2500/12500)

    # Print formatted user prompt
    print(dataset["train"][345]["messages"][1]["content"])

    return dataset

def inference_model(model_name, sentence):
    # Apple Silicon GPU (MPS) 사용 가능 여부 확인
    if torch.backends.mps.is_available():
        device = "mps"
        # Apple Silicon GPU는 bfloat16을 지원.
        torch_dtype = torch.bfloat16
    elif torch.cuda.get_device_capability()[0] >= 8:
        device = "gpu"
        # Nividia GPU는 bfloat16을 지원.
        torch_dtype = torch.bfloat16
    else:
        device = "cpu"
        # CPU에서는 float16보다 float32가 더 안정적임.
        torch_dtype = torch.float32

    print(f"Using device: {device}")
    print(f"Using dtype: {torch_dtype}")

    # 모델 추론 테스트 및 SQL 쿼리 생성
    """
    학습이 완료되면 모델을 평가하고 테스트해야 한다.
    테스트 데이터 세트에서 다양한 샘플을 로드하고 이러한 샘플에서 모델을 평가할 수 있다.
    테스트 데이터 세트에서 무작위 샘플을 로드하고 SQL 명령어를 생성한다.
    """

    #model_id = f"{model_name}-text-to-sql"
    model_id = f"{model_name}_merged_model"

    # Load Model with PEFT adapter
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch_dtype,
        attn_implementation="eager",
    #    use_cache=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load the model and tokenizer into the pipeline
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    data = inference_genre_prompt("fantasy", sentence)

    # Convert as test example into a prompt with the Gemma template
    stop_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<end_of_turn>")]
    prompt = pipe.tokenizer.apply_chat_template(data["messages"][:2], tokenize=False, add_generation_prompt=True)

    # Generate our SQL query.
    outputs = pipe(prompt, max_new_tokens=256, do_sample=False, temperature=0.1, top_k=50, top_p=0.1, eos_token_id=stop_token_ids, disable_compile=True)

    return outputs

def translate_genre(model_name, genre, text):
    model_id = f"{model_name}_merged_model"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation="eager", # Use "flash_attention_2" when running on Ampere or newer GPU
        torch_dtype="auto",
        device_map="auto"  # 모델을 자동 로드
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # 패딩 토큰 설정 (Causal LM의 일반적인 관행)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt = f"""### Instruction:
Translate the following text from English to Korean as {genre} genre.

### Input:
{text}

### Response:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    print("inputs", inputs)
    outputs = model.generate(**inputs, max_new_tokens=len(text) * 3, eos_token_id=tokenizer.eos_token_id)
    print("outputs", outputs)
    #result = tokenizer.decode(outputs, skip_special_tokens=True)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_part = result.split("### Response:")
    print("result", result)
    print("response_part", response_part)
    return response_part

def __parse_args():
    parser = argparse.ArgumentParser(
        description='Command line to convert from html to markdown')

    parser.add_argument('--method', help='Method ex) finetune, merge, inference', default='finetune')
    parser.add_argument('--model', help='Gemma3 base model ex) gemma-3-1b, gemma-3-4b, gemma-3-12b, gemma-3-27b', default='gemma-3-4b')
    parser.add_argument('--csv-file', help='Translation Dataset csv file')
    parser.add_argument('--output', help='Finetuned merged directory', default='trans-en-ko')
    parser.add_argument('--sentence', help='Inference sentence', required=True, default="Hello, how are you?")

    return parser.parse_args()

if __name__ == "__main__":
    """
    python3 finetune_trans.py --method finetune --model gemma-3-4b --csv-file datasets/The_Wonderful_Wizard_of_Oz_1.csv --output trans-en-ko
    python3 finetune_trans.py --method merge --model gemma-3-4b --csv-file datasets/The_Wonderful_Wizard_of_Oz_1.csv --output trans-en-ko

    python3 finetune_trans.py --method finetune --model gemma-3-12b --csv-file datasets/The_Wonderful_Wizard_of_Oz_1.csv --output trans-en-ko
    python3 finetune_trans.py --method merge --model gemma-3-12b --csv-file datasets/The_Wonderful_Wizard_of_Oz_1.csv --output trans-en-ko

    python3 test_trans.py --method inference --model gemma-3-4b --csv-file datasets/The_Wonderful_Wizard_of_Oz_1.csv --sentence "Hi"
    """

    args = __parse_args()
    print('args', args)

    # --- 기본 설정 및 인증 ---
    # --- Basic Settings and Authentication ---
    # .env 파일 로드
    # Load .env file
    load_dotenv()

    # 환경 변수 설정
    # Setup environment variables
    #os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
    hf_token_secret_id = os.getenv("hf_token_secret_id")
    model_name = os.getenv("model_name")

    #model_name = "gemma-3-4b"
    model_name = args.model
    local_csv_file = args.csv_file
    output = args.output
    sentence = args.sentence
    #local_csv_file = "datasets/The_Wonderful_Wizard_of_Oz_1.csv"
    sql_dataset_name ="philschmid/gretel-synthetic-text-to-sql"
    #output_dir = f"{model_name}-trans-en-ko"
    output_dir = f"{model_name}-{output}"

    hf_token = get_hf_token(hf_token_secret_id)
    print("HF_TOKEN", hf_token)
    os.environ["HF_TOKEN"] = hf_token
    #login_huggingface(hf_token)

    dataset = load_dataset_file(local_csv_file)
    dataset = dataset.map(translation_genre_prompt, remove_columns=dataset.features,batched=False)

    # Finetune base model
    # Merge base model and LoRA
    print("model_name", model_name)
    print("output_dir", output_dir)
    outputs = inference_model(model_name, sentence=sentence)
    print("outputs", outputs)

    # Extract the user query and original answer
    #print(f"Context:\n", re.search(r'<SCHEMA>\n(.*?)\n</SCHEMA>', test_sample['messages'][0]['content'], re.DOTALL).group(1).strip())
    #print(f"Query:\n", re.search(r'<USER_QUERY>\n(.*?)\n</USER_QUERY>', test_sample['messages'][0]['content'], re.DOTALL).group(1).strip())
    #print(f"Original Answer:\n{test_sample['messages'][1]['content']}")
    #print(f"Generated Answer:\n{outputs[0]['generated_text'][len(prompt):].strip()}")
