# -*- coding: utf-8 -*-
#!/usr/bin/env python3

# ==============================================================================
# GCE Nvidia L4 & Gemma 3 기반 영어-한국어 번역 모델 파인튜닝
# GCE Nvidia L4 & Gemma 3 based English-Korean Translation Model Fine-tuning
# ==============================================================================

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig, PeftModel
from trl import SFTConfig,  SFTTrainer

from dotenv import load_dotenv
from huggingface_hub import login

import google.auth
from google.cloud import secretmanager

def get_hf_token(hf_token_secret_id):
    # 기본 인증 정보 로드
    # GCE 인스턴스는 서비스 계정을 통해 자동으로 인증됩니다.
    # Load basic credentials
    # GCE instance are automatically authenticated through a service account.

    credentials, project = google.auth.default()
    print('project', project)
    print('credentials', credentials)
    print('credentials.service_account_email', credentials.service_account_email)
    #project = '1037372895180'

    # Secret Manager 클라이언트 생성
    # Create a Secret Manager client

    client = secretmanager.SecretManagerServiceClient()

    # 시크릿의 이름 (버전 포함)
    # Secret name (include version)

    secret_name = f"projects/{project}/secrets/{hf_token_secret_id}/versions/latest"

    payload = None
    try:
        # 시크릿 버전 접근 요청
        # Request access to secret version

        response = client.access_secret_version(name=secret_name)
        
        # 페이로드 데이터 디코딩
        # Decoding payload data

        payload = response.payload.data.decode("UTF-8")
        
        # 시크릿 값 출력
        # Print secret value

        print(f"Secret value: {payload}")
        
    except Exception as e:
        print(f"Error accessing secret: {e}")

    return payload

def login_huggingface(hf_token):
    #os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
    if hf_token:
        # Hugging Face Hub에 로그인합니다.
        # login Hugging Face Hub

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

def load_dataset_file(file_paths):
    # 단일 파일 로드
    # Load one file
    raw_dataset_dict = load_dataset('csv', data_files=file_paths)
    raw_dataset = raw_dataset_dict['train']

    # 여러 파일 한번에 로드 (train/test 분리)
    # Load multiple files at once (train/test separated)
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

def finetune_model(model_name, dataset):
    # Hugging Face model id
    #model_id = "google/gemma-3-1b-pt" # or `google/gemma-3-4b-pt`, `google/gemma-3-12b-pt`, `google/gemma-3-27b-pt`
    model_id = f"google/{model_name}-pt"

    # Check if GPU benefits from bfloat16
    if torch.cuda.get_device_capability()[0] >= 8:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16

    # Define model init arguments
    model_kwargs = dict(
        attn_implementation="eager", # Use "flash_attention_2" when running on Ampere or newer GPU
        torch_dtype=torch_dtype, # What torch dtype to use, defaults to auto
        #load_in_8bit=True, # or load_in_4bit=True
        device_map="auto", # Let torch decide how to load the model
        use_cache=False,
    )

    # BitsAndBytesConfig: Enables 4-bit quantization to reduce model size/memory usage
    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=model_kwargs['torch_dtype'],
        bnb_4bit_quant_storage=model_kwargs['torch_dtype'],
    )

    print("model_id", model_id)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    #tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it") # Load the Instruction Tokenizer to use the official Gemma template
    tokenizer = AutoTokenizer.from_pretrained(f"google/{model_name}-it") # Load the Instruction Tokenizer to use the official Gemma template

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        gradient_checkpointing=False,               # Disable gradient checkpointing
        modules_to_save=["lm_head", "embed_tokens"] # make sure to save the lm_head and embed_tokens as you train the special tokens
    )

    args = SFTConfig(
        #output_dir="gemma-3-1b-text-to-sql",    # directory to save and repository id
        output_dir=output_dir,                  # directory to save and repository id
        #max_seq_length=512,                     # max sequence length for model and packing of the dataset
        #max_length=512,                         # max sequence length for model and packing of the dataset
        #max_length=384,                         # max sequence length for model and packing of the dataset
        #packing=True,                           # Groups multiple samples in the dataset into a single sequence
        packing=False,                          # Groups multiple samples in the dataset into a single sequence
        num_train_epochs=3,                     # number of training epochs
        per_device_train_batch_size=1,          # batch size per device during training
        gradient_accumulation_steps=4,          # number of steps before performing a backward/update pass
        gradient_checkpointing=True,            # use gradient checkpointing to save memory
        #optim="adamw_torch_fused",              # use fused adamw optimizer
        optim="paged_adamw_8bit",               # Use the 8-bit paged optimizer
        logging_steps=10,                       # log every 10 steps
        save_strategy="epoch",                  # save checkpoint every epoch
        learning_rate=2e-4,                     # learning rate, based on QLoRA paper
        fp16=True if torch_dtype == torch.float16 else False,   # use float16 precision
        bf16=True if torch_dtype == torch.bfloat16 else False,   # use bfloat16 precision
        max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
        warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
        lr_scheduler_type="constant",           # use constant learning rate scheduler
        push_to_hub=True,                       # push model to hub
        report_to="tensorboard",                # report metrics to tensorboard
        dataset_kwargs={
            "add_special_tokens": False, # We template with special tokens
            "append_concat_token": True, # Add EOS token as separator token between examples
        }
    )

    # Create Trainer object
    trainer = SFTTrainer(
        model=model,
        args=args,
        #train_dataset=dataset["train"],
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer
    )

    # Start training, the model will be automatically saved to the Hub and the output directory
    trainer.train()

    # Save the final model again to the Hugging Face Hub
    trainer.save_model()

    # free the memory again
    #del model, trainer
    #torch.cuda.empty_cache()
    return model, tokenizer

def __finetune_model(trainer, peft_model, tokenizer):
    print("Starting training...")
    trainer.train()
    print("Training completed.")

    # Save the final model again to the Hugging Face Hub
    #trainer.save_model()

    # --- 모델 저장 및 병합 ---
    # --- Save and merge models ---

    print("Saving final adapter and merging...")
    #final_adapter_path = os.path.join(output_dir, "final_adapter")
    final_adapter_path = f"{model_name}-en-ko-trans_final_adapter"
    peft_model.save_pretrained(final_adapter_path)
    tokenizer.save_pretrained(final_adapter_path)

    # 병합을 위해 모델을 CPU로 이동 (메모리 문제 방지)
    # Move models to the CPU for merging (avoid memory issues)

    merged_model = peft_model.merge_and_unload()

    #merged_model_path = os.path.join(output_dir, f"{model_name}-trans_merged_model")
    merged_model_path = f"{model_name}-en-ko-trans_merged_model"
    merged_model.save_pretrained(merged_model_path)
    tokenizer.save_pretrained(merged_model_path)
    print(f"Adapter saved to '{final_adapter_path}'")
    print(f"Merged model saved to '{merged_model_path}'")

    # 메모리 정리
    # Memory cleanup

    del model, peft_model, merged_model, trainer
    torch.cuda.empty_cache() # MPS에서는 torch.mps.empty_cache()


def merge_model(model_name):
    # Hugging Face model id
    #model_id = "google/gemma-3-1b-pt" # or `google/gemma-3-4b-pt`, `google/gemma-3-12b-pt`, `google/gemma-3-27b-pt`
    model_id = f"google/{model_name}-pt"
    
    # 기본 모델 로드 
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True)

    # LoRA 그리고 기본 모델 변합 후 저장
    # Merge LoRA and base model and save
    peft_model = PeftModel.from_pretrained(model, output_dir)
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(f"{model_name}_merged_model", safe_serialization=True, max_shard_size="2GB")

    processor = AutoTokenizer.from_pretrained(output_dir)
    processor.save_pretrained(f"{model_name}_merged_model")


if __name__ == "__main__":
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

    model_name = "gemma-3-4b"
    local_csv_file = "datasets/The_Wonderful_Wizard_of_Oz_1.csv"
    sql_dataset_name ="philschmid/gretel-synthetic-text-to-sql"
    output_dir =f"{model_name}-trans-en-ko"

    hf_token = get_hf_token(hf_token_secret_id)
    print("HF_TOKEN", hf_token)
    os.environ["HF_TOKEN"] = hf_token
    #login_huggingface(hf_token)

    dataset = load_dataset_file(local_csv_file)
    dataset = dataset.map(translation_genre_prompt, remove_columns=dataset.features,batched=False)

    finetune_model(model_name, dataset)
    # Merge base model and LoRA
    print("model_name", model_name)
    print("output_dir", output_dir)
    #merge_model(model_name)