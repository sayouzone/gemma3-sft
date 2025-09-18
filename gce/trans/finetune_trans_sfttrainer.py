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

# System message for the assistant
system_message = """You are a text to SQL query translator. Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA."""

# User prompt that combines the user query and the schema
user_prompt = """Given the <USER_QUERY> and the <SCHEMA>, generate the corresponding SQL command to retrieve the desired data, considering the query's syntax, semantics, and schema constraints.

<SCHEMA>
{context}
</SCHEMA>

<USER_QUERY>
{question}
</USER_QUERY>
"""

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

def create_conversation(sample):
  return {
    "messages": [
      # {"role": "system", "content": system_message},
      {"role": "user", "content": user_prompt.format(question=sample["sql_prompt"], context=sample["sql_context"])},
      {"role": "assistant", "content": sample["sql"]}
    ]
  }

hf_token = get_hf_token(hf_token_secret_id)
print("HF_TOKEN", hf_token)
os.environ["HF_TOKEN"] = hf_token
#login_huggingface(hf_token)

# Load dataset from the hub
dataset = load_dataset("philschmid/gretel-synthetic-text-to-sql", split="train")
dataset = dataset.shuffle().select(range(12500))

# Convert dataset to OAI messages
dataset = dataset.map(create_conversation, remove_columns=dataset.features,batched=False)
# split dataset into 10,000 training samples and 2,500 test samples
dataset = dataset.train_test_split(test_size=2500/12500)

# Print formatted user prompt
print(dataset["train"][345]["messages"][1]["content"])

model_name = "gemma-3-12b"
# Hugging Face model id
#model_id = "google/gemma-3-1b-pt" # or `google/gemma-3-4b-pt`, `google/gemma-3-12b-pt`, `google/gemma-3-27b-pt`
model_id = f"google/{model_name}-pt"

# Select model class based on id
if model_id in ["google/gemma-3-1b-pt", "google/gemma-3-4b-pt", "google/gemma-3-12b-pt", "google/gemma-3-27b-pt"]:
    model_class = AutoModelForCausalLM
else:
    model_class = AutoModelForImageTextToText

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
#    use_cache=False,
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
model = model_class.from_pretrained(model_id, **model_kwargs)
#tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it") # Load the Instruction Tokenizer to use the official Gemma template
tokenizer = AutoTokenizer.from_pretrained(f"google/{model_name}-it") # Load the Instruction Tokenizer to use the official Gemma template

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
    modules_to_save=["lm_head", "embed_tokens"] # make sure to save the lm_head and embed_tokens as you train the special tokens
)

args = SFTConfig(
    #output_dir="gemma-3-1b-text-to-sql",    # directory to save and repository id
    output_dir=f"{model_name}-text-to-sql", # directory to save and repository id
    #max_seq_length=512,                     # max sequence length for model and packing of the dataset
    #max_length=512,                         # max sequence length for model and packing of the dataset
    max_length=384,                         # max sequence length for model and packing of the dataset
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
    train_dataset=dataset["train"],
    peft_config=peft_config,
    processing_class=tokenizer
)

# free the memory again
#del model
#del trainer
#torch.cuda.empty_cache()

# Start training, the model will be automatically saved to the Hub and the output directory
trainer.train()

# Save the final model again to the Hugging Face Hub
trainer.save_model()


# Load Model base model
model = model_class.from_pretrained(model_id, low_cpu_mem_usage=True)

# Merge LoRA and base model and save
peft_model = PeftModel.from_pretrained(model, args.output_dir)
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained(f"{model_name}_merged_model", safe_serialization=True, max_shard_size="2GB")

processor = AutoTokenizer.from_pretrained(args.output_dir)
processor.save_pretrained(f"{model_name}_merged_model")
