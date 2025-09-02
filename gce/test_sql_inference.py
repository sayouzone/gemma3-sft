import os
import torch
from random import randint
import re

from datasets import load_dataset
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText

# 모델 추론 테스트 및 SQL 쿼리 생성
"""
학습이 완료되면 모델을 평가하고 테스트해야 한다.
테스트 데이터 세트에서 다양한 샘플을 로드하고 이러한 샘플에서 모델을 평가할 수 있다.
테스트 데이터 세트에서 무작위 샘플을 로드하고 SQL 명령어를 생성한다.

https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora?hl=ko#test-model-inference-and-generate-sql-queries
"""

model_name = "gemma-3-4b"
# Hugging Face model id
model_id = f"google/{model_name}-pt" # or `google/gemma-3-4b-pt`, `google/gemma-3-12b-pt``, `google/gemma-3-27b-pt`
print("model_id", model_id)

# Select model class based on id
if model_id in ["google/gemma-3-1b-pt", "google/gemma-3-4b-pt", "google/gemma-3-12b-pt", "google/gemma-3-27b-pt"]:
    model_class = AutoModelForCausalLM
else:
    model_class = AutoModelForImageTextToText

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
model = model_class.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch_dtype,
    attn_implementation="eager",
#    use_cache=False,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load the model and tokenizer into the pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

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

def create_conversation(sample):
  return {
    "messages": [
      # {"role": "system", "content": system_message},
      {"role": "user", "content": user_prompt.format(question=sample["sql_prompt"], context=sample["sql_context"])},
      {"role": "assistant", "content": sample["sql"]}
    ]
  }

# Load dataset from the hub
dataset = load_dataset("philschmid/gretel-synthetic-text-to-sql", split="train")
dataset = dataset.shuffle().select(range(12500))

# Convert dataset to OAI messages
dataset = dataset.map(create_conversation, remove_columns=dataset.features, batched=False)
# split dataset into 10,000 training samples and 2,500 test samples
dataset = dataset.train_test_split(test_size=2500/12500)

# Print formatted user prompt
print(dataset["train"][345]["messages"][1]["content"])


# Load a random sample from the test dataset
rand_idx = randint(0, len(dataset["test"]))
test_sample = dataset["test"][rand_idx]

# Convert as test example into a prompt with the Gemma template
stop_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<end_of_turn>")]
prompt = pipe.tokenizer.apply_chat_template(test_sample["messages"][:2], tokenize=False, add_generation_prompt=True)

# Generate our SQL query.
outputs = pipe(prompt, max_new_tokens=256, do_sample=False, temperature=0.1, top_k=50, top_p=0.1, eos_token_id=stop_token_ids, disable_compile=True)

# Extract the user query and original answer
print(f"Context:\n", re.search(r'<SCHEMA>\n(.*?)\n</SCHEMA>', test_sample['messages'][0]['content'], re.DOTALL).group(1).strip())
print(f"Query:\n", re.search(r'<USER_QUERY>\n(.*?)\n</USER_QUERY>', test_sample['messages'][0]['content'], re.DOTALL).group(1).strip())
print(f"Original Answer:\n{test_sample['messages'][1]['content']}")
print(f"Generated Answer:\n{outputs[0]['generated_text'][len(prompt):].strip()}")
