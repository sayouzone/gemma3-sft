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

"""
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
"""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

"""
untimeError: MPS backend out of memory (MPS allocated: 11.70 GiB, other allocations: 14.83 GiB, max allowed: 27.20 GiB). Tried to allocate 952.00 MiB on private pool. 
Use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to disable upper limit for memory allocations (may cause system failure).
"""
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Hugging Face model id
model_id = "google/gemma-3-1b-pt" # or `google/gemma-3-4b-pt`, `google/gemma-3-12b-pt``, `google/gemma-3-27b-pt`

# Select model class based on id
if model_id == "google/gemma-3-1b-pt":
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

model_id = "gemma-text-to-sql"

# Load Model with PEFT adapter
model = model_class.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch_dtype,
    attn_implementation="eager",
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

"""
Using device: mps
Using dtype: torch.bfloat16
/Users/seongjungkim/Development/python/agentspace/gemma3-env/lib/python3.11/site-packages/bitsandbytes/cextension.py:34: UserWarning: The installed version of bitsandbytes was compiled without GPU support. 8-bit optimizers, 8-bit multiplication, and GPU quantization are unavailable.
  warn("The installed version of bitsandbytes was compiled without GPU support. "
'NoneType' object has no attribute 'cadam32bit_grad_fp32'
Device set to use mps
Map: 100%|██████████████████████████████████| 12500/12500 [00:00<00:00, 22195.31 examples/s]
SELECT country, SUM(installed_capacity) FROM renewable_energy_sources GROUP BY country;
The following generation flags are not valid and may be ignored: ['temperature', 'top_p']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
Context:
 CREATE TABLE if not exists asia_financial_assets (id INT, institution_name VARCHAR(100), country VARCHAR(50), is_shariah_compliant BOOLEAN, assets DECIMAL(15,2));
Query:
 Find the total assets of Shariah-compliant institutions in Indonesia and Malaysia?
Original Answer:
SELECT SUM(assets) FROM asia_financial_assets WHERE (country = 'Indonesia' OR country = 'Malaysia') AND is_shariah_compliant = TRUE;
Generated Answer:
SELECT SUM(assets) FROM asia_financial_assets WHERE country IN ('Indonesia', 'Malaysia') AND is_shariah_compliant = TRUE;
"""