import os
import re
import torch
from random import randint

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText, BitsAndBytesConfig
from transformers import QuantoConfig
from transformers import pipeline
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer

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

# Check if GPU benefits from bfloat16
#if torch.cuda.get_device_capability()[0] >= 8:
#    torch_dtype = torch.bfloat16
#else:
#    torch_dtype = torch.float16

"""
Traceback (most recent call last):
  File "/Users/seongjungkim/Development/python/agentspace/base-framework/src/sayou/sft/gemma3/m4/fine-tune_sfttrainer.py", line 14, in <module>
    if torch.cuda.get_device_capability()[0] >= 8:
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/seongjungkim/Library/Python/3.11/lib/python/site-packages/torch/cuda/__init__.py", line 600, in get_device_capability
    prop = get_device_properties(device)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/seongjungkim/Library/Python/3.11/lib/python/site-packages/torch/cuda/__init__.py", line 616, in get_device_properties
    _lazy_init()  # will define _get_device_properties
    ^^^^^^^^^^^^
  File "/Users/seongjungkim/Library/Python/3.11/lib/python/site-packages/torch/cuda/__init__.py", line 403, in _lazy_init
    raise AssertionError("Torch not compiled with CUDA enabled")
AssertionError: Torch not compiled with CUDA enabled
"""

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

# Define model init arguments
model_kwargs = dict(
    attn_implementation="eager", # Use "flash_attention_2" when running on Ampere or newer GPU
    torch_dtype=torch_dtype, # What torch dtype to use, defaults to auto
    device_map="auto", # Let torch decide how to load the model
    use_cache=False,
)

# BitsAndBytesConfig: Enables 4-bit quantization to reduce model size/memory usage
#model_kwargs["quantization_config"] = BitsAndBytesConfig(
#    load_in_4bit=True,
#    bnb_4bit_use_double_quant=True,
#    bnb_4bit_quant_type="nf4",
#    bnb_4bit_compute_dtype=model_kwargs['torch_dtype'],
#    bnb_4bit_quant_storage=model_kwargs['torch_dtype'],
#)

# BitsAndBytesConfig 부분은 주석 처리 (Apple Silicon M4 Pro)
# Quanto를 사용한 4비트 양자화 설정
model_kwargs["quantization_config"] = QuantoConfig(weights="int4")

# Load model and tokenizer
model = model_class.from_pretrained(model_id, **model_kwargs)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it") # Load the Instruction Tokenizer to use the official Gemma template

"""
config.json: 100%|████████████████████████████████████████████████████| 880/880 [00:00<00:00, 1.80MB/s]
Traceback (most recent call last):
  File "/Users/seongjungkim/Development/python/agentspace/base-framework/src/sayou/sft/gemma3/m4/fine-tune_sfttrainer.py", line 69, in <module>
    model = model_class.from_pretrained(model_id, **model_kwargs)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/seongjungkim/Library/Python/3.11/lib/python/site-packages/transformers/models/auto/auto_factory.py", line 600, in from_pretrained
    return model_class.from_pretrained(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/seongjungkim/Library/Python/3.11/lib/python/site-packages/transformers/modeling_utils.py", line 315, in _wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/Users/seongjungkim/Library/Python/3.11/lib/python/site-packages/transformers/modeling_utils.py", line 4819, in from_pretrained
    hf_quantizer.validate_environment(
  File "/Users/seongjungkim/Library/Python/3.11/lib/python/site-packages/transformers/quantizers/quantizer_bnb_4bit.py", line 88, in validate_environment
    raise ImportError(
ImportError: The installed version of bitsandbytes (<0.43.1) requires CUDA, but CUDA is not available. You may need to install PyTorch with CUDA support or upgrade bitsandbytes to >=0.43.1.
"""

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
    output_dir="gemma-3-1b-text-to-sql",         # directory to save and repository id
    #max_seq_length=512,                     # max sequence length for model and packing of the dataset
    packing=True,                           # Groups multiple samples in the dataset into a single sequence
    num_train_epochs=3,                     # number of training epochs
    per_device_train_batch_size=1,          # batch size per device during training
    gradient_accumulation_steps=4,          # number of steps before performing a backward/update pass
    gradient_checkpointing=True,            # use gradient checkpointing to save memory
    optim="adamw_torch_fused",              # use fused adamw optimizer
    logging_steps=10,                       # log every 10 steps
    save_strategy="epoch",                  # save checkpoint every epoch
    learning_rate=2e-4,                     # learning rate, based on QLoRA paper
    #fp16=True if torch_dtype == torch.float16 else False,   # use float16 precision
    #bf16=True if torch_dtype == torch.bfloat16 else False,   # use bfloat16 precision
    #bf16=True, # bfloat16 활성화
    bf16=False, # bfloat16 비활성화
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

# Create Trainer Object
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    peft_config=peft_config,
    processing_class=tokenizer
)

# Start training, the model will be automatically saved to the Hub and the output directory
trainer.train()

# Save the final model again to the Hugging Face Hub
trainer.save_model()

"""
QLoRA를 사용하면 전체 모델이 아닌 어댑터만 학습한다.
학습 중에 모델을 저장할 때는 전체 모델이 아닌 어댑터 가중치만 저장한다.
vLLM 또는 TGI와 같은 서빙 스택에서 더 쉽게 사용할 수 있도록 전체 모델을 저장하려면 merge_and_unload 메서드를 사용하여 어댑터 가중치를 모델 가중치에 병합한 다음 save_pretrained 메서드로 모델을 저장한다.
추론에 사용할 수 있는 기본 모델이 저장한다.

참고: 어댑터를 모델에 병합하려면 30GB 이상의 CPU 메모리가 필요하다. 이 단계를 건너뛰고 테스트 모델 추론을 진행해도 된다.
"""

# Load Model base model
model = model_class.from_pretrained(model_id, low_cpu_mem_usage=True)

# Merge LoRA and base model and save
peft_model = PeftModel.from_pretrained(model, args.output_dir)
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained("merged_model", safe_serialization=True, max_shard_size="2GB")

processor = AutoTokenizer.from_pretrained(args.output_dir)
processor.save_pretrained("merged_model")

# 모델 추론 테스트 및 SQL 쿼리 생성



"""
Traceback (most recent call last):
  File "/Users/seongjungkim/Development/python/agentspace/base-framework/src/sayou/sft/gemma3/m4/fine-tune_sfttrainer.py", line 144, in <module>
    args = SFTConfig(
           ^^^^^^^^^^
TypeError: SFTConfig.__init__() got an unexpected keyword argument 'max_seq_length'
"""

"""
Traceback (most recent call last):
  File "/Users/seongjungkim/Development/python/agentspace/base-framework/src/sayou/sft/gemma3/m4/fine-tune_sfttrainer.py", line 144, in <module>
    args = SFTConfig(
           ^^^^^^^^^^
  File "<string>", line 149, in __init__
  File "/Users/seongjungkim/Development/python/agentspace/gemma3-env/lib/python3.11/site-packages/trl/trainer/sft_config.py", line 247, in __post_init__
    super().__post_init__()
  File "/Users/seongjungkim/Development/python/agentspace/gemma3-env/lib/python3.11/site-packages/transformers/training_args.py", line 1729, in __post_init__
    raise ValueError(error_message)
ValueError: Your setup doesn't support bf16/gpu.
"""

"""
Traceback (most recent call last):
  File "/Users/seongjungkim/Development/python/agentspace/base-framework/src/sayou/sft/gemma3/m4/fine-tune_sfttrainer.py", line 172, in <module>
    trainer = SFTTrainer(
              ^^^^^^^^^^^
  File "/Users/seongjungkim/Development/python/agentspace/gemma3-env/lib/python3.11/site-packages/trl/trainer/sft_trainer.py", line 544, in __init__
    super().__init__(
  File "/Users/seongjungkim/Development/python/agentspace/gemma3-env/lib/python3.11/site-packages/transformers/utils/deprecation.py", line 172, in wrapped_func
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/Users/seongjungkim/Development/python/agentspace/gemma3-env/lib/python3.11/site-packages/transformers/trainer.py", line 688, in __init__
    self.callback_handler = CallbackHandler(
                            ^^^^^^^^^^^^^^^^
  File "/Users/seongjungkim/Development/python/agentspace/gemma3-env/lib/python3.11/site-packages/transformers/trainer_callback.py", line 449, in __init__
    self.add_callback(cb)
  File "/Users/seongjungkim/Development/python/agentspace/gemma3-env/lib/python3.11/site-packages/transformers/trainer_callback.py", line 466, in add_callback
    cb = callback() if isinstance(callback, type) else callback
         ^^^^^^^^^^
  File "/Users/seongjungkim/Development/python/agentspace/gemma3-env/lib/python3.11/site-packages/transformers/integrations/integration_utils.py", line 680, in __init__
    raise RuntimeError(
RuntimeError: TensorBoardCallback requires tensorboard to be installed. Either update your PyTorch version or install tensorboardX.
"""

"""
Using device: mps
Using dtype: torch.bfloat16
Map: 100%|█████████████████████████████████████████████| 12500/12500 [00:00<00:00, 33257.87 examples/s]
SELECT campaign, SUM(funding) AS total_funding FROM climate_communication_campaigns WHERE year = 2018 GROUP BY campaign ORDER BY total_funding DESC LIMIT 2;
/Users/seongjungkim/Development/python/agentspace/gemma3-env/lib/python3.11/site-packages/bitsandbytes/cextension.py:34: UserWarning: The installed version of bitsandbytes was compiled without GPU support. 8-bit optimizers, 8-bit multiplication, and GPU quantization are unavailable.
  warn("The installed version of bitsandbytes was compiled without GPU support. "
'NoneType' object has no attribute 'cadam32bit_grad_fp32'
/Users/seongjungkim/Development/python/agentspace/gemma3-env/lib/python3.11/site-packages/trl/trainer/sft_trainer.py:453: UserWarning: Padding-free training is enabled, but the attention implementation is not set to 'flash_attention_2'. Padding-free training flattens batches into a single sequence, and 'flash_attention_2' is the only known attention mechanism that reliably supports this. Using other implementations may lead to unexpected behavior. To ensure compatibility, set `attn_implementation='flash_attention_2'` in the model configuration, or verify that your attention mechanism can handle flattened sequences.
  warnings.warn(
/Users/seongjungkim/Development/python/agentspace/gemma3-env/lib/python3.11/site-packages/trl/trainer/sft_trainer.py:495: UserWarning: You are using packing, but the attention implementation is not set to 'flash_attention_2' or 'kernels-community/vllm-flash-attn3'. Packing flattens batches into a single sequence, and Flash Attention is the only known attention mechanisms that reliably support this. Using other implementations may lead to cross-contamination between batches. To avoid this, either disable packing by setting `packing=False`, or set `attn_implementation='flash_attention_2'` or `attn_implementation='kernels-community/vllm-flash-attn3'` in the model configuration.
  warnings.warn(
Tokenizing train dataset: 100%|█████████████████████████| 10000/10000 [00:02<00:00, 4651.18 examples/s]
Packing train dataset: 100%|██████████████████████████| 10000/10000 [00:00<00:00, 186265.33 examples/s]
Traceback (most recent call last):
  File "/Users/seongjungkim/Development/python/agentspace/gemma3-env/lib/python3.11/site-packages/huggingface_hub/utils/_http.py", line 409, in hf_raise_for_status
    response.raise_for_status()
  File "/Users/seongjungkim/Development/python/agentspace/gemma3-env/lib/python3.11/site-packages/requests/models.py", line 1026, in raise_for_status
    raise HTTPError(http_error_msg, response=self)
requests.exceptions.HTTPError: 403 Client Error: Forbidden for url: https://huggingface.co/api/repos/create

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/seongjungkim/Development/python/agentspace/base-framework/src/sayou/sft/gemma3/m4/fine-tune_sfttrainer.py", line 172, in <module>
    trainer = SFTTrainer(
              ^^^^^^^^^^^
  File "/Users/seongjungkim/Development/python/agentspace/gemma3-env/lib/python3.11/site-packages/trl/trainer/sft_trainer.py", line 544, in __init__
    super().__init__(
  File "/Users/seongjungkim/Development/python/agentspace/gemma3-env/lib/python3.11/site-packages/transformers/utils/deprecation.py", line 172, in wrapped_func
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/Users/seongjungkim/Development/python/agentspace/gemma3-env/lib/python3.11/site-packages/transformers/trainer.py", line 699, in __init__
    self.init_hf_repo()
  File "/Users/seongjungkim/Development/python/agentspace/gemma3-env/lib/python3.11/site-packages/transformers/trainer.py", line 4725, in init_hf_repo
    repo_url = create_repo(repo_name, token=token, private=self.args.hub_private_repo, exist_ok=True)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/seongjungkim/Development/python/agentspace/gemma3-env/lib/python3.11/site-packages/huggingface_hub/utils/_validators.py", line 114, in _inner_fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/Users/seongjungkim/Development/python/agentspace/gemma3-env/lib/python3.11/site-packages/huggingface_hub/hf_api.py", line 3768, in create_repo
    raise err
  File "/Users/seongjungkim/Development/python/agentspace/gemma3-env/lib/python3.11/site-packages/huggingface_hub/hf_api.py", line 3755, in create_repo
    hf_raise_for_status(r)
  File "/Users/seongjungkim/Development/python/agentspace/gemma3-env/lib/python3.11/site-packages/huggingface_hub/utils/_http.py", line 473, in hf_raise_for_status
    raise _format(HfHubHTTPError, message, response) from e
huggingface_hub.errors.HfHubHTTPError: (Request ID: Root=1-689d67df-47c312723bbb71cd4be26ef0;27dfb3d2-c1bb-4f31-9897-bb9d0e8daf92)

403 Forbidden: You don't have the rights to create a model under the namespace "java2core".
Cannot access content at: https://huggingface.co/api/repos/create.
Make sure your token has the correct permissions.
"""

"""
HF_TOKEN의 Role을 체크
"""

"""
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
"""

"""
/Users/seongjungkim/Development/python/agentspace/gemma3-env/lib/python3.11/site-packages/torch/utils/data/dataloader.py:692: UserWarning: 'pin_memory' argument is set as true but not supported on MPS now, device pinned memory won't be used.
  warnings.warn(warn_msg)
`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.

해결방법:
use_cache=False 추가

# Define model init arguments
model_kwargs = dict(
    attn_implementation="eager", # Use "flash_attention_2" when running on Ampere or newer GPU
    torch_dtype=torch_dtype, # What torch dtype to use, defaults to auto
    device_map="auto", # Let torch decide how to load the model
    use_cache=False,
)
"""