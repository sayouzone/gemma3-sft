import os
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText
from transformers import QuantoConfig
from trl import SFTConfig

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

# Define model init arguments
model_kwargs = dict(
    attn_implementation="eager", # Use "flash_attention_2" when running on Ampere or newer GPU
    torch_dtype=torch_dtype, # What torch dtype to use, defaults to auto
    device_map="auto", # Let torch decide how to load the model
    use_cache=False,
)

# BitsAndBytesConfig 부분은 주석 처리 (Apple Silicon M4 Pro)
# Quanto를 사용한 4비트 양자화 설정
model_kwargs["quantization_config"] = QuantoConfig(weights="int4")

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

# Load Model base model
model = model_class.from_pretrained(model_id, **model_kwargs, low_cpu_mem_usage=True)

# Merge LoRA and base model and save
peft_model = PeftModel.from_pretrained(model, args.output_dir)
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained("merged_model", safe_serialization=True, max_shard_size="2GB")

processor = AutoTokenizer.from_pretrained(args.output_dir)
processor.save_pretrained("merged_model")

"""
Using device: mps
Using dtype: torch.bfloat16
/Users/seongjungkim/Development/python/agentspace/gemma3-env/lib/python3.11/site-packages/bitsandbytes/cextension.py:34: UserWarning: The installed version of bitsandbytes was compiled without GPU support. 8-bit optimizers, 8-bit multiplication, and GPU quantization are unavailable.
  warn("The installed version of bitsandbytes was compiled without GPU support. "
'NoneType' object has no attribute 'cadam32bit_grad_fp32'
"""

"""
Using device: mps
Using dtype: torch.bfloat16
The following generation flags are not valid and may be ignored: ['cache_implementation']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
The following generation flags are not valid and may be ignored: ['cache_implementation']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
/Users/seongjungkim/Development/python/agentspace/gemma3-env/lib/python3.11/site-packages/bitsandbytes/cextension.py:34: UserWarning: The installed version of bitsandbytes was compiled without GPU support. 8-bit optimizers, 8-bit multiplication, and GPU quantization are unavailable.
  warn("The installed version of bitsandbytes was compiled without GPU support. "
'NoneType' object has no attribute 'cadam32bit_grad_fp32'
Traceback (most recent call last):
  File "/Users/seongjungkim/Development/python/agentspace/base-framework/src/sayou/sft/gemma3/m4/fine-tune_peft.py", line 93, in <module>
    merged_model.save_pretrained("merged_model", safe_serialization=True, max_shard_size="2GB")
  File "/Users/seongjungkim/Development/python/agentspace/gemma3-env/lib/python3.11/site-packages/transformers/modeling_utils.py", line 3856, in save_pretrained
    raise ValueError(
ValueError: The model is quantized with QuantizationMethod.QUANTO and is not serializable - check out the warnings from the logger on the traceback to understand the reason why the quantized model is not serializable.
"""