# tpu-test-gemma-2b.py
import jax
import torch
import torch_xla as xm
import torch_xla.runtime as xr
import warnings

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from optimum.tpu import fsdp_v2
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig

"""
https://github.com/frank-morales2020/MLxDL/blob/main/Gemma_Finetuning_TPU_COLAB.ipynb
Fine-Tune GEMMA on GCE TPU
"""

model_name = "gemma-2b"
model_id = f"google/{model_name}"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Add custom token for padding Llama
tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)



data = load_dataset("Abirate/english_quotes")

def preprocess_function(samples):
    # Create prompt, completion, and combined text columns
    prompts = [f"Generate a quote:\n\n" for _ in samples["quote"]]
    completions = [f"{quote}{tokenizer.eos_token}" for quote in samples["quote"]]
    texts = [p + c for p, c in zip(prompts, completions)]
    return {"prompt": prompts, "completion": completions, "text": texts}

# data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)
data = data.map(
    preprocess_function,
    batched=True,
    remove_columns=data["train"].column_names
)


fsdp_training_args = fsdp_v2.get_fsdp_training_args(model)
print(fsdp_training_args)
"""
{'fsdp': 'full_shard', 'fsdp_config': {'transformer_layer_cls_to_wrap': ['GemmaDecoderLayer'], 'xla': True, 'xla_fsdp_v2': True, 'xla_fsdp_grad_ckpt': True}}
"""



# Set up PEFT LoRA for fine-tuning.
lora_config = LoraConfig(
    r=8,
    target_modules=["k_proj", "v_proj"],
    task_type="CAUSAL_LM",
)



print(f"Number of available JAX devices (TPU cores): {jax.device_count()}")
"""
Number of available JAX devices (TPU cores): 4
"""


num_devices = xm.runtime.world_size()
print(f"Number of available devices (TPU): {num_devices}")
"""
Number of available devices (TPU): 1
"""


warnings.filterwarnings("ignore")


def formatting_func(examples):
    # The 'prompt' column already contains the combined prompt and completion
    return examples["prompt"]

# Create a copy to avoid modifying the original dictionary
fsdp_training_args_copy = fsdp_training_args.copy()

# Extract fsdp_config from the copy
fsdp_config_value = fsdp_training_args_copy.get('fsdp_config', None)

# Remove keys that are explicitly passed to SFTConfig from the copy
if 'fsdp' in fsdp_training_args_copy:
    del fsdp_training_args_copy['fsdp']
if 'fsdp_config' in fsdp_training_args_copy:
    del fsdp_training_args_copy['fsdp_config']

# Reload the model and tokenizer to ensure a clean instance
print(f"Reloading model and tokenizer: {model_id}")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

# Load model configuration and set use_cache=False to avoid warning
config = AutoConfig.from_pretrained(model_id)
config.use_cache = False

model = AutoModelForCausalLM.from_pretrained(model_id, config=config, torch_dtype=torch.bfloat16)


# Check if the model is already a PeftModel and unload if necessary (should not be needed after reload, but kept as a safeguard)
if isinstance(model, PeftModel):
    print("Unloading existing PEFT adapters...")
    model = model.unload()

# Enable SPMD for torch_xla
xr.use_spmd()


sft_config = SFTConfig(
    per_device_train_batch_size=8,
    num_train_epochs=1,
    max_steps=-50,
    output_dir=f"./output/{model_name}-finetuned",
    optim="adafactor",
    logging_steps=50,
    dataloader_drop_last=True,  # Required by FSDP v2
    completion_only_loss=False, # Disable completion_only_loss
    fsdp="full_shard", # Explicitly set fsdp to "full_shard"
    fsdp_config=fsdp_config_value, # Pass fsdp_config dictionary directly
    **fsdp_training_args_copy, # Unpack remaining args from the copy
)


trainer = SFTTrainer(
    model=model,
    train_dataset=data["train"],
    args=sft_config, # Pass the SFTConfig object here
    peft_config=lora_config,
    #formatting_func=formatting_func, # Remove formatting_func when using SFTConfig
    #max_seq_length=512,
    #packing=True,
)

# 훈련 시작
print("Starting training...", flush=True)
trainer.train()
print("Training completed.", flush=True)

# Save the final model again to the Hugging Face Hub
# 학습된 LoRA 어댑터 저장
trainer.save_model()