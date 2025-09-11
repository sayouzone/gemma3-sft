import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from PIL import Image

from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForCausalLM, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig, PeftModel
from trl import SFTConfig,  SFTTrainer

"""
Hugging Face TRL SFTTrainer를 사용하면 개방형 LLM의 미세 조정을 간편하게 감독
SFTTrainer는 transformers 라이브러리의 Trainer의 서브클래스이며 로깅, 평가, 체크포인트 설정을 비롯한 동일한 모든 기능을 지원

- 대화형 및 안내 형식을 포함한 데이터 세트 형식 지정
- 프롬프트를 무시하고 완료만 학습
- 더 효율적인 학습을 위해 데이터 세트 패킹
- QloRA를 포함한 매개변수 효율적인 미세 조정 (PEFT) 지원
- 대화형 미세 조정을 위한 모델 및 토큰 생성기 준비 (예: 특수 토큰 추가)

https://ai.google.dev/gemma/docs/core/huggingface_vision_finetune_qlora?hl=ko
"""

os.environ["HF_TOKEN"] = "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"

# System message for the assistant
system_message = "You are an expert product description writer for Amazon."

# User prompt that combines the user query and the schema
user_prompt = """Create a Short Product description based on the provided <PRODUCT> and <CATEGORY> and image.
Only return description. The description should be SEO optimized and for a better mobile search experience.

<PRODUCT>
{product}
</PRODUCT>

<CATEGORY>
{category}
</CATEGORY>
"""

# Convert dataset to OAI messages
def format_data(sample):
    return {
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt.format(
                            product=sample["Product Name"],
                            category=sample["Category"],
                        ),
                    },
                    {
                        "type": "image",
                        "image": sample["image"],
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["description"]}],
            },
        ],
    }

def process_vision_info(messages: list[dict]) -> list[Image.Image]:
    image_inputs = []
    # Iterate through each conversation
    for msg in messages:
        # Get content (ensure it's a list)
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]

        # Check each content element for images
        for element in content:
            if isinstance(element, dict) and (
                "image" in element or element.get("type") == "image"
            ):
                # Get the image and convert to RGB
                if "image" in element:
                    image = element["image"]
                else:
                    image = element
                image_inputs.append(image.convert("RGB"))
    return image_inputs

# Load dataset from the hub
dataset = load_dataset("philschmid/amazon-product-descriptions-vlm", split="train")

# Convert dataset to OAI messages
# need to use list comprehension to keep Pil.Image type, .mape convert image to bytes
dataset = [format_data(sample) for sample in dataset]

print(dataset[345]["messages"])

model_name = "gemma-3-4b"
# Hugging Face model id
model_id = f"google/{model_name}-pt" # or `google/gemma-3-12b-pt`, `google/gemma-3-27b-pt`

# Check if GPU benefits from bfloat16
if torch.cuda.get_device_capability()[0] < 8:
    raise ValueError("GPU does not support bfloat16, please use a GPU that supports bfloat16.")

# Select model class based on id
if model_id in ["google/gemma-3-1b-pt", "google/gemma-3-4b-pt", "google/gemma-3-12b-pt", "google/gemma-3-27b-pt"]:
    model_class = AutoModelForCausalLM
else:
    model_class = AutoModelForImageTextToText

# Define model init arguments
model_kwargs = dict(
    attn_implementation="eager", # Use "flash_attention_2" when running on Ampere or newer GPU
    torch_dtype=torch.bfloat16, # What torch dtype to use, defaults to auto
    #load_in_8bit=True, # or load_in_4bit=True
    device_map="auto", # Let torch decide how to load the model
)

# BitsAndBytesConfig int-4 config
model_kwargs["quantization_config"] = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
    bnb_4bit_quant_storage=model_kwargs["torch_dtype"],
)

print("model_id", model_id)
# Load model and tokenizer
model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
processor = AutoProcessor.from_pretrained(f"google/{model_name}-it")

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
    output_dir=f"{model_name}-product-description",     # directory to save and repository id
    num_train_epochs=1,                         # number of training epochs
    per_device_train_batch_size=1,              # batch size per device during training
    gradient_accumulation_steps=4,              # number of steps before performing a backward/update pass
    gradient_checkpointing=True,                # use gradient checkpointing to save memory
    optim="adamw_torch_fused",                  # use fused adamw optimizer
    logging_steps=5,                            # log every 5 steps
    save_strategy="epoch",                      # save checkpoint every epoch
    learning_rate=2e-4,                         # learning rate, based on QLoRA paper
    bf16=True,                                  # use bfloat16 precision
    max_grad_norm=0.3,                          # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,                          # warmup ratio based on QLoRA paper
    lr_scheduler_type="constant",               # use constant learning rate scheduler
    push_to_hub=True,                           # push model to hub
    report_to="tensorboard",                    # report metrics to tensorboard
    gradient_checkpointing_kwargs={
        "use_reentrant": False
    },  # use reentrant checkpointing
    dataset_text_field="",                      # need a dummy field for collator
    dataset_kwargs={"skip_prepare_dataset": True},  # important for collator
)
args.remove_unused_columns = False # important for collator

# Create a data collator to encode text and image pairs
def collate_fn(examples):
    texts = []
    images = []
    for example in examples:
        image_inputs = process_vision_info(example["messages"])
        text = processor.apply_chat_template(
            example["messages"], add_generation_prompt=False, tokenize=False
        )
        texts.append(text.strip())
        images.append(image_inputs)

    # Tokenize the texts and process the images
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    # The labels are the input_ids, and we mask the padding tokens and image tokens in the loss computation
    labels = batch["input_ids"].clone()

    # Mask image tokens
    image_token_id = [
        processor.tokenizer.convert_tokens_to_ids(
            processor.tokenizer.special_tokens_map["boi_token"]
        )
    ]
    # Mask tokens for not being used in the loss computation
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == image_token_id] = -100
    labels[labels == 262144] = -100

    batch["labels"] = labels
    return batch

# Create Trainer object
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=processor,
    data_collator=collate_fn,
)

# Start training, the model will be automatically saved to the Hub and the output directory
trainer.train()

# Save the final model again to the Hugging Face Hub
trainer.save_model()


# Load Model base model
model = AutoModelForImageTextToText.from_pretrained(model_id, low_cpu_mem_usage=True)

# Merge LoRA and base model and save
peft_model = PeftModel.from_pretrained(model, args.output_dir)
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained(f"{model_name}-product_merged_model", safe_serialization=True, max_shard_size="2GB")

processor = AutoProcessor.from_pretrained(args.output_dir)
processor.save_pretrained(f"{model_name}-product_merged_model")
