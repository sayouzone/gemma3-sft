import os
import torch
from random import randint
import requests
from PIL import Image

from datasets import load_dataset
from transformers import pipeline
from transformers import AutoProcessor, AutoModelForCausalLM, AutoModelForImageTextToText, BitsAndBytesConfig
from trl import SFTConfig

# 모델 추론 테스트 및 제품 설명 생성
"""
학습이 완료되면 모델을 평가하고 테스트해야 한다.
테스트 데이터 세트에서 다양한 샘플을 로드하고 이러한 샘플에서 모델을 평가할 수 있다.

https://ai.google.dev/gemma/docs/core/huggingface_vision_finetune_qlora?hl=ko
"""

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
model_id = f"google/{model_name}-pt" # or `google/gemma-3-12b-pt``, `google/gemma-3-27b-pt`
print("model_id", model_id)

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

#model_id = args.output_dir
model_id = f"{model_name}_merged_model"

# Load Model with PEFT adapter
model = AutoModelForImageTextToText.from_pretrained(
  args.output_dir,
  device_map="auto",
  torch_dtype=torch.bfloat16,
  attn_implementation="eager",
)
processor = AutoProcessor.from_pretrained(args.output_dir)


# Test sample with Product Name, Category and Image
sample = {
  "product_name": "Hasbro Marvel Avengers-Serie Marvel Assemble Titan-Held, Iron Man, 30,5 cm Actionfigur",
  "category": "Toys & Games | Toy Figures & Playsets | Action Figures",
  "image": Image.open(requests.get("https://m.media-amazon.com/images/I/81+7Up7IWyL._AC_SY300_SX300_.jpg", stream=True).raw).convert("RGB")
}

def generate_description(sample, model, processor):
    # Convert sample into messages and then apply the chat template
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {"role": "user", "content": [
            {"type": "image","image": sample["image"]},
            {"type": "text", "text": user_prompt.format(product=sample["product_name"], category=sample["category"])},
        ]},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # Process the image and text
    image_inputs = process_vision_info(messages)
    # Tokenize the text and process the images
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    # Move the inputs to the device
    inputs = inputs.to(model.device)

    # Generate the output
    stop_token_ids = [processor.tokenizer.eos_token_id, processor.tokenizer.convert_tokens_to_ids("<end_of_turn>")]
    generated_ids = model.generate(**inputs, max_new_tokens=256, top_p=1.0, do_sample=True, temperature=0.8, eos_token_id=stop_token_ids, disable_compile=True)
    # Trim the generation and decode the output to text
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

# generate the description
description = generate_description(sample, model, processor)
print(description)
