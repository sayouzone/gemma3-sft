import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments

model_name = "gemma-3-1b"
merged_model_path = f"{model_name}-en-ko-trans_merged_model"

# 병합된 모델 로드
loaded_model = AutoModelForCausalLM.from_pretrained(
    merged_model_path,
    attn_implementation="eager", # Use "flash_attention_2" when running on Ampere or newer GPU
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
loaded_tokenizer = AutoTokenizer.from_pretrained(merged_model_path)

def translate_en_to_ko(model, tokenizer, text):
    prompt = f"""### Instruction:
Translate the following text from English to Korean.

### Input:
{text}
 
### Response:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    #print("inputs", inputs)
    outputs = model.generate(**inputs, max_new_tokens=len(text) * 3, eos_token_id=tokenizer.eos_token_id)
    #print("outputs", outputs)
    #result = tokenizer.decode(outputs, skip_special_tokens=True)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("result", result)
    response_part = result.split("### Response:")
    return response_part

# 테스트 문장
test_sentence = "The ability to fine-tune powerful language models on consumer hardware is a significant breakthrough for the AI community."
translation = translate_en_to_ko(loaded_model, loaded_tokenizer, test_sentence)

print(f"English Input:\n{test_sentence}")
print(f"\nKorean Translation:\n{translation}")
