import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments

model_name = "gemma-3-1b"
merged_model_path = f"{model_name}-en-ko-trans_merged_model_genre"

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

def translate_genre_en_to_ko(model, tokenizer, genre, text):
    prompt = f"""### Instruction:
Translate the following text from English to Korean as {genre} genre.

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
    response_part = result.split("### Response:")
    return response_part

# 테스트 문장
#test_sentence = "The ability to fine-tune powerful language models on consumer hardware is a significant breakthrough for the AI community."
#translation = translate_en_to_ko(loaded_model, loaded_tokenizer, test_sentence)
test_sentence = """You will remember there was no road—not even a pathway—between the castle of the Wicked Witch and the Emerald City. When the four travelers went in search of the Witch she had seen them coming, and so sent the Winged Monkeys to bring them to her. It was much harder to find their way back through the big fields of buttercups and yellow daisies than it was being carried. They knew, of course, they must go straight east, toward the rising sun; and they started off in the right way. But at noon, when the sun was over their heads, they did not know which was east and which was west, and that was the reason they were lost in the great fields. They kept on walking, however, and at night the moon came out and shone brightly. So they lay down among the sweet smelling yellow flowers and slept soundly until morning—all but the Scarecrow and the Tin Woodman."""
translation = translate_genre_en_to_ko(loaded_model, loaded_tokenizer, "fantasy", test_sentence)

print(f"English Input:\n{test_sentence}")
print(f"\nKorean Translation:\n{translation}")
