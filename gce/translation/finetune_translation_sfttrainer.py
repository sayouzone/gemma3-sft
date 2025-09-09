import os
from dotenv import load_dotenv

import torch
from PIL import Image

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer

# .env 파일 로드
load_dotenv()

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

# 환경 변수 설정
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# System message for the assistant
system_message = "You are an English to Korean professional translator."
#system_message = "당신은 영어를 한국어로 번역하는 전문가입니다."

# User prompt that combines the user query and the schema
# 모델이 학습한 명령어 형식을 따르는 프롬프트 템플릿
user_prompt = """<start_of_turn>user
Translate the following English sentence into Korean.

English: "{english_sentence}"
Korean: "{korean_sentence}"
<end_of_turn>
"""
#user_prompt = "<s> 다음 영문 텍스트를 한국어로 번역하세요: {english_sentence} {korean_sentence}</s>"

# [1, 24]의 alpaca_prompt 형식을 번역 작업에 맞게 변형한 예시
def prompt_translation(data):
    # 'translation' 딕셔너리에서 'en'과 'ko' 키로 문장을 가져옵니다.
    # 데이터셋 구조에 따라 키 이름은 달라질 수 있습니다. (예: example['en'], example['ko'])
    #english_sentence = data['translation']['en']
    #korean_sentence = data['translation']['ko']
    english_sentence = data['english']
    korean_sentence = data['korean']

    return {"text": user_prompt.format(
        english_sentence=english_sentence,
        korean_sentence=korean_sentence)
    }

def load_translation_dataset(dataset_name):
    # Load dataset from the hub
    # 데이터셋 로드
    dataset = load_dataset(dataset_name, split="train")

    # Convert dataset to OAI messages
    # need to use list comprehension to keep Pil.Image type, .mape convert image to bytes
    # 데이터셋 형식 변환
    formatted_dataset = dataset.map(prompt_translation, remove_columns=dataset.column_names)

    # 데이터셋 섞기 및 분할 (95% 훈련, 5% 검증)
    shuffled_dataset = formatted_dataset.shuffle(seed=42)
    split_dataset = shuffled_dataset.train_test_split(test_size=0.05)

    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    # 결과 확인
    print(f"훈련 데이터셋 크기: {len(train_dataset)}")
    print(f"검증 데이터셋 크기: {len(eval_dataset)}")
    print("\n변환된 데이터 샘플:")
    print(train_dataset['text'][5])

    return train_dataset, eval_dataset

dataset_name = "lemon-mint/korean_english_parallel_wiki_augmented_v1"
train_dataset, eval_dataset = load_translation_dataset(dataset_name)

model_name = "gemma-3-4b"
# Hugging Face model id
model_id = f"google/{model_name}-pt" # or `google/gemma-3-12b-pt`, `google/gemma-3-27b-pt`

# Check if GPU benefits from bfloat16
if torch.cuda.get_device_capability()[0] < 8:
    raise ValueError("GPU does not support bfloat16, please use a GPU that supports bfloat16.")

# Define model init arguments
model_kwargs = dict(
    attn_implementation="eager", # Use "flash_attention_2" when running on Ampere or newer GPU
    torch_dtype=torch.bfloat16, # What torch dtype to use, defaults to auto
    #load_in_8bit=True, # or load_in_4bit=True
    device_map="auto", # Let torch decide how to load the model
)

"""
BitsAndBytesConfig 설정: 4비트 양자화를 활성화하고 세부 옵션을 지정합니다.

load_in_4bit=True: QLoRA의 핵심으로, 모델 가중치를 4비트로 로드합니다.
bnb_4bit_quant_type="nf4": 양자화 방식으로, 일반적으로 'nf4'가 좋은 성능을 보입니다.
bnb_4bit_compute_dtype=torch.bfloat16: 계산 정밀도를 지정합니다. A100과 같은 최신 GPU는 bfloat16을 지원하여 학습 안정성을 높이지만, T4와 같은 구형 GPU에서는 torch.float16으로 대체해야 합니다.15
bnb_4bit_use_double_quant=True: 양자화 상수를 다시 양자화하여 약간의 메모리를 추가로 절약합니다.
"""

# BitsAndBytesConfig int-4 config
# QLoRA를 위한 4비트 양자화 설정
model_kwargs["quantization_config"] = BitsAndBytesConfig(
    load_in_4bit=True,                                  # 4비트 양자화 활성화
    bnb_4bit_use_double_quant=True,                     # 이중 양자화를 통해 메모리 추가 절약
    bnb_4bit_quant_type="nf4",                          # NF4(Normal Float 4) 양자화 타입 사용 (고성능)
    bnb_4bit_compute_dtype=model_kwargs["torch_dtype"], # 계산 시 사용할 데이터 타입 (bfloat16 권장)
    bnb_4bit_quant_storage=model_kwargs["torch_dtype"],
)

print("model_id", model_id)
# Load model and tokenizer
# 토크나이저 로드
# Instruction-Tuned 모델의 공식 템플릿을 사용하기 위해 'it' 버전 토크나이저를 로드할 수 있습니다.
model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
#tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it") # Load the Instruction Tokenizer to use the official Gemma template
tokenizer = AutoTokenizer.from_pretrained(f"google/{model_name}-it") # Load the Instruction Tokenizer to use the official Gemma template

# 모델 로드 (이후 QLoRA 설정을 위해 quantization_config를 함께 전달)
# 이 단계에서는 모델의 기본 구조만 확인하고, 실제 로드는 QLoRA 설정과 함께 진행됩니다.

"""
r: LoRA 어댑터의 저차원 행렬의 랭크(rank)입니다. 값이 클수록 더 많은 파라미터를 학습하여 표현력이 높아지지만, 계산 비용도 증가합니다. 일반적으로 8, 16, 32와 같은 값으로 시작하는 것이 좋습니다.13
lora_alpha: LoRA 활성화를 스케일링하는 역할을 합니다. 2 * r로 설정하는 것이 일반적인 휴리스틱입니다.16
target_modules: LoRA 어댑터를 적용할 모델 내의 특정 레이어(모듈)를 지정합니다. Gemma 모델의 경우, 어텐션 블록의 쿼리, 키, 값, 출력 프로젝션 레이어(q_proj, k_proj, v_proj, o_proj)를 대상으로 하는 것이 일반적입니다.12
"all-linear"로 설정하여 모든 선형 레이어를 대상으로 할 수도 있습니다.15
task_type="CAUSAL_LM": PEFT에 이 모델이 인과적 언어 모델(Causal Language Model)임을 알려줍니다.
"""

# LoRA 어댑터 설정
lora_config = LoraConfig(
    lora_alpha=32,  # LoRA 스케일링 알파 값 (값: 16)
    lora_dropout=0.05,  # LoRA 레이어에 적용할 드롭아웃 비율
    r=16,  # LoRA 행렬의 rank (attention dimension)
    bias="none",  # bias는 학습하지 않음
    task_type="CAUSAL_LM",  # 작업 유형 지정
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # LoRA를 적용할 모듈 지정
    #modules_to_save=["lm_head", "embed_tokens"] # make sure to save the lm_head and embed_tokens as you train the special tokens
)

args = SFTConfig(
    output_dir=f"{model_name}-en-ko-trans",     # directory to save and repository id
                                                # 모델 출력 및 저장 디렉토리
    num_train_epochs=1,                         # number of training epochs
                                                # 총 학습 에폭 수
    per_device_train_batch_size=1,              # batch size per device during training
    #per_device_train_batch_size=4,              # 디바이스당 학습 배치 크기
    gradient_accumulation_steps=4,              # number of steps before performing a backward/update pass
                                                # 그래디언트 축적 스텝 (메모리 부족 시 배치 크기를 늘리는 효과)
    gradient_checkpointing=True,                # use gradient checkpointing to save memory
    optim="adamw_torch_fused",                  # use fused adamw optimizer
    #optim="paged_adamw_8bit",                   # 메모리 효율적인 AdamW 옵티마이저
    logging_steps=5,                            # log every 5 steps
                                                # 로그 출력 주기
    save_strategy="epoch",                      # save checkpoint every epoch
                                                # 저장 전략
    learning_rate=2e-4,                         # learning rate, based on QLoRA paper
                                                # 학습률
    bf16=True,                                  # use bfloat16 precision
                                                # bf16 사용 여부 (A100 등 지원 GPU에서만 가능)
    #fp16=False,                                 # fp16 사용 여부 (bfloat16 사용 시 False)
    max_grad_norm=0.3,                          # max gradient norm based on QLoRA paper
                                                # 그래디언트 클리핑
    warmup_ratio=0.03,                          # warmup ratio based on QLoRA paper
                                                # 웜업 비율
    lr_scheduler_type="constant",               # use constant learning rate scheduler
    #lr_scheduler_type="cosine",                 # 학습률 스케줄러 타입
    push_to_hub=True,                           # push model to hub
    report_to="tensorboard",                    # report metrics to tensorboard
    gradient_checkpointing_kwargs={
        "use_reentrant": False
    },  # use reentrant checkpointing
    dataset_text_field="text",                  # need a dummy field for collator
    dataset_kwargs={"skip_prepare_dataset": True},  # important for collator
    #evaluation_strategy="epoch",                # 평가 전략
    group_by_length=True,                       # 비슷한 길이의 샘플을 묶어 패딩 최소화
)
args.remove_unused_columns = False # important for collator

# SFTTrainer 초기화
trainer = SFTTrainer(
    model=model,
    args=args,
    #train_dataset=dataset,                 # 포맷팅된 학습 데이터셋
    train_dataset=train_dataset,           # 포맷팅된 학습 데이터셋
    eval_dataset=eval_dataset,             # 포맷팅된 검증 데이터셋
    peft_config=lora_config,
    #dataset_text_field="text",             # 데이터셋에서 텍스트 필드 이름
    #max_seq_length=1024,                   # 최대 시퀀스 길이
    #max_length=1024,                       # 최대 시퀀스 길이
    tokenizer=tokenizer,
    #packing=False,                         # 여러 샘플을 하나의 시퀀스로 묶을지 여부
    #processing_class=processor,
    #data_collator=collate_fn,
)

# Start training, the model will be automatically saved to the Hub and the output directory
# 미세 조정 시작
trainer.train()

# Save the final model again to the Hugging Face Hub
# 학습된 LoRA 어댑터 저장
trainer.save_model()


"""
어댑터 가중치 병합
QLoRA를 사용하면 전체 모델이 아닌 어댑터만 학습
학습 중에 모델을 저장할 때는 전체 모델이 아닌 어댑터 가중치만 저장
vLLM 또는 TGI와 같은 서빙 스택에서 더 쉽게 사용할 수 있도록 전체 모델을 저장하려면 merge_and_unload 메서드를 사용하여 어댑터 가중치를 모델 가중치에 병합한 다음 save_pretrained 메서드로 모델을 저장
"""

# Load Model base model
model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True)

# Merge LoRA and base model and save
peft_model = PeftModel.from_pretrained(model, args.output_dir)
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained(f"{model_name}-trans_merged_model", safe_serialization=True, max_shard_size="2GB")

processor = AutoTokenizer.from_pretrained(args.output_dir)
processor.save_pretrained(f"{model_name}-trans_merged_model")
