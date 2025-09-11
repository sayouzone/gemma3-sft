from datasets import load_dataset

import os
from dotenv import load_dotenv
from huggingface_hub import login

"""
데이터 로딩: load_dataset 함수를 사용하여 Hugging Face Hub에서 데이터셋을 로드합니다.23
전처리 워크플로우:
1. 셔플(Shuffle): 데이터셋을 무작위로 섞어 학습 데이터의 편향을 방지합니다 (dataset.shuffle()).
2. 분할(Split): 과적합(overfitting)을 모니터링하기 위해 전체 데이터셋을 학습(train) 및 검증(validation) 세트로 분할합니다 (dataset.train_test_split()).
https://colab.research.google.com/github/google-health/medgemma/blob/main/notebooks/fine_tune_with_hugging_face.ipynb
3. 포맷팅 적용: .map() 함수를 사용하여 위에서 정의한 create_translation_prompt 함수를 데이터셋의 모든 샘플에 일괄적으로 적용하여, 모델이 요구하는 형식으로 변환합니다.
https://colab.research.google.com/github/google-gemini/gemma-cookbook/blob/main/Gemma/[Gemma_2]Finetune_with_Unsloth.ipynb
4. 토크나이징: 실제 토크나이징 과정은 이후 단계의 SFTTrainer가 내부적으로 처리하지만, 이 단계에서 포맷팅된 텍스트가 모델의 입력 ID로 변환된다는 점을 이해하는 것이 중요합니다.
"""
# --- 1. 기본 설정 및 인증 ---
# .env 파일 로드
load_dotenv()

# 환경 변수 설정
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

def login_huggingface(hf_token):
    #os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
    if hf_token:
        # Hugging Face Hub에 로그인합니다.
        login(token=hf_token)
        print("Successfully logged in to Hugging Face Hub!")
    else:
        print("HF_TOKEN environment variable not found.")

# --- 3. 데이터셋 준비 및 형식화 ---
dataset_name = "lemon-mint/korean_parallel_sentences_v1.1"
print(f"Loading and preparing dataset '{dataset_name}'...")

raw_dataset = load_dataset(dataset_name, split="train")

# 훈련 시간을 줄이기 위해 작은 서브셋으로 테스트 (실제 훈련 시에는 전체 데이터 사용)
# raw_dataset = raw_dataset.shuffle(seed=42).select(range(10000))

def create_translation_prompt(data):
    """데이터셋 샘플을 명령어 형식의 프롬프트로 변환하는 함수"""
    user_prompt = """### Instruction:
Translate the following text from English to Korean.

### Input:
{english_sentence}

### Response:
{korean_sentence}"""

    return {
        "text": user_prompt.format(english_sentence=data['english'], korean_sentence=data['korean'])
    }

# [1, 24]의 alpaca_prompt 형식을 번역 작업에 맞게 변형한 예시
def prompt_translation(data):
    # 'translation' 딕셔너리에서 'en'과 'ko' 키로 문장을 가져옵니다.
    # 데이터셋 구조에 따라 키 이름은 달라질 수 있습니다. (예: data['en'], data['ko'])
    english_sentence = data['translation']['en']
    korean_sentence = data['translation']['ko']

    # 모델이 학습한 명령어 형식을 따르는 프롬프트 템플릿
    prompt = f"""<start_of_turn>user
Translate the following English sentence into Korean.
English: "{english_sentence}"
"""
    # 내용 추가 필요
    pass

print("raw_dataset", type(raw_dataset))
print("raw_dataset", raw_dataset[:5])
#.map()을 사용하여 전체 데이터셋에 프롬프트 형식 적용
formatted_dataset = raw_dataset.map(create_translation_prompt, num_proc=4, remove_columns=raw_dataset.column_names)
#formatted_dataset = formatted_dataset.shuffle().select(range(1500))
print("Dataset formatted.")
#print(f"Sample formatted prompt:\n{formatted_dataset['text']}")
print(f"Sample formatted prompt:\n{formatted_dataset['text'][5]}")

print(formatted_dataset[345])

"""
Loading and preparing dataset 'lemon-mint/korean_parallel_sentences_v1.1'...
Using the latest cached version of the dataset since lemon-mint/korean_parallel_sentences_v1.1 couldn't be found on the Hugging Face Hub
Found the latest cached dataset configuration 'default' at /Users/seongjungkim/.cache/huggingface/datasets/lemon-mint___korean_parallel_sentences_v1.1/default/0.0.0/c3ffa5bfe5bf0cd5b4d634e863978b5eb265c9e1 (last modified on Wed Sep 10 11:50:52 2025).

raw_dataset 
{
  'korean': [
    '무두족류는 머리가 없고, 대신 몸통에 직접 붙어 있는 발달된 발이 있습니다.', 
    '프랑스와 미국의 관계는 오랜 역사를 가지고 있으며, 긴밀한 협력과 때로는 갈등으로 특징지어집니다. 두 나라는 미국 독립 전쟁에서 동맹국이었으며, 그 이후로도 세계 정치에서 긴밀한 협력 관계를 유지해 왔습니다. 그러나 이라크 전쟁과 같은 문제에 대해서는 의견 차이가 있었습니다.', 
    '마을 사람들은 마을 축제를 준비하며 마을 거리를 장식하고 있었습니다.', 
    '뱀은 길고 가늘며, 비늘로 덮여 있습니다. 뱀은 독이 있거나 독이 없을 수 있습니다. 독이 있는 뱀은 먹이를 죽이거나 자신을 방어하기 위해 독을 사용합니다.', 
    '새 신발은 검은색 가죽으로 만들어졌고, 매우 편안했다. 신발을 신고 밖을 산책하니 마음이 상쾌했다.'
  ], 
  'english': [
    'Acephala have no head and instead have well-developed feet that are directly attached to the torso.', 
    'France and the United States have a long and complex relationship, marked by both close cooperation and occasional conflict. The two countries were allies during the American Revolutionary War, and they have maintained a close partnership in world affairs ever since. However, they have also had disagreements, such as over the Iraq War.', 
    'The villagers were decorating the village streets in preparation for the village festival.', 
    'Ophidia are elongated, legless, carnivorous reptiles covered in scales. Snakes can be venomous or nonvenomous. Venomous snakes use their venom to kill prey or defend themselves.', 
    'The new shoes were made of black leather and were very comfortable. I felt refreshed when I put on my shoes and went for a walk outside.'
  ]
}

Dataset formatted.
Sample formatted prompt:
### Instruction:
Translate the following text from English to Korean.

### Input:
The uterus is located below the belly button, and its size and shape vary from woman to woman. The uterus is usually pear-shaped and about 7~8 cm in length.

### Response:
자궁은 배꼽 아래에 위치하며, 크기와 모양은 여성마다 다릅니다. 자궁은 일반적으로 배 모양이며, 길이는 약 7~8cm입니다.

{'text': '### Instruction:\nTranslate the following text from English to Korean.\n\n### Input:\nI heard something creeping in the dark.\n\n### Response:\n어둠 속에서 무언가가 기어오는 소리가 들렸다.'}
"""