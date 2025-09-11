import pandas as pd
from datasets import Dataset

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

# Pandas DataFrame 생성
df = pd.DataFrame({
    "english": ["I heard something creeping in the dark."],
    "korean": ["어둠 속에서 무언가가 기어오는 소리가 들렸다."]
})

# Dataset.from_pandas() 사용
raw_dataset = Dataset.from_pandas(df)

print(raw_dataset)

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


print("raw_dataset", type(raw_dataset))
print("raw_dataset", raw_dataset[:1])
#.map()을 사용하여 전체 데이터셋에 프롬프트 형식 적용
formatted_dataset = raw_dataset.map(create_translation_prompt, num_proc=4, remove_columns=raw_dataset.column_names)
#formatted_dataset = formatted_dataset.shuffle().select(range(1500))
print("Dataset formatted.")
#print(f"Sample formatted prompt:\n{formatted_dataset['text']}")
#print(f"Sample formatted prompt:\n{formatted_dataset['text'][0]}")

print(formatted_dataset[0])