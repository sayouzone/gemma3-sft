from datasets import load_dataset
from PIL import Image

"""
미세 조정 데이터 세트

제품 이미지 및 카테고리를 포함한 Amazon 제품 설명 데이터
philschmid/amazon-product-descriptions-vlm 데이터 세트
Hugging Face TRL은 멀티모달 대화를 지원

Hugging Face Datasets 라이브러리를 사용하여 데이터 세트를 로드하고 프롬프트 템플릿을 만들어 이미지, 제품 이름, 카테고리를 결합하고 시스템 메시지를 추가.
데이터 세트에는 이미지가 Pil.Image 객체로 포함

https://ai.google.dev/gemma/docs/core/huggingface_vision_finetune_qlora?hl=ko
"""

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