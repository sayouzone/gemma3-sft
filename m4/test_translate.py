import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, QuantoConfig

"""
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
"""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def translate_english_to_korean(text_to_translate: str, model, tokenizer) -> str:
    """
    주어진 영어 텍스트를 한국어로 번역합니다.

    Args:
        text_to_translate (str): 번역할 영어 문장.
        model: 로드된 언어 모델.
        tokenizer: 로드된 토크나이저.

    Returns:
        str: 번역된 한국어 문장.
    """
    # Gemma 3의 공식 지시(instruction) 템플릿 형식에 맞춰 프롬프트를 구성합니다.
    # 이 템플릿은 모델이 번역 작업을 명확히 인지하고 더 좋은 결과를 생성하도록 돕습니다.
    messages = [
        {
            "role": "user",
            "content": f"Translate the following English text to Korean: \"{text_to_translate}\""
        },
    ]

    # 템플릿을 적용하고 토큰화합니다.
    # .to(model.device)를 통해 입력 데이터를 모델이 있는 장치(CPU, MPS, CUDA 등)로 보냅니다.
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    # 그래디언트 계산을 비활성화하여 추론 속도를 높이고 메모리 사용량을 줄입니다.
    with torch.no_grad():
        # 모델을 사용하여 새로운 토큰을 생성합니다.
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=100,  # 번역 결과의 최대 길이를 설정합니다.
            eos_token_id=tokenizer.eos_token_id
        )

    #print(outputs)

    # 생성된 결과에서 입력 프롬프트 부분을 제외하고 순수 응답 부분만 디코딩합니다.
    response_ids = outputs[0][input_ids.shape[-1]:]
    translated_text = tokenizer.decode(response_ids, skip_special_tokens=True)

    return translated_text.strip()

def main():
    """
    메인 실행 함수: 모델을 로드하고 번역을 수행합니다.
    """
    # 1. 모델 및 토크나이저 설정
    model_id = "google/gemma-3-1b-it"
    #model_id = "google/gemma-3-4b-it"
    #model_id = "google/gemma-3-12b-it"
    #model_id = "google/gemma-3-27b-it"
    print(f"'{model_id}' 모델을 로드합니다. 잠시 기다려 주세요...")

    # 4비트 양자화 설정: 모델을 더 가볍게 만들어 메모리 사용량을 줄입니다.
    quantization_config = QuantoConfig(weights="int4")
    
    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,        # 모델의 데이터 타입을 설정.
        device_map="auto",                 # 사용 가능한 장치(MPS, CUDA, CPU)를 자동으로 할당.
        #device_map="mps",                 # 장치를 MPS 설정
        attn_implementation="eager"        # Gemma 3 모델의 안정성을 위한 권장 설정.
    )

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    print("모델 로드가 완료되었습니다.\n")

    # 2. 번역할 문장 정의
    english_sentence = "The current time is Wednesday, August 13, 2025 at 9:34 AM in Seoul, South Korea."

    # 3. 번역 실행 및 결과 출력
    print(f"입력 (영어): {english_sentence}")
    
    korean_translation = translate_english_to_korean(english_sentence, model, tokenizer)
    
    print(f"출력 (한국어): {korean_translation}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"총 소요 시간: {end_time - start_time:.2f} 초")


"""
'google/gemma-3-1b-it' 모델을 로드합니다. 잠시 기다려 주세요...
모델 로드가 완료되었습니다.

입력 (영어): The current time is Wednesday, August 13, 2025 at 9:34 AM in Seoul, South Korea.
출력 (한국어): Here's a translation of the text into Korean:

**오늘 (Oneul) - Wednesday**
**8월 13일 (Eollmay 13il) - August 13th**
**9:34 AM (Septu-myeok (am) 9:34) - 9:34 AM**
**서울 (Seoul) - Seoul**
**한국 (Hanguk) - South Korea**

**Complete
"""