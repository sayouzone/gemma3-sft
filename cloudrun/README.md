# Cloud Run에서 Gemma 3 모델 실행

Cloud Run에서 Gemma 3 모델을 실행할 때 적합한 환경을 확인하는 것이 목적이다.<br>
Fine-tuning된 모델을 Hugging Face Hub에서 다운로드하거나,
Google Cloud Storage에서 다운로드하는 방법을 확인할 것이다.

환경에 따른 성능 비교

**Models**
- Pre-trained models
- Fine-tuned models

**진행사항**
- 빠르게 빌드하고 배포하는 스크립트를 작성
- .env으로 HF_TOKEN 관리
- Secret Manager으로 HF_TOKEN 관리
- Cloud Storage Volume Mount으로 FT 모델 다운로드
- Cloud Storage API Transfer Manager 동시 다운로드

## Pre-trained model 실행

[]()

#### Fine-tuned model 실행 (Hugging Face Hub)

HF_TOKEN 관리
- .env
- Secret Manager

[Hugging Face Hub](https://github.com/sayouzone/gemma3-sft/tree/main/cloudrun/gemma3-base)

#### Fine-tuned model 실행 (Google Cloud Storage Volume Mount)

[Cloud Storage Volume Mount](https://github.com/sayouzone/gemma3-sft/tree/main/cloudrun/gemma3-product-mount)

#### Fine-tuned model 실행 (Google Cloud Storage Transfer Manager)

[Cloud Storage Transfer Manager](https://github.com/sayouzone/gemma3-sft/tree/main/cloudrun/gemma3-product)
