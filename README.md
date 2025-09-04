# Gemma 3 Fine-tuning in GCE + GPU, TPU, Cloud Run + GPU, GKE + TPU

대규모 언어 모델(Large Language Models, LLM)은 수천억 개의 파라미터를 가진 파운데이션 모델(Foundation Model)으로 방대한 양의 텍스트 데이터로 사전 학습(pre-training)되어 광범위한 언어적 지식과 추론 능력을 갖춘 거대 모델을 기반이다. 
Fine tuning은 이를 특정 목적에 맞게 미세하게 조정하는 방식을 채택하고 있다. 

이 새로운 패러다임의 중심에는 지도 미세조정(Supervised Fine-Tuning, SFT) 이라는 강력한 기술이 자리 잡고 있다. 
예를 들어, 일반적인 대화가 가능한 LLM을 법률 문서 분석, 의료 상담, 혹은 특정 기업의 내부 정책에 대한 질의응답 시스템으로 변모시키는 과정의 핵심이 바로 SFT이다. 
이 과정을 통해 모델의 응답 정확성과 신뢰도를 높이고, 사용자의 구체적인 요구사항을 충족시킴으로써 LLM의 실질적인 가치를 극대화할 수 있다.

##### SFT (Supervised Fine-Tuning)

지도 미세조정(Supervised Fine-Tuning, SFT)은 이미 방대한 양의 비정형 텍스트 데이터로 사전 학습(pre-trained)을 마친 언어 모델을, 특정 작업(task)에 특화시키기 위해 레이블이 지정된(labeled) 데이터셋으로 추가 학습을 진행하는 지도 학습(supervised learning) 기반의 최적화 기법이다.
사전 학습을 통해 모델이 언어의 일반적인 문법, 의미, 문맥, 그리고 세상에 대한 상식까지 습득했다면,
SFT는 이 지식을 바탕으로 특정 질문에 어떻게 대답해야 하는지, 혹은 특정 지시를 어떻게 수행해야 하는지 구체적인 '행동 양식'을 가르치는 과정이라 할 수 있다.

#### Gemma 3

Gemma는 생성형 인공지능 (AI) 모델 제품군으로, 질문 답변, 요약, 추론을 비롯한 다양한 생성 작업에 사용할 수 있습니다. Gemma 모델은 개방형 가중치로 제공되며 책임감 있는 상업적 사용을 허용하므로 자체 프로젝트 및 애플리케이션에서 모델을 조정하고 배포할 수 있습니다.

Google의 Gemma 3는 Gemini 모델군과 동일한 연구 및 기술을 기반으로 구축된 최첨단 경량 오픈 모델입니다. 이 모델은 텍스트와 이미지를 모두 처리할 수 있는 멀티모달 기능을 갖추고 있으며, 본질적으로는 이전 토큰을 기반으로 다음 토큰을 예측하는 디코더-온리(decoder-only) 트랜스포머 아키텍처를 따릅니다. 특히 140개 이상의 언어를 지원하는 다국어 능력은 번역과 같은 언어 간 변환 작업에 강력한 초기 기반을 제공합니다.

- [Gemma 3](https://huggingface.co/docs/transformers/main/model_doc/gemma3)
- [Gemma 3 모델 개요](https://ai.google.dev/gemma/docs/core?hl=ko)

## Environments

#### Hugging Face Token

Create .env file and add Hugging Face Token

```
HF_TOKEN=YOUR_HUGGING_FACE_TOKEN
```

또는 Secret Manager에 등록

```bash
export HF_TOKEN=sayouzone-huggingface-token:latest
```

gcloud run deploy에 옵션 추가

```bash
    --update-secrets=HF_TOKEN=$HF_TOKEN
```

## Datasets

####

## GCE + GPU

PyTorch + Hugging Face

Fine-tuning
Serving

[GCE](https://github.com/sayouzone/gemma3-sft/tree/main/gce)

## TPU

[TPU](https://github.com/sayouzone/gemma3-sft/tree/main/tpu)

## Cloud Run + GPU

대규모 언어 모델(LLM)에 중점을 두고 AI 추론용 GPU와 함께 Cloud Run 서비스를 사용할 때 성능을 최적화하기 위한 권장사항
- 빠르게 로드되고 GPU 지원 구조로 최소한의 변환이 필요한 모델을 사용하고 로드 방법을 최적화
- 최대의 효율적인 동시 실행을 허용하는 구성을 사용하면 비용을 낮추면서 초당 목표 요청을 처리하는 데 필요한 GPU 수를 줄임

NVIDIA L4 GPU에 24GB의 GPU 메모리(VRAM)를 사용
asia-southeast1(싱가포르), asia-south1(뭄바이), europe-west1(벨기에), europe-west4(네덜란드), us-central1(아이오와), us-east4(북 버지니아)

**모델 다운로드**
- 사전 빌드된 컨테이너 사용
  - --image us-docker.pkg.dev/cloudrun/container/gemma/gemma3-1b 사용
  - gemma3-1b, gemma3-4b, gemma3-12b, gemma3-27b
- 컨테이너 이미지에 모델 저장
  - 컨테이너 빌드 시간이 오래 걸림
- Cloud Storage에 모델 저장
  - Cloud Storage 볼륨 마운트
  - Cloud Storage API 또는 명령줄을 직접 사용
  - ML 모델 로드를 최적화: 비공개 Google 액세스와 함께 이그레스 설정 값을 all-traffic으로 설정
- 인터넷에서 모델 로드
  - ML 모델 로드를 최적화: 이그레스 설정 값을 all-traffic으로 설정 (?)

**IAM 역할 부여**
- Cloud Run 관리자(roles/run.admin)
- 프로젝트 IAM 관리자(roles/resourcemanager.projectIamAdmin)
- 서비스 사용량 소비자(roles/serviceusage.serviceUsageConsumer)

**참조**
- [Cloud Run](https://github.com/sayouzone/gemma3-sft/tree/main/cloudrun)
- [Cloud Run에서 Gemma 3 실행](https://cloud.google.com/run/docs/run-gemma-on-cloud-run?hl=ko)
- [Deploy Gemma 3 to Cloud Run with Google AI Studio](https://ai.google.dev/gemma/docs/core/deploy_to_cloud_run_from_ai_studio)

## MacBook Pro M4

Apple의 M4 Pro 칩은 TSMC의 2세대 3나노미터 공정 기술을 기반으로 제작되었으며, CPU, GPU, Neural Engine 및 고대역폭 메모리 서브시스템을 단일 칩에 통합하여 전력 효율성과 성능을 극대화한다.

PyTorch가 Metal Performance Shaders(MPS) 백엔드를 통해 GPU 가속을 지원한다는 점입니다. MPS는 각 Metal GPU 제품군의 고유한 특성에 맞게 미세 조정된 커널을 통해 연산 성능을 최적화합니다. 여기서 핵심은 통합 메모리 아키텍처(Unified Memory Architecture)입니다. 이 구조는 CPU와 GPU가 동일한 메모리 풀에 직접 접근하게 하여, 데이터 검색 지연 시간을 획기적으로 줄입니다. 이론적으로 이는 동일한 VRAM 용량을 가진 외장 GPU 시스템보다 더 큰 배치 사이즈나 모델을 처리할 수 있게 하는 중요한 하드웨어적 이점입니다.

MLX는 단순한 "Mac용 PyTorch"가 아니다. 지연 연산과 통합 메모리라는 근본적으로 다른 설계 원칙을 통해, 개별 GPU 아키텍처를 위해 설계된 프레임워크에서는 구조적으로 구현하기 어려운 최적화를 가능하게 한다.
mlx-lm 툴킷은 이러한 강력하고 최적화된 코어 위에 사용자 친화적인 추상화 계층을 제공하여, 복잡한 LLM 미세조정 작업을 몇 줄의 명령어로 수행할 수 있게 해준다.

- 통합 메모리 아키텍처(Unified Memory Architecture, UMA)
- PyTorch가 Metal Performance Shaders(MPS) 백엔드를 통해 GPU 가속을 지원
- Apple MLX 프레임워크

- [MacBook Pro M4](https://github.com/sayouzone/gemma3-sft/tree/main/m4)
- [Apple MLX](https://towardsdatascience.com/deploying-llms-locally-with-apples-mlx-framework-2b3862049a93/)

## GKE + TPU

## Tests