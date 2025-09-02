# Google Compute Engine에서 TPU를 사용하여 한글-영어 번역에 대한 Fine-tuning을 진행

Fine-tuning으로 번역을 적용하면 좀 더 섬세하게 번역되는지 확인<br>
Fine-tuning 시간 단축<br>
서비스 비용 최소화 등 검증

## 요약

- 리전, TPU 유형, TPU 소프트웨어 버전 선택에 대한 고민
- 패키지 설치 및 Optimum TPU 적용
- [TPU Pods Gemma3 파인튜닝 정리](https://blog.worldsw.dev/tpu-pods-gemma3-finetune/) 시도
  - "Model {model} configuration cannot be auto-generated, use get_fsdp_config instead." 오류 발생
- [🤗 Optimum TPU](https://huggingface.co/docs/optimum-tpu/index) 시도
  - 패키지 호환 오류 발생

Gemma3 팀: gemma-3-report@google.com

## Gemma 3

Gemma는 Tensor Processing Unit (TPU) 하드웨어 (TPUv4p, TPUv5p, TPUv5e)를 사용하여 학습되었습니다. 비전 언어 모델 (VLM)을 학습시키려면 상당한 컴퓨팅 성능이 필요합니다. 머신러닝에서 흔히 볼 수 있는 행렬 연산을 위해 특별히 설계된 TPU는 이 도메인에서 다음과 같은 여러 이점을 제공합니다.

[Gemma 3 모델 카드](https://ai.google.dev/gemma/docs/core/model_card_3?hl=ko)

IT (instruction-tuned models) - 명령어 조정 모델
PT (pre-trained models) - 사전 학습된 모델

#### 🤗 Optimum TPU

Optimum TPU provides all the necessary machinery to leverage and optimize AI workloads runningon [Google Cloud TPU devices](https://cloud.google.com/tpu/docs). Optimum-TPU is a HuggingFace solution to optimize HuggingFace products for the TPU platform.

Optimum-TPU serves as the bridge between the HuggingFace ecosystem and Google Cloud TPU hardware.

```bash
pip install optimum-tpu -f https://storage.googleapis.com/libtpu-releases/index.html
```

## Errors

####


```bash
ERROR: (gcloud.compute.tpus.tpu-vm.create) PERMISSION_DENIED: Permission denied on 'locations/us-central2-b' (or it may not exist). This command is authenticated as sjkim@sayouzone.com which is the active account specified by the [core/account] property.
Permission denied on 'locations/us-central2-b' (or it may not exist).
- '@type': type.googleapis.com/google.rpc.ErrorInfo
  domain: googleapis.com
  metadata:
    consumer: projects/sayouzone-ai
    location: us-central2-b
    service: tpu.googleapis.com
  reason: LOCATION_POLICY_VIOLATED
```

```bash
"Quota 'TPUV5sLitepodServingPerProjectPerZoneForTPUAPI' exhausted. Limit 4 in zone us-central1-a"
```

```bash
ERROR: (gcloud.compute.tpus.tpu-vm.create) INVALID_ARGUMENT: Cloud TPU received a bad request. The requested accelerator configuration for accelerator type, "v5litepod-8", could not be found in the zonal accelerator configurations for "us-central1-b".; the accelerator v5litepod-8 was not found in zone us-central1-b [EID: 0x5be9ca996f9773a4]
```


####

```bash
ERROR: Could not find a version that satisfies the requirement torch-xla[tpu]==2.5.1 (from optimum-tpu) (from versions: none)
ERROR: No matching distribution found for torch-xla[tpu]==2.5.1 (from optimum-tpu)
```