# Gemma 3 fine-tuned model on Cloud Run using Google Cloud Storage

Hugging Face 모델을 Google Cloud Storage(GCS)에서 직접 로드하려면 model_id를 GCS 경로로 지정해야 합니다. 이렇게 하면 모델 파일을 로컬에 다운로드하지 않고 GCS에서 직접 스트리밍할 수 있습니다.

Google Cloud Storage(GCS) API 에서 Transfer Manager 동시 다운로드

- [Cloud Run에서 Gemma 3 실행](https://cloud.google.com/run/docs/run-gemma-on-cloud-run?hl=ko)
- [권장사항: GPU를 사용하는 Cloud Run의 AI 추론](https://cloud.google.com/run/docs/configuring/services/gpu-best-practices?hl=ko)
- [동시에 청크의 파일 다운로드](https://cloud.google.com/storage/docs/samples/storage-transfer-manager-download-chunks-concurrently?hl=ko)
- [슬라이스 객체 다운로드](https://cloud.google.com/storage/docs/sliced-object-downloads?hl=ko)

## 프로젝트 구조 설정

## 컨테이너 빌드 및 Artifact Registry에 푸시

**1. 변수 설정:**

GPU를 지원하는 리전 선택

```bash
export PROJECT_ID=$(gcloud config get-value project)
export REGION="us-central1"
export REPO_NAME="gemma3-repo"
export TRANS_SERVICE_NAME="gemma3-trans-ft-service"
export TRANS_IMAGE_TAG="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${TRANS_SERVICE_NAME}:latest"
export HF_TOKEN=sayouzone-huggingface-token:latest
```

**2. Artifact Registry 저장소 생성 (최초 한 번만 실행)::**

```bash
gcloud artifacts repositories create $REPO_NAME \
    --repository-format=docker \
    --location=$REGION
```

**3. Cloud Build를 사용하여 이미지 빌드 및 푸시:**

```bash
gcloud builds submit --tag $TRANS_IMAGE_TAG
```

## Cloud Run에 배포

빌드된 컨테이너 이미지를 사용하여 GPU가 장착된 Cloud Run 서비스에 배포합니다.

- --gpu=1: 서비스에 1개의 GPU를 할당.
- --gpu-type=nvidia-l4: 사용할 GPU 타입을 지정 (L4는 추론에 효율적이다).
- --cpu=4, --memory=16Gi: 충분한 CPU와 메모리를 할당. 모델 크기에 따라 조절이 필요.
- --concurrency=1: GPU를 사용하는 경우, 일반적으로 동시성(concurrency)을 1로 설정하여 한 번에 하나의 요청만 처리하도록 하는 것이 안정적이다.
- --no-cpu-throttling: 백그라운드 작업이 CPU에 의해 제한되지 않도록 한다.

```bash
gcloud run deploy $TRANS_SERVICE_NAME \
    --image=$TRANS_IMAGE_TAG \
    --region=$REGION \
    --gpu=1 \
    --gpu-type=nvidia-l4 \
    --cpu=8 \
    --memory=32Gi \
    --concurrency=1 \
    --no-cpu-throttling \
    --allow-unauthenticated \
    --timeout=30m
```

#### Startup Probe

```bash
gcloud run deploy $TRANS_SERVICE_NAME \
    --image=$TRANS_IMAGE_TAG \
    --region=$REGION \
    --gpu=1 \
    --gpu-type=nvidia-l4 \
    --cpu=8 \
    --memory=32Gi \
    --concurrency=1 \
    --no-cpu-throttling \
    --allow-unauthenticated \
    --startup-probe tcpSocket.port=8080,initialDelaySeconds=240,failureThreshold=5,timeoutSeconds=240,periodSeconds=240 \
    --timeout=30m
```

#### Delete Cloud Run Service

```bash
gcloud run services delete $TRANS_SERVICE_NAME \
    --project $PROJECT_ID \
    --region $REGION \
    --quiet
```

## Tests

```bash
SERVICE_URL=$(gcloud run services describe ${TRANS_SERVICE_NAME} --region ${REGION} --format 'value(status.url)')
```

```bash
curl -X POST "${SERVICE_URL}/generate" \
-H "Content-Type: application/json" \
-d '{"type": "translation", "prompt": "The ability to fine-tune powerful language models on consumer hardware is a significant breakthrough for the AI community.", "max_tokens": 1500}'

["### Instruction:\nTranslate the following text from English to Korean.\n\n### Input:\nThe ability to fine-tune powerful language models on consumer hardware is a significant breakthrough for the AI community.\n \n","\n이러한 소비자용 하드웨어에서 강력한 언어 모델을 조정할 수 있게 된 것은 AI 커뮤니티에 획기적인 발전입니다.\n","\n이러한 소비자용 하드웨어에서 강력한 언어 모델을 조정할 수 있게 된 것은 AI 커뮤니티에 획기적인 발전입니다.\n","\n이러한 소비자용 하드웨어에서 강력한 언어 모델을 조정할 수 있게 된 것은 AI 커뮤니티에 획기적인 발전입니다.\n","\n이러한 소비자용 하드웨어에서 강력한 언어 모델을 조정할 수 있게 된 것은 AI 커뮤니티에 획기적인 발전입니다.\n","\n이러한 소비자용 하드웨어에서 강력한 언어 모델을 조정할 수 있게 된 것은 AI 커뮤니티에 획기적인 발전입니다.\n","\n이러한 소비자용 하드웨어에서 강력한 언어 모델을 조정할 수 있게 된 것은 AI 커뮤니티에 획기적인 발전입니다.\n","\n이러한 소비자용 하드웨어에서 강력한 언어 모델을 조정할 수 있게 된 것은 AI 커뮤니티에 획기적인 발전입니다.\n","\n이러한 소비자용 하드웨어에서 강력한 언어 모델을 조정할 수 있게 된 것은 AI 커뮤니티에 획기적인 발전입니다.\n","\n이러한 소비자용 하드웨어에서 강력한 언어 모델을 조정할 수 있게 된 것은 AI 커뮤니티에 획기적인 발전입니다.\n","\n이러한 소비자용 하드웨어에서 강력한 언어 모델을"]
```

```bash
curl -X POST "${SERVICE_URL}/generate" \
-H "Content-Type: application/json" \
-d '{"type": "text", "prompt": "Google Cloud Run의 장점은 무엇인가요?", "max_tokens": 1024}'

"user\nGoogle Cloud Run의 장점은 무엇인가요?\nmodel\nGoogle Cloud Run으로 웹서비스를 배포하고 운영하는 것의 장점은 다음과 같습니다: 1. 확장성과 무한대스케일링: Google Cloud Run은 자동 스케일링이 가능하여 트래픽 요구사항에 따라 서버 인원을 증가 또는 감소시킬 수 있어 확장성과 무한대스케일링이 가능합니다. 2. 로드밸런싱 및 SSL 인증: Google Cloud Run은 자동으로 로드밸런싱과 SSL 인증을 관리하여 안정적인 서비스 제공을 보장합니다. 3. 저준비금액정책: 비용 효율적입니다. 배포 시 불필요한 자원이 사용될 경우 이를 자동으로 감축하여 비용을 관리합니다. 4. 자동확장성: 트래픽 요구사항 변화에 따라 자동으로 서비스 인원을 확장하거나 감소시켜 긴장을 관리할 수 있습니다. 5. 지속가능한 개발방법: 개발자가 코드 작성과 테스트 및 배포를 간소화할 수 있는 지속가능한 개발방법을 제공합니다. 이러한 장점들은 Google Cloud Run이 현대 웹서비스 발리다시 위해 탁월한 선택임을 보여줍니다."
```

```bash
curl -X POST "${SERVICE_URL}/generate" \
-H "Content-Type: application/json" \
-d '{"type": "product", "prompt": "test", "max_tokens": 1000}'

"Avengers Assemble Titanheld Iron Man Actionfigur: 30.5 cm große Iron Man Figur, ideal für sammelnden Fans. Hasbro Marvel Avengers Serie.  Diese Titanheld Iron Man Figur ist ein Muss für jeden Marvel Avengers Collector! Mit ihrer hohen Qualität und detaillierten Gestaltung ist sie ein perfektes Geschenk für Kinder und erwachsene Fans.  Für weiteren Marvel Avengers Enthusiasmus kaufen Sie diese großartige Titanheld Iron Man Figur heute!"
```

## 

```bash
gs://ayouzone-ai-gemma3/gce-us-central1/gemma-3-4b-product_merged_model/
├── config.json
├── model-00001-of-00010.safetensors
├── model-00002-of-00010.safetensors
├── model-00003-of-00010.safetensors
├── model-00004-of-00010.safetensors
├── model-00005-of-00010.safetensors
├── model-00006-of-00010.safetensors
├── model-00007-of-00010.safetensors
├── model-00008-of-00010.safetensors
├── model-00009-of-00010.safetensors
├── model-00010-of-00010.safetensors
├── model.safetensors.index.json
├── special_tokens_map.json
├── tokenizer_config.json
├── tokenizer.json
└── ...
```

## Errors

#### IAM Role Error

```bash
ERROR: (gcloud.builds.submit) PERMISSION_DENIED: The caller does not have permission. This command is authenticated as sjkim@sayouzone.com which is the active account specified by the [core/account] property
```

```bash
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="user:sjkim@sayouzone.com" \
    --role="roles/cloudbuild.builds.editor"
```

```bash
ERROR: (gcloud.builds.submit) INVALID_ARGUMENT: could not resolve source: googleapi: Error 403: 1037372895180-compute@developer.gserviceaccount.com does not have storage.objects.get access to the Google Cloud Storage object. Permission 'storage.objects.get' denied on resource (or it may not exist)., forbidden
```

```bash
gcloud projects add-iam-policy-binding $(gcloud config get-value project) \
    --member="serviceAccount:1037372895180-compute@developer.gserviceaccount.com" \
    --role="roles/storage.objectViewer"
```

```bash
INFO: The service account running this build projects/sayouzone-ai/serviceAccounts/1037372895180-compute@developer.gserviceaccount.com does not have permission to write logs to Cloud Logging. To fix this, grant the Logs Writer (roles/logging.logWriter) role to the service account.

1 message(s) issued.
ERROR: (gcloud.builds.submit) build 5e4fb29b-66d9-46ea-a687-902b1a5270df completed with status "FAILURE"
```

```bash
gcloud projects add-iam-policy-binding $(gcloud config get-value project) \
    --member="serviceAccount:1037372895180-compute@developer.gserviceaccount.com" \
    --role="roles/logging.logWriter"
```

```bash
denied: Permission "artifactregistry.repositories.uploadArtifacts" denied on resource "projects/sayouzone-ai/locations/us-central1/repositories/gemma3-repo" (or it may not exist)
```

```bash
gcloud projects add-iam-policy-binding $(gcloud config get-value project) \
    --member="serviceAccount:1037372895180-compute@developer.gserviceaccount.com" \
    --role="roles/artifactregistry.writer"
```

#### Startup Probe Errors

```bash
ERROR: (gcloud.run.deploy) Revision 'gemma3-product-ft-service-00001-qls' is not ready and cannot serve traffic. Container failed to become healthy. Startup probes timed out after 4m (1 attempts with a timeout of 4m each). There was an initial delay of 0s. If this happens frequently, consider adjusting the probe settings.
```