## 프로젝트 구조 설정

## 컨테이너 빌드 및 Artifact Registry에 푸시

**1. 변수 설정:**

```bash
export PROJECT_ID=$(gcloud config get-value project)
export REGION="us-central1" # GPU를 지원하는 리전 선택
export REPO_NAME="gemma3-repo"
export BASE_IMAGE_NAME="gemma3-base-service"
export BASE_IMAGE_TAG="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${BASE_IMAGE_NAME}:latest"
```

**2. Artifact Registry 저장소 생성 (최초 한 번만 실행)::**

```bash
gcloud artifacts repositories create ${REPO_NAME} \
    --repository-format=docker \
    --location=${REGION}
```

**3. Cloud Build를 사용하여 이미지 빌드 및 푸시:**

```bash
gcloud builds submit --tag ${BASE_IMAGE_TAG}
```

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

```bash
gcloud projects add-iam-policy-binding $(gcloud config get-value project) \
    --member="serviceAccount:1037372895180-compute@developer.gserviceaccount.com" \
    --role="roles/storage.objectViewer" \
    --role="roles/logging.logWriter" \
    --role="roles/artifactregistry.writer"
```

## Cloud Run에 배포

빌드된 컨테이너 이미지를 사용하여 GPU가 장착된 Cloud Run 서비스에 배포합니다.

- --gpu=1: 서비스에 1개의 GPU를 할당.
- --gpu-type=nvidia-l4: 사용할 GPU 타입을 지정 (L4는 추론에 효율적이다).
- --cpu=4, --memory=16Gi: 충분한 CPU와 메모리를 할당. 모델 크기에 따라 조절이 필요.
- --concurrency=1: GPU를 사용하는 경우, 일반적으로 동시성(concurrency)을 1로 설정하여 한 번에 하나의 요청만 처리하도록 하는 것이 안정적이다.
- --no-cpu-throttling: 백그라운드 작업이 CPU에 의해 제한되지 않도록 한다.

```bash
gcloud run deploy ${BASE_IMAGE_NAME} \
    --image=${BASE_IMAGE_TAG} \
    --region=${REGION} \
    --gpu=1 \
    --gpu-type=nvidia-l4 \
    --cpu=4 \
    --memory=16Gi \
    --concurrency=1 \
    --no-cpu-throttling \
    --allow-unauthenticated \
    --timeout=15m
```

```bash
SERVICE_URL=$(gcloud run services describe ${BASE_IMAGE_NAME} --region ${REGION} --format 'value(status.url)')

curl -X POST "${SERVICE_URL}/generate" \
-H "Content-Type: application/json" \
-d '{"prompt": "Google Cloud Run의 장점은 무엇인가요?", "max_tokens": 150}'
```

```bash
{"response":"user\nGoogle Cloud Run의 장점은 무엇인가요?\nmodel\nGoogle Cloud Run은 다양한 장점을 가진 강력하고 유연한 서버리스 플랫폼입니다. 다음은 주요 장점입니다.\n\n**1. 확장성 & 비용 효율성:**\n\n* **자동 확장:** 클라우드 환경에 따라 자동으로 자원을 확장하거나 축소하여 필요에 따라 리소스를 조정할 수 있습니다.  사용하지 않을 때는 비용을 절감하는 데 도움이 됩니다.\n* **유연한 가격 책정:** 사용한 만큼만 지불합니다.  사용량에 따라 비용이 부과되므로, 사용량에 맞춰 비용을 최적화할 수 있습니다.\n* **낮은 운영 비용:** 인프라 관리의 부담 없이 클라우드 제공업체에 대한 의존성을"}
```

```bash
ERROR: (gcloud.run.deploy) Revision 'gemma3-ft-sql-service-00001-5nb' is not ready and cannot serve traffic. The user-provided container failed to start and listen on the port defined provided by the PORT=8080 environment variable within the allocated timeout. This can happen when the container port is misconfigured or if the timeout is too short. The health check timeout can be extended. Logs for this revision might contain more information.

Logs URL: https://console.cloud.google.com/logs/viewer?project=sayouzone-ai&resource=cloud_run_revision/service_name/gemma3-ft-sql-service/revision_name/gemma3-ft-sql-service-00001-5nb&advancedFilter=resource.type%3D%22cloud_run_revision%22%0Aresource.labels.service_name%3D%22gemma3-ft-sql-service%22%0Aresource.labels.revision_name%3D%22gemma3-ft-sql-service-00001-5nb%22 
For more troubleshooting guidance, see https://cloud.google.com/run/docs/troubleshooting#container-failed-to-start
```

```bash
torch._inductor.exc.InductorError: RuntimeError: Failed to find C compiler. Please specify via CC environment variable or set triton.knobs.build.impl.
Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"
Starting new instance. Reason: AUTOSCALING - Instance started due to configured scaling factors (e.g. CPU utilization, request throughput, etc.) or no existing capacity for current traffic.
```

```Dockerfile
RUN apt-get update && apt-get install -y build-essential git
```
