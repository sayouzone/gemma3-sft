#

Hugging Face 모델을 Google Cloud Storage(GCS)에서 직접 로드하려면 model_id를 GCS 경로로 지정하고, 필요한 라이브러리(gcsfs)를 설치해야 합니다. 이렇게 하면 모델 파일을 로컬에 다운로드하지 않고 GCS에서 직접 스트리밍할 수 있습니다.

## 프로젝트 구조 설정

## 컨테이너 빌드 및 Artifact Registry에 푸시

**1. 변수 설정:**

```bash
export PROJECT_ID=$(gcloud config get-value project)
export REGION="us-central1" # GPU를 지원하는 리전 선택
export REPO_NAME="gemma3-repo"
export PRODUCT_IMAGE_NAME="gemma3-product-ft-service"
export PRODUCT_IMAGE_TAG="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${PRODUCT_IMAGE_NAME}:latest"
```

**2. Artifact Registry 저장소 생성 (최초 한 번만 실행)::**

```bash
gcloud artifacts repositories create ${REPO_NAME} \
    --repository-format=docker \
    --location=${REGION}
```

**3. Cloud Build를 사용하여 이미지 빌드 및 푸시:**

```bash
gcloud builds submit --tag ${PRODUCT_IMAGE_TAG}
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

## Cloud Run에 배포

빌드된 컨테이너 이미지를 사용하여 GPU가 장착된 Cloud Run 서비스에 배포합니다.

- --gpu=1: 서비스에 1개의 GPU를 할당.
- --gpu-type=nvidia-l4: 사용할 GPU 타입을 지정 (L4는 추론에 효율적이다).
- --cpu=4, --memory=16Gi: 충분한 CPU와 메모리를 할당. 모델 크기에 따라 조절이 필요.
- --concurrency=1: GPU를 사용하는 경우, 일반적으로 동시성(concurrency)을 1로 설정하여 한 번에 하나의 요청만 처리하도록 하는 것이 안정적이다.
- --no-cpu-throttling: 백그라운드 작업이 CPU에 의해 제한되지 않도록 한다.

```bash
gcloud run deploy ${PRODUCT_IMAGE_NAME} \
    --image=${PRODUCT_IMAGE_TAG} \
    --region=${REGION} \
    --gpu=1 \
    --gpu-type=nvidia-l4 \
    --cpu=8 \
    --memory=32Gi \
    --concurrency=1 \
    --no-cpu-throttling \
    --allow-unauthenticated \
    --timeout=30m
```

```bash
SERVICE_URL=$(gcloud run services describe ${PRODUCT_IMAGE_NAME} --region ${REGION} --format 'value(status.url)')

curl -X POST "${SERVICE_URL}/generate" \
-H "Content-Type: application/json" \
-d '{"type": "text", "prompt": "Google Cloud Run의 장점은 무엇인가요?", "max_tokens": 150}'
```

```bash
curl -X POST "${SERVICE_URL}/generate" \
-H "Content-Type: application/json" \
-d '{"type": "product", "prompt": "test", "max_tokens": 1000}'

"user\nGiven the <USER_QUERY> and the <SCHEMA>, generate the corresponding SQL command to retrieve the desired data, considering the query's syntax, semantics, and schema constraints.\n\n<SCHEMA>\nCREATE TABLE Donor (DonorID int, DonorName varchar(50), Country varchar(50)); INSERT INTO Donor VALUES (1, 'John Smith', 'USA'), (2, 'Jane Smith', 'Canada');\n</SCHEMA>\n\n<USER_QUERY>\nWhat is the total amount donated by each donor in the US?\n</USER_QUERY>\nmodel\nSELECT DonorName, SUM(DonationAmount) as TotalDonated FROM Donor JOIN Donation ON Donor.DonorID = Donation.DonorID WHERE Country = 'USA' GROUP BY DonorName;\nmodel\nSELECT DonorName, SUM(DonationAmount) as TotalDonated FROM Donor JOIN Donation ON Donor.DonorID = Donation.DonorID WHERE Country = 'USA' GROUP BY DonorName;"
```

## 

```bash
gs://ayouzone-ai-gemma3/gce-us-central1/gemma-3-4b-product_merged_model/
├── config.json
├── model.safetensors
├── special_tokens_map.json
├── tokenizer_config.json
├── tokenizer.json
└── ...
```
