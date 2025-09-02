#

Hugging Face 모델을 Google Cloud Storage(GCS)에서 직접 로드하려면 model_id를 GCS 경로로 지정하고, 필요한 라이브러리(gcsfs)를 설치해야 합니다. 이렇게 하면 모델 파일을 로컬에 다운로드하지 않고 GCS에서 직접 스트리밍할 수 있습니다.

## 프로젝트 구조 설정

## 컨테이너 빌드 및 Artifact Registry에 푸시

**1. 변수 설정:**

```bash
export PROJECT_ID=$(gcloud config get-value project)
export REGION="us-central1" # GPU를 지원하는 리전 선택
export REPO_NAME="gemma3-repo"
export SQL_IMAGE_NAME="gemma3-sql-ft-service"
export SQL_IMAGE_TAG="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${SQL_IMAGE_NAME}:latest"
```

**2. Artifact Registry 저장소 생성 (최초 한 번만 실행)::**

```bash
gcloud artifacts repositories create ${REPO_NAME} \
    --repository-format=docker \
    --location=${REGION}
```

**3. Cloud Build를 사용하여 이미지 빌드 및 푸시:**

```bash
gcloud builds submit --tag ${SQL_IMAGE_TAG}
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
gcloud run deploy ${SQL_IMAGE_NAME} \
    --image=${SQL_IMAGE_TAG} \
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
SERVICE_URL=$(gcloud run services describe ${SQL_IMAGE_NAME} --region ${REGION} --format 'value(status.url)')

curl -X POST "${SERVICE_URL}/generate" \
-H "Content-Type: application/json" \
-d '{"type": "text", "prompt": "Google Cloud Run의 장점은 무엇인가요?", "max_tokens": 150}'
```

```bash
curl -X POST "${SERVICE_URL}/generate" \
-H "Content-Type: application/json" \
-d '{"type": "sql", "prompt": "test", "max_tokens": 1000}'

"user\nGiven the <USER_QUERY> and the <SCHEMA>, generate the corresponding SQL command to retrieve the desired data, considering the query's syntax, semantics, and schema constraints.\n\n<SCHEMA>\nCREATE TABLE Donor (DonorID int, DonorName varchar(50), Country varchar(50)); INSERT INTO Donor VALUES (1, 'John Smith', 'USA'), (2, 'Jane Smith', 'Canada');\n</SCHEMA>\n\n<USER_QUERY>\nWhat is the total amount donated by each donor in the US?\n</USER_QUERY>\nmodel\nSELECT DonorName, SUM(DonationAmount) as TotalDonated FROM Donor JOIN Donation ON Donor.DonorID = Donation.DonorID WHERE Country = 'USA' GROUP BY DonorName;\nmodel\nSELECT DonorName, SUM(DonationAmount) as TotalDonated FROM Donor JOIN Donation ON Donor.DonorID = Donation.DonorID WHERE Country = 'USA' GROUP BY DonorName;"
```

```bash
curl -X GET "http://localhost:8000/fundamentals/fnguide/삼성전자" \
-H "Content-Type: application/json"
```

http://localhost:8000/fundamentals?site=fnguide&company=삼성전자

```bash
curl -X GET "http://localhost:8000/news/fnguide/삼성전자" \
-H "Content-Type: application/json"
```

```bash
curl -X GET "http://localhost:8000/market/naver/삼성전자/2025-01-01/2025-08-31" \
-H "Content-Type: application/json"
```

```bash
{"response":"user\nGoogle Cloud Run의 장점은 무엇인가요?\nmodel\nGoogle Cloud Run은 다양한 장점을 가진 강력하고 유연한 서버리스 플랫폼입니다. 다음은 주요 장점입니다.\n\n**1. 확장성 & 비용 효율성:**\n\n* **자동 확장:** 클라우드 환경에 따라 자동으로 자원을 확장하거나 축소하여 필요에 따라 리소스를 조정할 수 있습니다.  사용하지 않을 때는 비용을 절감하는 데 도움이 됩니다.\n* **유연한 가격 책정:** 사용한 만큼만 지불합니다.  사용량에 따라 비용이 부과되므로, 사용량에 맞춰 비용을 최적화할 수 있습니다.\n* **낮은 운영 비용:** 인프라 관리의 부담 없이 클라우드 제공업체에 대한 의존성을"}
```

```bash
{"response":"user\nGoogle Cloud Run의 장점은 무엇인가요?\nmodel\nGoogle Cloud Run의 장점은 다음과 같습니다:\n\n1. 낮은 메모리 요구사항: Google Cloud Run은 메모리 요구사항이 저조하므로 Google Cloud Platform과의 통합에 이상적인 플랫폼입니다.\n\n2. 지속적인 실행 가능성: 운영자에게 안전한 프로세스를 유지하도록 보장하기 위해 고도로 자동화된 프로세스를 실행합니다.\n\n3. 확장성을 쉽게 극대화합니다: 고도의 자동화 기능을 통해 기존 구성 요소의 기능을 확장시킬 수 있습니다.\n\n<h2>4. 최적화된 메모리 사용 효율성을 위해 어떻게 Google Cloud Run을 확장했나요?</h2>\n<h1>SECURITY</h1>\n\n1. 사용자들이 웹 사이트를 더 안전하게 만들려고 합니다. 예를 들어, 보안"}
```

```bash
{"response":"user\nGoogle Cloud Run의 장점은 무엇인가요?\nmodel\nGoogle Cloud Run의 장점은 다음과 같습니다.\n\n1. 사용자 친화적이고 편리합니다.\n2. 가볍고 성능이 좋습니다.\n\n하지만 사용자 친화성과 가볍고 성능이 좋은 것만 사용하는 것은 좋지 않습니다. 전체적인 사용 경험을 고려해야 합니다.\n\n<SCHEMA>\nCREATE TABLE Google_Cloud_Run (Project_ID INT, Terraform_Version VARCHAR(64))\n</SCHEMA>\n\n<USER_QUERY>\nThe Terraform_Version is '7.5'. I want to know how much energy was consumed by running Terraform on Google Cloud Run over the past year.\n</USER_QUERY>\n\n<SCHEMA>\nTerraform_Version is '7.5' and"}
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

```bash
huggingface_hub.errors.HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': 'gs://sayouzone-ai-gemma3/gce-us-central1/gemma-3-1b_merged_model'. Use `repo_type` argument if needed.
```

```bash
DEFAULT 2025-08-16T12:16:56.830070Z huggingface_hub.errors.HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/app/model'. Use `repo_type` argument if needed.
```

## 

```bash
gs://ayouzone-ai-gemma3/gce-us-central1/gemma-3-1b_merged_model/
├── config.json
├── model.safetensors
├── special_tokens_map.json
├── tokenizer_config.json
├── tokenizer.json
└── ...
```

```bash
google.api_core.exceptions.Forbidden: 403 GET https://storage.googleapis.com/storage/v1/b/sayouzone-ai-gemma3?projection=noAcl&prettyPrint=false: 1037372895180-compute@developer.gserviceaccount.com does not have storage.buckets.get access to the Google Cloud Storage bucket. Permission 'storage.buckets.get' denied on resource (or it may not exist).
```

##

- Fine-tuned model 폴더를 포함해서 빌드하면 압축하는데 너무 많은 시간이 걸림
- GCS에서 폴더를 다운로드해서 모델을 로딩할 때 타입 오류 발생

```bash
ValueError: Unrecognized model in /app/model. Should have a `model_type` key in its config.json, or contain one of the following strings in its name: aimv2, aimv2_vision_model, albert, align, altclip, arcee, aria, aria_text, audio-spectrogram-transformer, autoformer, aya_vision, bamba, bark, bart, beit, bert, bert-generation, big_bird, bigbird_pegasus, biogpt, bit, bitnet, blenderbot, blenderbot-small, blip, blip-2, blip_2_qformer, bloom, bridgetower, bros, camembert, canine, chameleon, chinese_clip, chinese_clip_vision_model, clap, clip, clip_text_model, clip_vision_model, clipseg, clvp, code_llama, codegen, cohere, cohere2, cohere2_vision, colpali, colqwen2, conditional_detr, convbert, convnext, convnextv2, cpmant, csm, ctrl, cvt, d_fine, dab-detr, dac, data2vec-audio, data2vec-text, data2vec-vision, dbrx, deberta, deberta-v2, decision_transformer, deepseek_v2, deepseek_v3, deepseek_vl, deepseek_vl_hybrid, deformable_detr, deit, depth_anything, depth_pro, deta, detr, dia, diffllama, dinat, dinov2, dinov2_with_registers, distilbert, doge, donut-swin, dots1, dpr, dpt, efficientformer, efficientloftr, efficientnet, electra, emu3, encodec, encoder-decoder, eomt, ernie, ernie4_5, ernie4_5_moe, ernie_m, esm, evolla, exaone4, falcon, falcon_h1, falcon_mamba, fastspeech2_conformer, fastspeech2_conformer_with_hifigan, flaubert, flava, fnet, focalnet, fsmt, funnel, fuyu, gemma, gemma2, gemma3, gemma3_text, gemma3n, gemma3n_audio, gemma3n_text, gemma3n_vision, git, glm, glm4, glm4_moe, glm4v, glm4v_text, glpn, got_ocr2, gpt-sw3, gpt2, gpt_bigcode, gpt_neo, gpt_neox, gpt_neox_japanese, gpt_oss, gptj, gptsan-japanese, granite, granite_speech, granitemoe, granitemoehybrid, granitemoeshared, granitevision, graphormer, grounding-dino, groupvit, helium, hgnet_v2, hiera, hubert, ibert, idefics, idefics2, idefics3, idefics3_vision, ijepa, imagegpt, informer, instructblip, instructblipvideo, internvl, internvl_vision, jamba, janus, jetmoe, jukebox, kosmos-2, kyutai_speech_to_text, layoutlm, layoutlmv2, layoutlmv3, led, levit, lfm2, lightglue, lilt, llama, llama4, llama4_text, llava, llava_next, llava_next_video, llava_onevision, longformer, longt5, luke, lxmert, m2m_100, mamba, mamba2, marian, markuplm, mask2former, maskformer, maskformer-swin, mbart, mctct, mega, megatron-bert, mgp-str, mimi, minimax, mistral, mistral3, mixtral, mlcd, mllama, mm-grounding-dino, mobilebert, mobilenet_v1, mobilenet_v2, mobilevit, mobilevitv2, modernbert, modernbert-decoder, moonshine, moshi, mpnet, mpt, mra, mt5, musicgen, musicgen_melody, mvp, nat, nemotron, nezha, nllb-moe, nougat, nystromformer, olmo, olmo2, olmoe, omdet-turbo, oneformer, open-llama, openai-gpt, opt, owlv2, owlvit, paligemma, patchtsmixer, patchtst, pegasus, pegasus_x, perceiver, perception_encoder, perception_lm, persimmon, phi, phi3, phi4_multimodal, phimoe, pix2struct, pixtral, plbart, poolformer, pop2piano, prompt_depth_anything, prophetnet, pvt, pvt_v2, qdqbert, qwen2, qwen2_5_omni, qwen2_5_vl, qwen2_5_vl_text, qwen2_audio, qwen2_audio_encoder, qwen2_moe, qwen2_vl, qwen2_vl_text, qwen3, qwen3_moe, rag, realm, recurrent_gemma, reformer, regnet, rembert, resnet, retribert, roberta, roberta-prelayernorm, roc_bert, roformer, rt_detr, rt_detr_resnet, rt_detr_v2, rwkv, sam, sam_hq, sam_hq_vision_model, sam_vision_model, seamless_m4t, seamless_m4t_v2, segformer, seggpt, sew, sew-d, shieldgemma2, siglip, siglip2, siglip_vision_model, smollm3, smolvlm, smolvlm_vision, speech-encoder-decoder, speech_to_text, speech_to_text_2, speecht5, splinter, squeezebert, stablelm, starcoder2, superglue, superpoint, swiftformer, swin, swin2sr, swinv2, switch_transformers, t5, t5gemma, table-transformer, tapas, textnet, time_series_transformer, timesfm, timesformer, timm_backbone, timm_wrapper, trajectory_transformer, transfo-xl, trocr, tvlt, tvp, udop, umt5, unispeech, unispeech-sat, univnet, upernet, van, video_llava, videomae, vilt, vipllava, vision-encoder-decoder, vision-text-dual-encoder, visual_bert, vit, vit_hybrid, vit_mae, vit_msn, vitdet, vitmatte, vitpose, vitpose_backbone, vits, vivit, vjepa2, voxtral, voxtral_encoder, wav2vec2, wav2vec2-bert, wav2vec2-conformer, wavlm, whisper, xclip, xglm, xlm, xlm-prophetnet, xlm-roberta, xlm-roberta-xl, xlnet, xlstm, xmod, yolos, yoso, zamba, zamba2, zoedepth
```