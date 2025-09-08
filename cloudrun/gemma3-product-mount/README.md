# Gemma 3 fine-tuned model on Cloud Run using Google Cloud Storage

Hugging Face 모델을 Google Cloud Storage(GCS)에서 직접 로드하려면 model_id를 GCS 경로로 지정해야 합니다. 이렇게 하면 모델 파일을 로컬에 다운로드하지 않고 GCS에서 직접 스트리밍할 수 있습니다.

Google Cloud Storage(GCS)에서 Volume Mount를 통해서 모델 로딩하는 방법

- [Cloud Run에서 Gemma 3 실행](https://cloud.google.com/run/docs/run-gemma-on-cloud-run?hl=ko)
- [권장사항: GPU를 사용하는 Cloud Run의 AI 추론](https://cloud.google.com/run/docs/configuring/services/gpu-best-practices?hl=ko)
- [Google Cloud Storage Volume Mount](https://cloud.google.com/run/docs/configuring/services/cloud-storage-volume-mounts?hl=ko#mount-volume)

## 프로젝트 구조 설정

## 컨테이너 빌드 및 Artifact Registry에 푸시

**1. 변수 설정:**

```bash
export PROJECT_ID=$(gcloud config get-value project)
export REGION="us-central1" # GPU를 지원하는 리전 선택
export REPO_NAME="gemma3-repo"
export PRODUCT_IMAGE_NAME="gemma3-product-ft-service"
export PRODUCT_IMAGE_TAG="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${PRODUCT_IMAGE_NAME}:latest"
export HF_TOKEN=sayouzone-huggingface-token:latest

export BUCKET_NAME="sayouzone-ai-gemma3"
export VOLUME_NAME="gemma3-product"
export MOUNT_PATH="/mnt/model"
```

**2. Artifact Registry 저장소 생성 (최초 한 번만 실행)::**

```bash
gcloud artifacts repositories create $REPO_NAME \
    --repository-format=docker \
    --location=$REGION
```

**3. Cloud Build를 사용하여 이미지 빌드 및 푸시:**

```bash
gcloud builds submit --tag $PRODUCT_IMAGE_TAG
```

## Cloud Run에 배포

빌드된 컨테이너 이미지를 사용하여 GPU가 장착된 Cloud Run 서비스에 배포합니다.

- --gpu=1: 서비스에 1개의 GPU를 할당.
- --gpu-type=nvidia-l4: 사용할 GPU 타입을 지정 (L4는 추론에 효율적이다).
- --cpu=4, --memory=16Gi: 충분한 CPU와 메모리를 할당. 모델 크기에 따라 조절이 필요.
- --concurrency=1: GPU를 사용하는 경우, 일반적으로 동시성(concurrency)을 1로 설정하여 한 번에 하나의 요청만 처리하도록 하는 것이 안정적이다.
- --no-cpu-throttling: 백그라운드 작업이 CPU에 의해 제한되지 않도록 한다.

#### HF_TOKEN 사용

```bash
gcloud run deploy $PRODUCT_IMAGE_NAME \
    --image=$PRODUCT_IMAGE_TAG \
    --region=$REGION \
    --gpu=1 \
    --gpu-type=nvidia-l4 \
    --cpu=8 \
    --memory=32Gi \
    --concurrency=1 \
    --no-cpu-throttling \
    --allow-unauthenticated \
    --update-secrets=HF_TOKEN=$HF_TOKEN \
    --timeout=30m
```

#### Google Cloud Storage Volume Mount

```bash
gcloud run deploy $PRODUCT_IMAGE_NAME \
    --image=$PRODUCT_IMAGE_TAG \
    --region=$REGION \
    --gpu=1 \
    --gpu-type=nvidia-l4 \
    --cpu=8 \
    --memory=32Gi \
    --concurrency=1 \
    --no-cpu-throttling \
    --allow-unauthenticated \
    --add-volume name=$VOLUME_NAME,type=cloud-storage,bucket=$BUCKET_NAME \
    --add-volume-mount volume=$VOLUME_NAME,mount-path=$MOUNT_PATH \
    --timeout=30m
```

#### Google Cloud Storage Volume Mount (readonly)

```bash
gcloud run deploy $PRODUCT_IMAGE_NAME \
    --image=$PRODUCT_IMAGE_TAG \
    --region=$REGION \
    --gpu=1 \
    --gpu-type=nvidia-l4 \
    --cpu=8 \
    --memory=32Gi \
    --concurrency=1 \
    --no-cpu-throttling \
    --allow-unauthenticated \
    --add-volume name=$VOLUME_NAME,type=cloud-storage,bucket=$BUCKET_NAME,readonly=true \
    --add-volume-mount volume=$VOLUME_NAME,mount-path=$MOUNT_PATH \
    --timeout=30m
```

#### Startup Probe

```bash
gcloud run deploy $PRODUCT_IMAGE_NAME \
    --image=$PRODUCT_IMAGE_TAG \
    --region=$REGION \
    --gpu=1 \
    --gpu-type=nvidia-l4 \
    --cpu=8 \
    --memory=32Gi \
    --concurrency=1 \
    --no-cpu-throttling \
    --allow-unauthenticated \
    --add-volume name=$VOLUME_NAME,type=cloud-storage,bucket=$BUCKET_NAME,readonly=true \
    --add-volume-mount volume=$VOLUME_NAME,mount-path=$MOUNT_PATH \
    --startup-probe tcpSocket.port=8080,initialDelaySeconds=240,failureThreshold=5,timeoutSeconds=240,periodSeconds=240 \
    --timeout=30m
```

#### Delete Cloud Run Service

```bash
gcloud run services delete $PRODUCT_IMAGE_NAME \
    --project $PROJECT_ID \
    --region $REGION \
    --quiet
```

## Tests

```bash
SERVICE_URL=$(gcloud run services describe $PRODUCT_IMAGE_NAME --region $REGION --format 'value(status.url)')
```

```bash
curl -X POST "${SERVICE_URL}/generate" \
-H "Content-Type: application/json" \
-d '{"type": "text", "prompt": "Google Cloud Run의 장점은 무엇인가요?", "max_tokens": 150}'

"user\nGoogle Cloud Run의 장점은 무엇인가요?\nmodel\nGoogle Cloud Run은 서버리스 앱 실행을 쉽게 하는 서버리스 플랫폼입니다. 기존 이식 코드와 최소 간섭으로 실행될 수 있으며, 구글 클라우드 네트워크에서 자동으로 확장 및 운영됩니다. 이를 통해 빠른 개발과 생산성 개선을 기대할 수 있습니다."
```

```bash
curl -X POST "${SERVICE_URL}/generate" \
-H "Content-Type: application/json" \
-d '{"type": "product", "prompt": "test", "max_tokens": 1000}'

"Avengers Assemble Titan Hero Iron Man Action Figure: Raise your Avengers game to Titan proportion with this colossal 30.5 cm Iron Man figure! Part of the acclaimed Marvel Assemble Titan Hero Series, this detailed action figure is perfect for fans of all ages. Collect all the Titan Hero figures to recreate your favourite Avengers scenes!"
```

## Fine-tuned Model 폴더

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

#### IAM Role Errors

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

#### Google Cloud Storage Errors

```bash
export BUCKET_NAME="sayouzone-ai-gemma3"
export VOLUME_NAME="gce-us-central1/gemma-3-4b-product_merged_model"
export MOUNT_PATH="model"

ERROR: (gcloud.run.deploy) service.spec.template.spec.containers[0].volume_mounts[0].mount_path: should be a valid unix absolute path
```

```bash
export BUCKET_NAME="sayouzone-ai-gemma3"
export VOLUME_NAME="gce-us-central1/gemma-3-4b-product_merged_model"
export MOUNT_PATH="/mnt/model"

ERROR: (gcloud.run.deploy) service.spec.template.spec.containers[0].volume_mounts[0].name: should only have alphanumeric characters, hyphens and underscores
```

```bash
export BUCKET_NAME="sayouzone-ai-gemma3"
export VOLUME_NAME="mnt"
export MOUNT_PATH="model"

ERROR: (gcloud.run.deploy) service.spec.template.spec.containers[0].volume_mounts[0].mount_path: should be a valid unix absolute path
```

```bash
export BUCKET_NAME="sayouzone-ai-gemma3"
export VOLUME_NAME="my-gcs-volume"
export MOUNT_PATH="/mnt/model"

ValueError: Unrecognized model in /mnt/model. Should have a `model_type` key in its config.json, or contain one of the following strings in its name: aimv2, aimv2_vision_model, albert, align, altclip, apertus, arcee, aria, aria_text, audio-spectrogram-transformer, autoformer, aya_vision, bamba, bark, bart, beit, bert, bert-generation, big_bird, bigbird_pegasus, biogpt, bit, bitnet, blenderbot, blenderbot-small, blip, blip-2, blip_2_qformer, bloom, bridgetower, bros, camembert, canine, chameleon, chinese_clip, chinese_clip_vision_model, clap, clip, clip_text_model, clip_vision_model, clipseg, clvp, code_llama, codegen, cohere, cohere2, cohere2_vision, colpali, colqwen2, conditional_detr, convbert, convnext, convnextv2, cpmant, csm, ctrl, cvt, d_fine, dab-detr, dac, data2vec-audio, data2vec-text, data2vec-vision, dbrx, deberta, deberta-v2, decision_transformer, deepseek_v2, deepseek_v3, deepseek_vl, deepseek_vl_hybrid, deformable_detr, deit, depth_anything, depth_pro, deta, detr, dia, diffllama, dinat, dinov2, dinov2_with_registers, dinov3_convnext, dinov3_vit, distilbert, doge, donut-swin, dots1, dpr, dpt, efficientformer, efficientloftr, efficientnet, electra, emu3, encodec, encoder-decoder, eomt, ernie, ernie4_5, ernie4_5_moe, ernie_m, esm, evolla, exaone4, falcon, falcon_h1, falcon_mamba, fastspeech2_conformer, fastspeech2_conformer_with_hifigan, flaubert, flava, florence2, fnet, focalnet, fsmt, funnel, fuyu, gemma, gemma2, gemma3, gemma3_text, gemma3n, gemma3n_audio, gemma3n_text, gemma3n_vision, git, glm, glm4, glm4_moe, glm4v, glm4v_moe, glm4v_moe_text, glm4v_text, glpn, got_ocr2, gpt-sw3, gpt2, gpt_bigcode, gpt_neo, gpt_neox, gpt_neox_japanese, gpt_oss, gptj, gptsan-japanese, granite, granite_speech, granitemoe, granitemoehybrid, granitemoeshared, granitevision, graphormer, grounding-dino, groupvit, helium, hgnet_v2, hiera, hubert, hunyuan_v1_dense, hunyuan_v1_moe, ibert, idefics, idefics2, idefics3, idefics3_vision, ijepa, imagegpt, informer, instructblip, instructblipvideo, internvl, internvl_vision, jamba, janus, jetmoe, jukebox, kosmos-2, kosmos-2.5, kyutai_speech_to_text, layoutlm, layoutlmv2, layoutlmv3, led, levit, lfm2, lightglue, lilt, llama, llama4, llama4_text, llava, llava_next, llava_next_video, llava_onevision, longformer, longt5, luke, lxmert, m2m_100, mamba, mamba2, marian, markuplm, mask2former, maskformer, maskformer-swin, mbart, mctct, mega, megatron-bert, metaclip_2, mgp-str, mimi, minimax, mistral, mistral3, mixtral, mlcd, mllama, mm-grounding-dino, mobilebert, mobilenet_v1, mobilenet_v2, mobilevit, mobilevitv2, modernbert, modernbert-decoder, moonshine, moshi, mpnet, mpt, mra, mt5, musicgen, musicgen_melody, mvp, nat, nemotron, nezha, nllb-moe, nougat, nystromformer, olmo, olmo2, olmoe, omdet-turbo, oneformer, open-llama, openai-gpt, opt, ovis2, owlv2, owlvit, paligemma, patchtsmixer, patchtst, pegasus, pegasus_x, perceiver, perception_encoder, perception_lm, persimmon, phi, phi3, phi4_multimodal, phimoe, pix2struct, pixtral, plbart, poolformer, pop2piano, prompt_depth_anything, prophetnet, pvt, pvt_v2, qdqbert, qwen2, qwen2_5_omni, qwen2_5_vl, qwen2_5_vl_text, qwen2_audio, qwen2_audio_encoder, qwen2_moe, qwen2_vl, qwen2_vl_text, qwen3, qwen3_moe, rag, realm, recurrent_gemma, reformer, regnet, rembert, resnet, retribert, roberta, roberta-prelayernorm, roc_bert, roformer, rt_detr, rt_detr_resnet, rt_detr_v2, rwkv, sam, sam2, sam2_hiera_det_model, sam2_video, sam2_vision_model, sam_hq, sam_hq_vision_model, sam_vision_model, seamless_m4t, seamless_m4t_v2, seed_oss, segformer, seggpt, sew, sew-d, shieldgemma2, siglip, siglip2, siglip_vision_model, smollm3, smolvlm, smolvlm_vision, speech-encoder-decoder, speech_to_text, speech_to_text_2, speecht5, splinter, squeezebert, stablelm, starcoder2, superglue, superpoint, swiftformer, swin, swin2sr, swinv2, switch_transformers, t5, t5gemma, table-transformer, tapas, textnet, time_series_transformer, timesfm, timesformer, timm_backbone, timm_wrapper, trajectory_transformer, transfo-xl, trocr, tvlt, tvp, udop, umt5, unispeech, unispeech-sat, univnet, upernet, van, video_llava, videomae, vilt, vipllava, vision-encoder-decoder, vision-text-dual-encoder, visual_bert, vit, vit_hybrid, vit_mae, vit_msn, vitdet, vitmatte, vitpose, vitpose_backbone, vits, vivit, vjepa2, voxtral, voxtral_encoder, wav2vec2, wav2vec2-bert, wav2vec2-conformer, wavlm, whisper, xclip, xcodec, xglm, xlm, xlm-prophetnet, xlm-roberta, xlm-roberta-xl, xlnet, xlstm, xmod, yolos, yoso, zamba, zamba2, zoedepth
```

#### Startup probe Errors

```bash
ERROR: (gcloud.run.deploy) Revision 'gemma3-product-ft-service-00001-8sb' is not ready and cannot serve traffic. Container failed to become healthy. Startup probes timed out after 4m (1 attempts with a timeout of 4m each). There was an initial delay of 0s. If this happens frequently, consider adjusting the probe settings.
```

```bash
ERROR: (gcloud.run.deploy) Revision 'gemma3-product-ft-service-00002-wx5' is not ready and cannot serve traffic. Container failed to become healthy. Startup probes timed out after 24m (5 attempts with a timeout of 4m each). There was an initial delay of 4m. If this happens frequently, consider adjusting the probe settings.
```