# Cloud Run에서 Gemma 3 모델 실행


##

```bash
export SERVICE_NAME=gemma3-test
export GEMMA_PARAMETER=gemma3-1b
export PROJECT_ID=sayouzone-ai
export REGION=us-central1

gcloud run deploy $SERVICE_NAME \
   --image us-docker.pkg.dev/cloudrun/container/gemma/$GEMMA_PARAMETER \
   --concurrency 4 \
   --cpu 8 \
   --set-env-vars OLLAMA_NUM_PARALLEL=4 \
   --gpu 1 \
   --gpu-type nvidia-l4 \
   --max-instances 1 \
   --memory 32Gi \
   --no-allow-unauthenticated \
   --no-cpu-throttling \
   --timeout=600 \
   --project $PROJECT_ID \
   --region $REGION
```

```bash
gcloud run services proxy ollama-gemma --port=9090
```

```bash
curl http://localhost:9090/api/generate -d '{
  "model": "gemma3:4b",
  "prompt": "Why is the sky blue?"
}'
```

```bash
export cloud_run_url=https://gemma3-test-1037372895180.us-central1.run.app

curl "$cloud_run_url/v1beta/models/*:generateContent" \
-H 'Content-Type: application/json' \
-H "Authorization: Bearer $(gcloud auth print-identity-token)" \
-X POST \
-d '{
  "contents": [{
    "parts":[{"text": "Write a story about a magic backpack. You are the narrator of an interactive text adventure game."}]
    }]
    }'
```