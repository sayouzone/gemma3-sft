# gemma3-sft
Gemma 3 Fine-tuning in GCE + GPU, TPU, Cloud Run + GPU, GKE + TPU


## Hugging Face Token

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

## GCE + GPU

[GCE](https://github.com/sayouzone/gemma3-sft/tree/main/gce)

## TPU

[TPU](https://github.com/sayouzone/gemma3-sft/tree/main/tpu)

## Cloud Run + GPU

[Cloud Run](https://github.com/sayouzone/gemma3-sft/tree/main/cloudrun)

## MacBook Pro M4

[MacBook Pro M4](https://github.com/sayouzone/gemma3-sft/tree/main/m4)

## GKE + TPU

## Tests