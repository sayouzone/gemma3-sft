

[TPU Pods Gemma3 파인튜닝 정리](https://blog.worldsw.dev/tpu-pods-gemma3-finetune/)

##

####

```bash
export PROJECT_ID=sayouzone-ai
export TPU_NAME=gemma3-tpu-v5-test
export ZONE=us-central1-a
export ACCELERATOR_TYPE=v5litepod-4
export RUNTIME_VERSION=v2-alpha-tpuv5-lite
```

```bash
export PROJECT_ID=sayouzone-ai
export TPU_NAME=gemma3-tpu-v5-test
export ZONE=us-central1-a
export ACCELERATOR_TYPE=v5litepod-4
export RUNTIME_VERSION=v2-alpha-tpuv5
```

```bash
export PROJECT_ID=sayouzone-ai
export TPU_NAME=gemma3-tpu-v5-test
export ZONE=us-central1-a
export ACCELERATOR_TYPE=v5litepod-8
export RUNTIME_VERSION=v2-alpha-tpuv5-lite
```

"Quota limit 'TPUV5sLitepodServingPerProjectPerZoneForTPUAPI' has been exceeded. Limit: 4 in zone us-central1-a."

**TPU VM 인스턴스 생성**

```bash
gcloud compute tpus tpu-vm create $TPU_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --accelerator-type=$ACCELERATOR_TYPE \
    --version=$RUNTIME_VERSION
```

##


```bash
sudo apt update
sudo apt install -y python3.10-venv screen

python3 -m venv gemma3-venv

screen -S gemma3

mkdir gemma3
cd gemma3

source gemma3-venv/bin/activate
```

```bash
screen -S gemma3
screen -r gemma3
```

```bash
git clone https://github.com/huggingface/optimum-tpu.git

cd optimum-tpu
pip install -e . -f https://storage.googleapis.com/libtpu-releases/index.html -q

cd ..

pip install trl peft -q
pip install datasets evaluate accelerate -q
pip install python-dotenv -q

pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```


```bash
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 10.14it/s]
Traceback (most recent call last):
  File "/home/sjkim/gemma3/test.py", line 36, in <module>
    fsdp_training_args = fsdp_v2.get_fsdp_training_args(model)
  File "/home/sjkim/gemma3/optimum-tpu/optimum/tpu/fsdp_v2.py", line 105, in get_fsdp_training_args
    raise ValueError(f"Model {model} configuration cannot be auto-generated, use get_fsdp_config instead.")
ValueError: Model Gemma3ForConditionalGeneration(
  (model): Gemma3Model(
    (vision_tower): SiglipVisionModel(
      (vision_model): SiglipVisionTransformer(
        (embeddings): SiglipVisionEmbeddings(
          (patch_embedding): Conv2d(3, 1152, kernel_size=(14, 14), stride=(14, 14), padding=valid)
          (position_embedding): Embedding(4096, 1152)
        )
        (encoder): SiglipEncoder(
          (layers): ModuleList(
            (0-26): 27 x SiglipEncoderLayer(
              (layer_norm1): LayerNorm((1152,), eps=1e-06, elementwise_affine=True)
              (self_attn): SiglipAttention(
                (k_proj): Linear(in_features=1152, out_features=1152, bias=True)
                (v_proj): Linear(in_features=1152, out_features=1152, bias=True)
                (q_proj): Linear(in_features=1152, out_features=1152, bias=True)
                (out_proj): Linear(in_features=1152, out_features=1152, bias=True)
              )
              (layer_norm2): LayerNorm((1152,), eps=1e-06, elementwise_affine=True)
              (mlp): SiglipMLP(
                (activation_fn): PytorchGELUTanh()
                (fc1): Linear(in_features=1152, out_features=4304, bias=True)
                (fc2): Linear(in_features=4304, out_features=1152, bias=True)
              )
            )
          )
        )
        (post_layernorm): LayerNorm((1152,), eps=1e-06, elementwise_affine=True)
      )
    )
    (multi_modal_projector): Gemma3MultiModalProjector(
      (mm_soft_emb_norm): Gemma3RMSNorm((1152,), eps=1e-06)
      (avg_pool): AvgPool2d(kernel_size=4, stride=4, padding=0)
    )
    (language_model): Gemma3TextModel(
      (embed_tokens): Gemma3TextScaledWordEmbedding(262208, 2560, padding_idx=0)
      (layers): ModuleList(
        (0-33): 34 x Gemma3DecoderLayer(
          (self_attn): Gemma3Attention(
            (q_proj): Linear(in_features=2560, out_features=2048, bias=False)
            (k_proj): Linear(in_features=2560, out_features=1024, bias=False)
            (v_proj): Linear(in_features=2560, out_features=1024, bias=False)
            (o_proj): Linear(in_features=2048, out_features=2560, bias=False)
            (q_norm): Gemma3RMSNorm((256,), eps=1e-06)
            (k_norm): Gemma3RMSNorm((256,), eps=1e-06)
          )
          (mlp): Gemma3MLP(
            (gate_proj): Linear(in_features=2560, out_features=10240, bias=False)
            (up_proj): Linear(in_features=2560, out_features=10240, bias=False)
            (down_proj): Linear(in_features=10240, out_features=2560, bias=False)
            (act_fn): PytorchGELUTanh()
          )
          (input_layernorm): Gemma3RMSNorm((2560,), eps=1e-06)
          (post_attention_layernorm): Gemma3RMSNorm((2560,), eps=1e-06)
          (pre_feedforward_layernorm): Gemma3RMSNorm((2560,), eps=1e-06)
          (post_feedforward_layernorm): Gemma3RMSNorm((2560,), eps=1e-06)
        )
      )
      (norm): Gemma3RMSNorm((2560,), eps=1e-06)
      (rotary_emb): Gemma3RotaryEmbedding()
      (rotary_emb_local): Gemma3RotaryEmbedding()
    )
  )
  (lm_head): Linear(in_features=2560, out_features=262208, bias=False)
) configuration cannot be auto-generated, use get_fsdp_config instead.
```