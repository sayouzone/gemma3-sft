# 

- Keras/JAX 경로
  - Keras 3의 멀티-백엔드 기능과 JAX 백엔드를 결합하여 모델 병렬처리(Model Parallelism)를 통한 고성능 분산 학습을 구현
  - 성능 최적화와 분산 구성에 대한 세밀한 제어를 제공
- Hugging Face/PyTorch 경로
  - transformers, TRL, PEFT 등 광범위한 Hugging Face 생태계와 optimum-tpu 라이브러리를 결합하여 PyTorch/XLA 환경에서 완전 샤딩 데이터 병렬처리(Fully Sharded Data Parallelism, FSDP)를 활성화하는 방식
  - 기존 Hugging Face 워크플로우와의 통합 용이성과 높은 수준의 추상화를 통한 개발 편의성

[Fine-Tune GEMMA on COLAB TPU](https://github.com/frank-morales2020/MLxDL/blob/main/Gemma_Finetuning_TPU_COLAB.ipynb)

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

**SSH으로 TPU VM 연결**

```bash
gcloud compute tpus tpu-vm ssh $TPU_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE
```

**TPU VM 인스턴스 삭제**

```bash
gcloud compute tpus tpu-vm delete $TPU_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE
```

**TPU VM 목록 확인**

```bash
gcloud compute tpus tpu-vm list \
    --project=$PROJECT_ID \
    --zone=$ZONE
```

##

```bash
sudo apt update
sudo apt install -y python3.10-venv screen

python3 -m venv gemma3-venv
source gemma3-venv/bin/activate

mkdir gemma3
cd gemma3
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
PJRT_DEVICE=TPU python tpu-test-gemma-2b.py

Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 75.15it/s]
{'fsdp': 'full_shard', 'fsdp_config': {'transformer_layer_cls_to_wrap': ['GemmaDecoderLayer'], 'xla': True, 'xla_fsdp_v2': True, 'xla_fsdp_grad_ckpt': True}}
Number of available JAX devices (TPU cores): 4
Number of available devices (TPU): 1
Reloading model and tokenizer: google/gemma-2b
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 76.83it/s]
Adding EOS to train dataset: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 2508/2508 [00:00<00:00, 22385.71 examples/s]
Tokenizing train dataset: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 2508/2508 [00:00<00:00, 4129.65 examples/s]
Truncating train dataset: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 2508/2508 [00:00<00:00, 300740.88 examples/s]
XLA_USE_BF16 will be deprecated after the 2.5 release, please convert your model to bf16 directly


  4%|██████▏                                                                                                                                    | 14/313 [13:41<4:44:46, 57.15s/it

{'loss': 2.7, 'grad_norm': 0.70703125, 'learning_rate': 1.6869009584664538e-05, 'num_tokens': 18278.0, 'mean_token_accuracy': 0.511875, 'epoch': 0.16}                             
 16%|██████████████████████▏                                                                                                                    | 50/313 [50:20<2:53:30, 39.59s/it]

100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 313/313 [3:54:31<00:00, 31.80s/it]
```

```bash
pip install tensorflow-datasets -q
pip install flax -q
pip install optax -q
pip install tensorflow_cpu -q
```

```bash
Traceback (most recent call last):
  File "/home/sjkim/gemma3/tpu-test-gemma3.py", line 36, in <module>
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

```bash
python3 test_jax_devices.py 

JAX version: 0.6.2
Optax version: 0.2.5
TensorFlow version (used for data loading): 2.20.0
Found JAX devices: [TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0), TpuDevice(id=1, process_index=0, coords=(1,0,0), core_on_chip=0), TpuDevice(id=2, process_index=0, coords=(0,1,0), core_on_chip=0), TpuDevice(id=3, process_index=0, coords=(1,1,0), core_on_chip=0)]
Number of TPU devices available: 4
```