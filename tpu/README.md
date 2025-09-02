#

TPU(Tensor Processing Unit)는 머신러닝 워크로드를 실행

## TPU VM 인스턴스

```bash
gcloud services enable tpu.googleapis.com
```

```bash
gcloud compute tpus tpu-vm create gemma3-tpu-v4-test \
    --project=sayouzone-ai \
    --zone=us-central1-b \
    --accelerator-type=v3-8 \
    --version=tpu-vm-v4-pt-2.0
```

```
v3-8
tpu-vm-v4-pt-2.0
```

```
Estimated monthly total US$3,285.00
~730 hours per month
Hourly rate US$4.50
```

```bash
sudo apt update
#sudo apt -y upgrade
sudo apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev
sudo apt install -y nano screen
```

**Python 3.12 Ubuntu 패키지 설치**

```bash
sudo apt install -y python3.9 python3.9-venv python3-pip
```

```bash
python3 -m venv gemma3-env
source gemma3-env/bin/activate
```

```bash
mkdir gemma3
cd gemma3
```

```bash
# PyTorch/XLA 및 기본 도구 설치
pip install torch torch_xla

# Hugging Face 생태계 라이브러리 설치
pip install transformers datasets accelerate evaluate peft trl optimum-tpu
```



us-central2-b가 없음

```bash
gcloud compute tpus tpu-vm create gemma3-tpu-v4-test \
    --project=sayouzone-ai \
    --zone=us-central2-b \
    --accelerator-type=v5litepod-8 \
    --version=tpu-vm-v4-pt-2.0
```

```bash
gcloud compute tpus tpu-vm ssh gemma3-tpu-v4-test --zone=us-central2-b
```

```bash
# PyTorch/XLA 및 기본 도구 설치
pip install torch~=2.3.0 torch_xla[tpu]~=2.3.0 -f https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.3.0-cp310-cp310-linux_x86_64.whl

# Hugging Face 생태계 라이브러리 설치
pip install transformers==4.53.1 datasets==2.21.0 accelerate==0.31.0 evaluate==0.4.2
pip install peft==0.11.0 trl==0.9.4 optimum-tpu
```

```bash
huggingface-cli login
```

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
ERROR: (gcloud.compute.tpus.tpu-vm.create) INVALID_ARGUMENT: Cloud TPU received a bad request. The requested accelerator configuration for accelerator type, "v5litepod-8", could not be found in the zonal accelerator configurations for "us-central1-b".; the accelerator v5litepod-8 was not found in zone us-central1-b [EID: 0x5be9ca996f9773a4]
```

```bash
ERROR: Could not find a version that satisfies the requirement torch_xla[tpu]~=2.3.0 (from versions: none)
ERROR: No matching distribution found for torch_xla[tpu]~=2.3.0
```

```bash
ERROR: Could not find a version that satisfies the requirement torch-xla[tpu]==2.5.1 (from optimum-tpu) (from versions: none)
ERROR: No matching distribution found for torch-xla[tpu]==2.5.1 (from optimum-tpu)
```