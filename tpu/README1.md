#

PyTorch를 사용하여 Cloud TPU VM에서 계산 실행

[PyTorch를 사용](https://cloud.google.com/tpu/docs/run-calculation-pytorch?hl=ko)

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
gcloud compute tpus tpu-vm create $TPU_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --accelerator-type=$ACCELERATOR_TYPE \
    --version=$RUNTIME_VERSION
```

####


```bash
export PROJECT_ID=sayouzone-ai
export TPU_NAME=gemma3-tpu-v4-test
export ZONE=us-central1-b
export ACCELERATOR_TYPE=v2-8
export RUNTIME_VERSION=tpu-vm-v4-pt-2.0
```

```bash
gcloud compute tpus tpu-vm create $TPU_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --accelerator-type=$ACCELERATOR_TYPE \
    --version=$RUNTIME_VERSION
```

####

```bash
export PROJECT_ID=sayouzone-ai
export TPU_NAME=gemma3-tpu-v4-test
export ZONE=us-central1-a
export ACCELERATOR_TYPE=v3-8
export RUNTIME_VERSION=tpu-vm-v4-pt-2.0
```

```bash
gcloud compute tpus tpu-vm create $TPU_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --accelerator-type=$ACCELERATOR_TYPE \
    --version=$RUNTIME_VERSION
```

####

```bash
gcloud compute tpus tpu-vm delete $TPU_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE
```

####

```bash
sudo apt-get update
sudo apt-get install libopenblas-dev -y
pip install numpy
pip install torch torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html
```

```bash
Installing collected packages: nvidia-cusparselt-cu12, mpmath, libtpu, typing-extensions, triton, sympy, pygments, protobuf, nvidia-nvtx-cu12, nvidia-nvjitlink-cu12, nvidia-nccl-cu12, nvidia-curand-cu12, nvidia-cufile-cu12, nvidia-cuda-runtime-cu12, nvidia-cuda-nvrtc-cu12, nvidia-cuda-cupti-cu12, nvidia-cublas-cu12, networkx, mdurl, grpcio, fsspec, absl-py, torch_xla, nvidia-cusparse-cu12, nvidia-cufft-cu12, nvidia-cudnn-cu12, markdown-it-py, rich, nvidia-cusolver-cu12, tpu-info, torch
  WARNING: The scripts proton and proton-viewer are installed in '/home/sjkim/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The script isympy is installed in '/home/sjkim/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The script pygmentize is installed in '/home/sjkim/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The script stablehlo-to-saved-model is installed in '/home/sjkim/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The script markdown-it is installed in '/home/sjkim/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The script tpu-info is installed in '/home/sjkim/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The scripts torchfrtrace and torchrun are installed in '/home/sjkim/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
Successfully installed absl-py-2.3.1 fsspec-2025.7.0 grpcio-1.74.0 libtpu-0.0.17 markdown-it-py-4.0.0 mdurl-0.1.2 mpmath-1.3.0 networkx-3.4.2 nvidia-cublas-cu12-12.8.4.1 nvidia-cuda-cupti-cu12-12.8.90 nvidia-cuda-nvrtc-cu12-12.8.93 nvidia-cuda-runtime-cu12-12.8.90 nvidia-cudnn-cu12-9.10.2.21 nvidia-cufft-cu12-11.3.3.83 nvidia-cufile-cu12-1.13.1.3 nvidia-curand-cu12-10.3.9.90 nvidia-cusolver-cu12-11.7.3.90 nvidia-cusparse-cu12-12.5.8.93 nvidia-cusparselt-cu12-0.7.1 nvidia-nccl-cu12-2.27.3 nvidia-nvjitlink-cu12-12.8.93 nvidia-nvtx-cu12-12.8.90 protobuf-6.32.0 pygments-2.19.2 rich-14.1.0 sympy-1.14.0 torch-2.8.0 torch_xla-2.8.0 tpu-info-0.5.1 triton-3.4.0 typing-extensions-4.14.1
```

```bash
#ACCELERATOR_TYPE=v3-8
python3 --version

Python 3.8.10
```

```bash
#ACCELERATOR_TYPE=v5litepod-4
python3 --version

Python 3.10.12
```

```bash
PJRT_DEVICE=TPU python3 -c "import torch_xla.core.xla_model as xm; print(xm.get_xla_supported_devices(\"TPU\"))"
```

```bash
['xla:0', 'xla:1', 'xla:2', 'xla:3', 'xla:4', 'xla:5', 'xla:6', 'xla:7']
```

```bash
PJRT_DEVICE=TPU python3 -c "import torch_xla; print(torch_xla.devicess())"
```

```bash
[device(type='xla', index=0), device(type='xla', index=1), device(type='xla', index=2), device(type='xla', index=3)]
```

```bash
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/home/sjkim/.local/lib/python3.10/site-packages/torch_xla/core/xla_model.py", line 82, in get_xla_supported_devices
    for i, _ in enumerate(devices[:max_devices] if max_devices else devices)
TypeError: slice indices must be integers or None or have an __index__ method
```

tpu-test.py

```python
#ACCELERATOR_TYPE=v3-8
import os
import torch
import torch_xla.core.xla_model as xm

os.environ["PJRT_DEVICE"] = "TPU"

dev = xm.xla_device()
t1 = torch.randn(3,3,device=dev)
t2 = torch.randn(3,3,device=dev)
print(t1 + t2)
```

```bash
WARNING:root:XRT configuration not detected. Defaulting to preview PJRT runtime. To silence this warning and continue using PJRT, explicitly set PJRT_DEVICE to a supported device or configure XRT. To disable default device selection, set PJRT_SELECT_DEFAULT_DEVICE=0
WARNING:root:For more information about the status of PJRT, see https://github.com/pytorch/xla/blob/master/docs/pjrt.md
WARNING:root:libtpu.so and TPU device found. Setting PJRT_DEVICE=TPU.
tensor([[-1.1846, -0.7140, -0.4168],
        [-0.3259, -0.5264, -0.8828],
        [-0.8562, -0.5813,  0.3264]], device='xla:0')
```

```bash
PJRT_DEVICE=TPU python3 tpu-test.py
```

```bash
tensor([[-1.1846, -0.7140, -0.4168],
        [-0.3259, -0.5264, -0.8828],
        [-0.8562, -0.5813,  0.3264]], device='xla:0')
```

```bash
/home/sjkim/tpu-test.py:4: DeprecationWarning: Use torch_xla.device instead
  dev = xm.xla_device()
tensor([[ 0.3355, -1.4628, -3.2610],
        [-1.4656,  0.3196, -2.8766],
        [ 0.8667, -1.5060,  0.7125]], device='xla:0'
```

```python
#ACCELERATOR_TYPE=v5litepod-4
import os
import torch
#import torch_xla.core.xla_model as xm
import torch_xla

os.environ["PJRT_DEVICE"] = "TPU"

#dev = xm.xla_device()
dev = torch_xla.device()
t1 = torch.randn(3,3,device=dev)
t2 = torch.randn(3,3,device=dev)
print(t1 + t2)
```

```bash
PJRT_DEVICE=TPU python3 tpu-test.py
```

## Errors

```bash
export PROJECT_ID=sayouzone-ai
export TPU_NAME=gemma3-tpu-v4-test
export ZONE=us-central1-b
export ACCELERATOR_TYPE=v3-8
export RUNTIME_VERSION=tpu-vm-v4-pt-2.0
```

```bash
gcloud compute tpus tpu-vm create $TPU_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --accelerator-type=$ACCELERATOR_TYPE \
    --version=$RUNTIME_VERSION

Create request issued for: [gemma3-tpu-v4-test]
Waiting for operation [projects/sayouzone-ai/locations/us-central1-b/operations/operation-1756107470419-63d2ba1f634
f4-1870ab42-710b6c51] to complete...failed.                                                                        
ERROR: (gcloud.compute.tpus.tpu-vm.create) {
  "code": 8,
  "message": "There is no more capacity in the zone \"us-central1-b\"; you can try in another zone where Cloud TPU Nodes are offered (see https://cloud.google.com/tpu/docs/regions) [EID: 0xe1d1ce9f2eae7a03]"
}
```


```bash
export PROJECT_ID=sayouzone-ai
export TPU_NAME=gemma3-tpu-v5-test
export ZONE=us-east5-a
export ACCELERATOR_TYPE=v5litepod-4
export RUNTIME_VERSION=v2-alpha-tpuv5-lite
```

```bash
gcloud compute tpus tpu-vm create $TPU_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --accelerator-type=$ACCELERATOR_TYPE \
    --version=$RUNTIME_VERSION

Create request issued for: [gemma3-tpu-v5-test]
Waiting for operation [projects/sayouzone-ai/locations/us-east5-a/operations/operation-1756109923719-63d2c34309864-
021a5aa5-4411702f] to complete...failed.                                                                           
ERROR: (gcloud.compute.tpus.tpu-vm.create) {
  "code": 5,
  "message": "Reservation not found"
}
```

```bash
export PROJECT_ID=sayouzone-ai
export TPU_NAME=gemma3-tpu-v5-test
export ZONE=us-east5-a
export ACCELERATOR_TYPE=v5litepod-8
export RUNTIME_VERSION=v2-alpha-tpuv5-lite
```

```bash
gcloud compute tpus tpu-vm create $TPU_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --accelerator-type=$ACCELERATOR_TYPE \
    --version=$RUNTIME_VERSION

ERROR: (gcloud.compute.tpus.tpu-vm.create) HttpError accessing <https://tpu.googleapis.com/v2/projects/sayouzone-ai/locations/us-east5-a/nodes?alt=json&nodeId=gemma3-tpu-v5-test>: response: <{'vary': 'Origin, X-Origin, Referer', 'content-type': 'application/json; charset=UTF-8', 'content-encoding': 'gzip', 'date': 'Mon, 25 Aug 2025 08:20:38 GMT', 'server': 'ESF', 'x-xss-protection': '0', 'x-frame-options': 'SAMEORIGIN', 'x-content-type-options': 'nosniff', 'alt-svc': 'h3=":443"; ma=2592000,h3-29=":443"; ma=2592000', 'transfer-encoding': 'chunked', 'status': 429}>, content <{
  "error": {
    "code": 429,
    "message": "Quota limit 'TPUV5sLitepodServingPerProjectPerZoneForTPUAPI' has been exceeded. Limit: 4 in zone us-east5-a.",
    "status": "RESOURCE_EXHAUSTED",
    "details": [
      {
        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
        "violations": [
          {
            "subject": "project:1037372895180",
            "description": "Quota 'TPUV5sLitepodServingPerProjectPerZoneForTPUAPI' exhausted. Limit 4 in zone us-east5-a"
          }
        ]
      }
    ]
  }
}
>
This may be due to network connectivity issues. Please check your network settings, and the status of the service you are trying to reach.
```
