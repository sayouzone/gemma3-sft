# verify_pytorch_xla.py
import torch
import torch_xla.core.xla_model as xm

if 'TPU_NAME' in __import__('os').environ:
    dev = xm.xla_device()
    print(f"PyTorch/XLA is using device: {dev}")

    # 간단한 텐서 연산 수행
    t1 = torch.randn(3, 3, device=dev)
    t2 = torch.randn(3, 3, device=dev)
    print("Sample tensor addition on TPU:")
    print(t1 + t2)

    # 사용 가능한 XLA 디바이스 수 확인
    num_devices = xm.xrt_world_size()
    print(f"Number of available XLA devices: {num_devices}")
else:
    print("TPU environment not detected.")