import torch

def verify_environment():
    print(f"PyTorch Version: {torch.__version__}")

    if torch.backends.mps.is_available():
        print("MPS backend is available.")
        # MPS 장치에 텐서 생성 테스트
        try:
            mps_device = torch.device("mps")
            x = torch.ones(1, device=mps_device)
            print("Successfully created a tensor on the MPS device:")
            print(x)
        except Exception as e:
            print(f"An error occurred while testing the MPS device: {e}")
    else:
        print("MPS device not found.")

if __name__ == "__main__":
    verify_environment()

"""
출력:
MPS device is available.
Tensor on MPS device: tensor([1.], device='mps:0')
"""