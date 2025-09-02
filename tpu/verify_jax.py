# verify_jax.py
import jax

try:
    device_count = jax.device_count()
    print(f"JAX has access to {device_count} TPU cores.")

    # 사용 가능한 디바이스 목록 출력
    devices = jax.devices()
    print("Available devices:")
    for device in devices:
        print(f"- {device}")

except RuntimeError as e:
    print(f"An error occurred: {e}")
    print("Please ensure you are running on a TPU VM and libtpu is correctly installed.")