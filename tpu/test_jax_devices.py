import tensorflow_datasets as tfds # For loading datasets
import tensorflow as tf # Used for tfds data loading, not for model building
import time
import math # For math.inf
import numpy as np # For np.array
import warnings
warnings.filterwarnings("ignore")

print(f"JAX version: {jax.__version__}")
print(f"Optax version: {optax.__version__}")
print(f"TensorFlow version (used for data loading): {tf.__version__}")

# --- 1. TPU Initialization (JAX style) ---
try:
    devices = jax.devices()
    tpu_devices = [d for d in devices if d.platform == 'tpu']
    if not tpu_devices:
        raise ValueError("No TPU devices found.")
    print(f"Found JAX devices: {devices}")
    print(f"Number of TPU devices available: {len(tpu_devices)}")
except ValueError as e:
    print(f"ERROR: {e}. Please ensure your Colab runtime is set to TPU.")
    print("Go to 'Runtime' -> 'Change runtime type' and select 'TPU'.")
    raise SystemExit("TPU not found or not initialized for JAX.")

from jax.sharding import NamedSharding, PartitionSpec as P
# Create a Sharding object to distribute a value across devices:
mesh = jax.make_mesh((4, 1), ('x', 'y'))
# Create an array of random values:
x = jax.random.normal(jax.random.key(0), (8192, 8192))
# and use jax.device_put to distribute it across devices:
y = jax.device_put(x, NamedSharding(mesh, P('x', 'y')))
jax.debug.visualize_array_sharding(y)
