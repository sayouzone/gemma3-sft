# verify_tensorflow.py
import tensorflow as tf

try:
    tpu_devices = tf.config.list_logical_devices('TPU')
    if tpu_devices:
        print(f"TensorFlow can access {len(tpu_devices)} TPU cores.")

        # TPU 시스템 초기화 및 전략 생성
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
        strategy = tf.distribute.TPUStrategy(resolver)
        print("TPUStrategy created successfully.")

        @tf.function
        def simple_computation():
            return tf.constant(1.0) + tf.constant(2.0)

        result = strategy.run(simple_computation)
        print(f"Simple computation result on TPU: {result}")

    else:
        print("No TPU devices found.")
except Exception as e:
    print(f"An error occurred while trying to access TPU: {e}")