import tf2onnx
import tensorflow as tf

model = tf.keras.models.load_model(
    "emotion-models/fixed_model_stable.h5"
)

spec = (
    tf.TensorSpec(
        (None, 48, 48, 3),
        tf.float32,
        name="input"
    ),
)

model_proto, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset=13
)

with open("model_fp32.onnx", "wb") as f:
    f.write(model_proto.SerializeToString())

print("✅ FP32 ONNX model exported")