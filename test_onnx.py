import onnxruntime as ort
import numpy as np

model_path = "emotion-models/onnx/model.onnx"

session = ort.InferenceSession(model_path)

input_info = session.get_inputs()[0]

print("Input Name:", input_info.name)
print("Input Shape:", input_info.shape)
print("Input Type:", input_info.type)

# 👇 أهم سطر
input_size = input_info.shape[1]

# test input
dummy = np.random.rand(1, input_size).astype(np.float32)

output = session.run(None, {input_info.name: dummy})

print("Output:", output)