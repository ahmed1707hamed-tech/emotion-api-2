from onnxruntime.quantization import quantize_dynamic, QuantType

# input model
input_model = "emotion-models/onnx/model.onnx"

# output model
output_model = "emotion-models/onnx/model_quant.onnx"

quantize_dynamic(
    input_model,
    output_model,
    weight_type=QuantType.QInt8
)

print("✅ Quantization done!")