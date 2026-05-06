import tensorflow as tf

print("Loading model...")

model = tf.keras.models.load_model(
    "emotion-models/fixed_model.keras",
    compile=False,
    safe_mode=False  # 🔥 ده الحل
)

print("Model loaded successfully")

model.save("emotion-models/fixed_model_fixed.keras")

print("✅ Model re-saved successfully")