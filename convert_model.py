import tensorflow as tf

print("Trying to load model safely...")

model = tf.keras.models.load_model(
    "emotion-models/fixed_model.keras",
    compile=False,
    safe_mode=False
)

print("Loaded successfully")

model.save("emotion-models/fixed_model_v2.keras")

print("Saved new compatible model")