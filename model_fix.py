import tensorflow as tf
import zipfile
import json
import tempfile
import os
import joblib
import numpy as np
import sys

def fix_image_model(src, dst):
    print(f"Fixing image model: {src} -> {dst}")
    with zipfile.ZipFile(src, "r") as z:
        config_data = json.load(z.open("config.json"))
        
        def clean_config(obj):
            if isinstance(obj, dict):
                if "batch_shape" in obj:
                    obj["batch_input_shape"] = obj.pop("batch_shape")
                obj.pop("module", None)
                obj.pop("registered_name", None)
                for v in obj.values():
                    clean_config(v)
            elif isinstance(obj, list):
                for item in obj:
                    clean_config(item)
        
        clean_config(config_data)
        
        # Reconstruct model
        k2_dict = {
            "class_name": config_data.get("class_name", "Sequential"),
            "config": config_data.get("config", config_data)
        }
        model = tf.keras.models.model_from_json(json.dumps(k2_dict))
        
        # Load weights
        with z.open("model.weights.h5") as wf:
            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
                tmp.write(wf.read())
                tmp_name = tmp.name
        
        try:
            # Manual weight injection for Keras 3 -> 2
            import h5py
            with h5py.File(tmp_name, "r") as f:
                for layer in model.layers:
                    group_path = f"layers/{layer.name}/vars"
                    if group_path in f:
                        vars_group = f[group_path]
                        weights = []
                        i = 0
                        while str(i) in vars_group:
                            weights.append(vars_group[str(i)][:])
                            i += 1
                        if weights:
                            layer.set_weights(weights)
            
            # Save as H5
            model.save(dst)
            print("DONE: Image model converted to H5 successfully")
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)

def fix_audio_encoder(dst):
    print(f"Recreating audio encoder -> {dst}")
    try:
        from sklearn.preprocessing import LabelEncoder
        import joblib
        import numpy as np
        
        encoder = LabelEncoder()
        # These are the known classes from the original encoder.pkl
        encoder.classes_ = np.array(["angry", "happy", "sad", "neutral"])
        
        joblib.dump(encoder, dst)
        print("DONE: Audio encoder recreated successfully")
    except Exception as e:
        print(f"ERROR: Failed to recreate encoder: {e}")

if __name__ == "__main__":
    if not os.path.exists("emotion-models/fixed_model_stable.h5"):
        fix_image_model("emotion-models/fixed_model.keras", "emotion-models/fixed_model_stable.h5")
    fix_audio_encoder("emotion-models/encoder_stable.pkl")


