import logging
import os
import numpy as np
import librosa
import onnxruntime as ort
import joblib

from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)


class AudioModelService:
    def __init__(self):
        self.session = None
        self.encoder = None
        self.input_name = None

    def load_model(self, model_path: str = None, encoder_path: str = None):
        try:
            # Check for local files first
            local_model = "emotion-models/audio_model.onnx"
            local_encoder = "emotion-models/encoder_stable.pkl"
            
            if os.path.exists(local_model):
                model_path = local_model
            else:
                model_path = hf_hub_download(
                    repo_id="ahmed-hamed/emotion-api-2",
                    filename="audio/audio_model.onnx",
                    repo_type="space"
                )
                logger.info("📦 Downloaded audio model: %s", model_path)

            if os.path.exists(local_encoder):
                encoder_path = local_encoder
            else:
                encoder_path = hf_hub_download(
                    repo_id="ahmed-hamed/emotion-api-2",
                    filename="emotion-models/encoder_stable.pkl",
                    repo_type="space"
                )
                logger.info("📦 Downloaded encoder: %s", encoder_path)

            # Load ONNX session
            self.session = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"]
            )

            # Load encoder
            self.encoder = joblib.load(encoder_path)

            # Input name
            self.input_name = self.session.get_inputs()[0].name

            logger.info("✅ Audio model loaded successfully")
            logger.info("🔌 Input name: %s", self.input_name)

        except Exception as e:
            logger.error("❌ Audio load error: %s", e)
            self.session = None

    def predict(self, file_path: str):
        if self.session is None:
            return None, 0.0

        try:
            # --- PREPROCESSING (User Requirement 5) ---
            # Load audio
            audio, sr = librosa.load(
                file_path,
                sr=16000,
                mono=True
            )
            
            # Trim silence
            audio, _ = librosa.effects.trim(audio)
            
            # Normalize amplitude
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            
            # Pad/Truncate to 3s
            target_len = 16000 * 3
            if len(audio) < target_len:
                audio = np.pad(audio, (0, target_len - len(audio)))
            else:
                audio = audio[:target_len]

            # Float32 conversion
            audio = audio.astype(np.float32)

            # --- DEBUG LOGGING (User Requirement 2) ---
            print("=" * 50)
            print("AUDIO FILE:", file_path)
            print("SAMPLE RATE:", sr)
            print("AUDIO SHAPE:", audio.shape)
            print("AUDIO DTYPE:", audio.dtype)
            print("MIN/MAX:", audio.min(), audio.max())

            # MFCC Features
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=sr,
                n_mfcc=40
            )
            raw_features = np.mean(mfcc.T, axis=0).reshape(1, 40).astype(np.float32)
            
            # --- APPLY CMVN SCALING (New Fix) ---
            # Standardizing features to mean 0, std 1
            features = (raw_features - np.mean(raw_features)) / (np.std(raw_features) + 1e-6)
            
            print("MODEL_INPUT_SHAPE:", features.shape)
            print("RAW_FEATURES_SAMPLE:", raw_features[0][:5])
            print("SCALED_FEATURES_SAMPLE:", features[0][:5])
            print("FEATURES_MIN/MAX:", features.min(), features.max())

            # ONNX inference
            outputs = self.session.run(
                None,
                {self.input_name: features}
            )
            print("RAW_OUTPUT:", outputs)
            print("OUTPUT_SHAPE:", np.array(outputs[0]).shape)

            # --- FIX ONNX OUTPUT PARSING (User Requirement 3) ---
            logits = np.array(outputs[0]).squeeze()
            print("LOGITS:", logits)

            # --- APPLY SOFTMAX CORRECTLY ---
            exp_scores = np.exp(logits - np.max(logits))
            probs = exp_scores / exp_scores.sum()
            print("AUDIO_PROBS:", probs)

            pred_idx = int(np.argmax(probs))
            confidence = float(np.max(probs))

            print("ARGMAX:", pred_idx)
            print("CONFIDENCE:", confidence)
            print("ENCODER_CLASSES:", self.encoder.classes_)

            # --- FIX LABEL MAPPING (User Requirement 4) ---
            # User Requirement: Use ONLY inverse_transform
            emotion = self.encoder.inverse_transform([pred_idx])[0]

            print("FINAL_EMOTION:", emotion)
            print("=" * 50)

            # --- NO FALLBACKS (User Requirement 1) ---
            # Threshold disabled for debugging
            return emotion, confidence

        except Exception as e:
            logger.error(
                "❌ Audio prediction error: %s",
                e
            )

            return None, 0.0


audio_model_service = AudioModelService()