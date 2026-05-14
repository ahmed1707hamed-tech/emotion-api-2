import logging
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
            # Download ONNX model from HF Hub
            model_path = hf_hub_download(
                repo_id="ahmed-hamed/emotion-api-2",
                filename="audio/audio_model.onnx",
                repo_type="space"
            )

            logger.info("📦 Downloaded audio model: %s", model_path)

            # Download encoder
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
            # Load audio (Force 16kHz for consistency)
            y, sr = librosa.load(
                file_path,
                sr=16000,
                duration=3
            )

            # Pad short audio
            if len(y) < sr * 3:
                y = np.pad(
                    y,
                    (0, sr * 3 - len(y))
                )

            # Normalize
            y = librosa.util.normalize(y)

            # MFCC features
            mfcc = librosa.feature.mfcc(
                y=y,
                sr=sr,
                n_mfcc=40
            )

            features = np.mean(
                mfcc.T,
                axis=0
            ).reshape(1, 40).astype(np.float32)

            # ONNX inference
            outputs = self.session.run(
                None,
                {self.input_name: features}
            )

            preds = outputs[0][0]

            # If outputs look like logits (not between 0-1), apply softmax
            if np.max(preds) > 1.0 or np.min(preds) < 0.0:
                exp_preds = np.exp(preds - np.max(preds))
                preds = exp_preds / exp_preds.sum()
                logger.info("ℹ️ Applied Softmax to logits")

            # Map labels manually if the encoder is mismatched (Model has 7 outputs)
            # Standard RAVDESS 7-class mapping: 0:neutral, 1:calm, 2:happy, 3:sad, 4:angry, 5:fear, 6:disgust
            AUDIO_LABELS = ["neutral", "calm", "happy", "sad", "angry", "fear", "disgust"]
            
            idx = int(np.argmax(preds))
            conf = float(np.max(preds))
            
            if idx < len(AUDIO_LABELS):
                label = AUDIO_LABELS[idx]
            else:
                label = "neutral" # Fallback

            # Map 'calm' to 'neutral' for our API
            if label == "calm":
                label = "neutral"

            # REQUIRED LOGS FOR DEBUGGING
            logger.info("AUDIO_PROBS: %s", preds.tolist())
            logger.info("AUDIO_CONFIDENCE: %.4f", conf)
            logger.info("AUDIO_PREDICTION: %s", label)

            # --- CONFIDENCE LOGIC ---
            # Stricter threshold for 'happy' to avoid over-prediction
            THRESHOLD = 0.6 if label == "happy" else 0.45

            if conf < THRESHOLD:
                logger.info("⚠️ Low confidence (%.2f < %.2f) → Fallback to neutral", conf, THRESHOLD)
                return "neutral", conf

            return label, conf

        except Exception as e:
            logger.error(
                "❌ Audio prediction error: %s",
                e
            )

            return None, 0.0


audio_model_service = AudioModelService()