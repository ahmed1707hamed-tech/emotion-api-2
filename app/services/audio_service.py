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
            # Load audio
            y, sr = librosa.load(
                file_path,
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

            logger.info(
                "📊 Audio predictions: %s",
                preds
            )

            idx = int(np.argmax(preds))

            conf = float(np.max(preds))

            label = str(
                self.encoder.classes_[idx]
            ).lower()

            logger.info(
                "🎯 Audio prediction: %s | Confidence: %.4f",
                label,
                conf
            )

            # Low confidence fallback
            if conf < 0.45:
                return "neutral", conf

            return label, conf

        except Exception as e:
            logger.error(
                "❌ Audio prediction error: %s",
                e
            )

            return None, 0.0


audio_model_service = AudioModelService()