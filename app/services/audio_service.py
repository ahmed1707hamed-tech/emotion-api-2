import logging
import numpy as np
import librosa
import onnxruntime as ort
import joblib

logger = logging.getLogger(__name__)

class AudioModelService:
    def __init__(self):
        self.session = None
        self.encoder = None
        self.input_name = None

    def load_model(self, model_path: str, encoder_path: str):
        try:
            # Use fixed assets
            if "stable" not in encoder_path:
                encoder_path = encoder_path.replace(".pkl", "_stable.pkl")

            self.session = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"]
            )
            self.encoder = joblib.load(encoder_path)
            self.input_name = self.session.get_inputs()[0].name
            logger.info("✅ Audio model loaded successfully")

        except Exception as e:
            logger.error("❌ Audio load error: %s", e)
            self.session = None

    def predict(self, file_path: str):
        if self.session is None:
            return None, 0.0

        try:
            y, sr = librosa.load(file_path, duration=3)

            # Handle short audio with padding
            if len(y) < sr * 3:
                y = np.pad(y, (0, sr * 3 - len(y)))

            # Normalize audio
            y = librosa.util.normalize(y)

            # Extract MFCC: n_mfcc=40, mean aggregation
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            features = np.mean(mfcc.T, axis=0).reshape(1, 40).astype(np.float32)

            preds = self.session.run(None, {self.input_name: features})[0][0]

            idx = int(np.argmax(preds))
            conf = float(np.max(preds))
            label = str(self.encoder.classes_[idx]).lower()

            if conf < 0.45:
                return "neutral", conf

            return label, conf

        except Exception as e:
            logger.error("❌ Audio prediction error: %s", e)
            return None, 0.0

audio_model_service = AudioModelService()