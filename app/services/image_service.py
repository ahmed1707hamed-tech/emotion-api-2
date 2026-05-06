import cv2
import numpy as np
import logging
import tensorflow as tf
from tensorflow.keras.models import load_model

logger = logging.getLogger(__name__)

class ImageModelService:
    def __init__(self):
        self.model = None

        # OpenCV Haar Cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Labels for FER-2013 model
        # Standard Order: 0:angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
        self.labels = [
            "angry", "disgust", "fear",
            "happy", "sad", "surprise", "neutral"
        ]

    def load_model(self):
        # Using the stabilized H5 model for production-ready loading
        path = "emotion-models/fixed_model_stable.h5"
        try:
            self.model = load_model(path, compile=False)
            logger.info("✅ Image model loaded successfully. Input shape: %s", self.model.input_shape)
            # Log layer names to check for Rescaling
            layer_names = [l.name for l in self.model.layers]
            logger.info("🔍 Model layers: %s", layer_names)
        except Exception as e:
            logger.error("❌ Image load failed: %s", e)
            self.model = None

    def _detect_face(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            logger.info("👀 Face detected! Cropping...")
            # Pick the largest face (by area)
            x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
            face = image[y:y+h, x:x+w]
            
            if face.size == 0:
                logger.warning("⚠️ Empty face detected, falling back to full image.")
                return image
            return face

        logger.info("🌚 No face detected, using full image as fallback.")
        return image

    def _preprocess(self, face):
        # EXACT PIPELINE MATCHING TRAINING:
        # 1. Convert to grayscale
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        # 2. Histogram equalization (Try with and without - user asked to verify)
        # Note: EqualizeHist often helps diversity in lighting-sensitive models
        gray = cv2.equalizeHist(gray)
        
        # 3. Resize to (48, 48)
        gray = cv2.resize(gray, (48, 48))

        # 4. Convert to 3 channels (Stacking) - Model expected shape is (None, 48, 48, 3)
        img = np.stack([gray, gray, gray], axis=-1)
        
        # 5. Convert to float32
        img = img.astype("float32")
        
        # 6. RESCALING CHECK: 
        # The model has a 'rescaling' layer as the first layer.
        # This layer usually performs (x * scale) + offset.
        # If scale is 1/255, it expects [0, 255] input.
        # If we divide by 255 here, it will divide AGAIN, resulting in near-zero values.
        # CRITICAL FIX: Do NOT divide by 255 here.
        logger.info("📐 Input range before model: Min: %.2f | Max: %.2f", img.min(), img.max())
        
        # 7. Add batch dimension
        img = np.expand_dims(img, axis=0)

        return img

    def predict(self, image_input):
        if self.model is None:
            logger.error("❌ Image model not loaded.")
            return "neutral", 0.0

        try:
            # Handle bytes input
            if isinstance(image_input, bytes):
                nparr = np.frombuffer(image_input, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                image = image_input

            if image is None:
                return "neutral", 0.0

            # 1. Face detection
            face = self._detect_face(image)
            
            # 2. Preprocessing
            img = self._preprocess(face)

            # 3. Model Prediction
            preds = self.model.predict(img, verbose=0)[0]
            
            # Softmax check: If values don't sum to ~1, apply softmax
            if not np.isclose(np.sum(preds), 1.0, atol=1e-3):
                exp_preds = np.exp(preds - np.max(preds))
                preds = exp_preds / exp_preds.sum()
                logger.info("🛠️ Applied manual softmax.")
            
            logger.info("📊 Raw prediction array: %s", preds)

            # 4. Label selection
            idx = int(np.argmax(preds))
            conf = float(np.max(preds))
            label = self.labels[idx]
            
            logger.info("🎯 Selected Index: %d | Label: %s | Conf: %.2f", idx, label, conf)

            # 5. Bias correction for "angry" (index 0)
            # If angry is top but another emotion is close, be skeptical
            sorted_indices = np.argsort(preds)[::-1]
            if label == "angry" and len(sorted_indices) > 1:
                second_idx = int(sorted_indices[1])
                second_conf = float(preds[second_idx])
                if (conf - second_conf) < 0.15:
                    logger.info("⚖️ Bias Correction: Angry is close to %s, choosing %s", self.labels[second_idx], self.labels[second_idx])
                    label = self.labels[second_idx]
                    conf = second_conf

            # 6. Confidence Logic
            if conf < 0.20:
                return "neutral", conf

            return label, conf

        except Exception as e:
            logger.error("❌ Image prediction error: %s", e)
            return "neutral", 0.0

image_model_service = ImageModelService()