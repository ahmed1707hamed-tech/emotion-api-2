import cv2
import numpy as np
import logging
import onnxruntime as ort

from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)


class ImageModelService:
    def __init__(self):
        self.session = None
        self.input_name = None

        # Face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # FER2013 labels
        self.labels = [
            "angry",
            "disgust",
            "fear",
            "happy",
            "sad",
            "surprise",
            "neutral"
        ]

    def load_model(self):
        try:
            # Download ONNX model from Hugging Face Hub
            path = hf_hub_download(
                repo_id="ahmed-hamed/emotion-image-model",
                filename="model.onnx",
                repo_type="model"
            )

            logger.info("📦 Downloaded model: %s", path)

            # Load ONNX session
            self.session = ort.InferenceSession(path)

            self.input_name = self.session.get_inputs()[0].name

            logger.info("✅ ONNX image model loaded.")
            logger.info("🔌 Input name: %s", self.input_name)

        except Exception as e:
            logger.error("❌ Image ONNX load failed: %s", e)
            self.session = None

    def _detect_face(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        if len(faces) > 0:
            logger.info("👀 Face detected")

            x, y, w, h = max(
                faces,
                key=lambda f: f[2] * f[3]
            )

            face = image[y:y + h, x:x + w]

            if face.size == 0:
                logger.warning("⚠️ Empty face crop")
                return image

            return face

        logger.info("🌚 No face detected, fallback to full image")

        return image

    def _preprocess(self, face):
        # grayscale
        gray = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2GRAY
        )

        # histogram equalization
        gray = cv2.equalizeHist(gray)

        # resize
        gray = cv2.resize(gray, (48, 48))

        # 3 channels
        img = np.stack(
            [gray, gray, gray],
            axis=-1
        )

        # float32
        img = img.astype(np.float32)

        # normalize
        img = img / 255.0

        # batch dimension
        img = np.expand_dims(img, axis=0)

        logger.info(
            "📐 Input shape: %s | Min: %.4f | Max: %.4f",
            img.shape,
            img.min(),
            img.max()
        )

        return img

    def predict(self, image_input):
        if self.session is None:
            logger.error("❌ Image model not loaded")
            return "neutral"

        try:
            # bytes → image
            if isinstance(image_input, bytes):

                nparr = np.frombuffer(
                    image_input,
                    np.uint8
                )

                image = cv2.imdecode(
                    nparr,
                    cv2.IMREAD_COLOR
                )

            else:
                image = image_input

            if image is None:
                return "neutral"

            # detect face
            face = self._detect_face(image)

            # preprocess
            img = self._preprocess(face)

            # inference
            outputs = self.session.run(
                None,
                {self.input_name: img}
            )

            preds = outputs[0][0]

            logger.info(
                "📊 Raw predictions: %s",
                preds
            )

            # softmax if needed
            if not np.isclose(
                np.sum(preds),
                1.0,
                atol=1e-3
            ):
                exp_preds = np.exp(
                    preds - np.max(preds)
                )

                preds = exp_preds / exp_preds.sum()

            idx = int(np.argmax(preds))

            conf = float(np.max(preds))

            label = self.labels[idx]

            logger.info(
                "🎯 Prediction: %s | Confidence: %.4f",
                label,
                conf
            )

            # angry bias correction
            sorted_indices = np.argsort(preds)[::-1]

            if label == "angry" and len(sorted_indices) > 1:

                second_idx = int(sorted_indices[1])

                second_conf = float(preds[second_idx])

                if (conf - second_conf) < 0.15:

                    label = self.labels[second_idx]

                    conf = second_conf

                    logger.info(
                        "⚖️ Bias corrected to: %s",
                        label
                    )

            if conf < 0.20:
                return "neutral"

            return label

        except Exception as e:
            logger.error(
                "❌ Image prediction error: %s",
                e
            )

            return "neutral"


image_model_service = ImageModelService()