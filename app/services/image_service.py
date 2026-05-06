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

        # face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            "haarcascade_frontalface_default.xml"
        )

        # IMPORTANT:
        # must match training order exactly
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

            # download fp32 model
            path = hf_hub_download(
                repo_id="ahmed-hamed/emotion-image-model",
                filename="model.onnx",
                repo_type="model"
            )

            logger.info(
                "📦 Downloaded model: %s",
                path
            )

            # load onnx session
            self.session = ort.InferenceSession(
                path,
                providers=["CPUExecutionProvider"]
            )

            self.input_name = (
                self.session
                .get_inputs()[0]
                .name
            )

            logger.info(
                "✅ ONNX image model loaded."
            )

            logger.info(
                "🔌 Input name: %s",
                self.input_name
            )

        except Exception as e:

            logger.error(
                "❌ Image ONNX load failed: %s",
                e
            )

            self.session = None

    def _detect_face(self, image):

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(60, 60)
        )

        if len(faces) > 0:

            logger.info("👀 Face detected")

            x, y, w, h = max(
                faces,
                key=lambda f: f[2] * f[3]
            )

            face = image[
                y:y + h,
                x:x + w
            ]

            if face.size > 0:
                return face

        logger.info(
            "🌚 No face detected → full image"
        )

        return image

    def _preprocess(self, face):

        # grayscale
        gray = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2GRAY
        )

        # EXACT training resize
        gray = cv2.resize(
            gray,
            (48, 48)
        )

        # normalize
        gray = gray.astype(
            np.float32
        ) / 255.0

        # exact model shape
        img = gray.reshape(
            1,
            48,
            48,
            1
        )

        logger.info(
            "📐 Shape: %s | Min=%.4f | Max=%.4f",
            img.shape,
            img.min(),
            img.max()
        )

        return img

    def predict(self, image_input):

        if self.session is None:

            logger.error(
                "❌ Model not loaded"
            )

            return "neutral"

        try:

            # bytes → image
            if isinstance(
                image_input,
                bytes
            ):

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
            face = self._detect_face(
                image
            )

            # preprocess
            img = self._preprocess(
                face
            )

            # inference
            outputs = self.session.run(
                None,
                {
                    self.input_name: img
                }
            )

            preds = outputs[0][0].astype(
                np.float32
            )

            logger.info(
                "📊 Raw predictions: %s",
                preds
            )

            # softmax
            exp_preds = np.exp(
                preds - np.max(preds)
            )

            probs = (
                exp_preds /
                np.sum(exp_preds)
            )

            logger.info(
                "📈 Softmax probs: %s",
                probs
            )

            # best prediction
            idx = int(
                np.argmax(probs)
            )

            conf = float(
                probs[idx]
            )

            label = self.labels[idx]

            logger.info(
                "🎯 Prediction: %s | %.4f",
                label,
                conf
            )

            return label

        except Exception as e:

            logger.error(
                "❌ Prediction error: %s",
                e
            )

            return "neutral"


image_model_service = ImageModelService()