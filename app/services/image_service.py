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

        # Labels
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

            logger.info("📦 Downloaded model: %s", path)

            self.session = ort.InferenceSession(
                path,
                providers=["CPUExecutionProvider"]
            )

            self.input_name = self.session.get_inputs()[0].name

            logger.info("✅ ONNX image model loaded.")
            logger.info("🔌 Input name: %s", self.input_name)

        except Exception as e:
            logger.error("❌ Image ONNX load failed: %s", e)
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

            x, y, w, h = max(
                faces,
                key=lambda f: f[2] * f[3]
            )

            face = image[y:y+h, x:x+w]

            logger.info("👀 Face detected")

            if face.size > 0:
                return face

        logger.info("🌚 No face detected → full image")

        return image

    def _preprocess(self, face):

        gray = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2GRAY
        )

        # CLAHE better than equalizeHist
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        gray = clahe.apply(gray)

        gray = cv2.resize(
            gray,
            (48, 48)
        )

        gray = gray.astype(np.float32) / 255.0

        # shape => (1,48,48,1)
        img = np.expand_dims(
            gray,
            axis=(0, -1)
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
            logger.error("❌ Model not loaded")
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

            preds = outputs[0][0].astype(np.float32)

            logger.info(
                "📊 Raw predictions: %s",
                preds
            )

            # stable softmax
            exp_preds = np.exp(
                preds - np.max(preds)
            )

            probs = exp_preds / np.sum(exp_preds)

            logger.info(
                "📈 Softmax probs: %s",
                probs
            )

            # top predictions
            top_indices = np.argsort(probs)[::-1]

            top1 = int(top_indices[0])
            top2 = int(top_indices[1])

            label1 = self.labels[top1]
            label2 = self.labels[top2]

            conf1 = float(probs[top1])
            conf2 = float(probs[top2])

            logger.info(
                "🎯 Top1=%s %.3f | Top2=%s %.3f",
                label1,
                conf1,
                label2,
                conf2
            )

            # weak confidence
            if conf1 < 0.35:
                return "neutral"

            # angry bias fix
            if label1 == "angry":

                if (conf1 - conf2) < 0.12:

                    logger.info(
                        "⚖️ Angry bias corrected → %s",
                        label2
                    )

                    return label2

            return label1

        except Exception as e:

            logger.error(
                "❌ Image prediction error: %s",
                e
            )

            return "neutral"


image_model_service = ImageModelService()