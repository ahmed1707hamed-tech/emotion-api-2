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
        self.input_shape = None

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            "haarcascade_frontalface_default.xml"
        )

    def load_model(self):

        try:

            path = hf_hub_download(
                repo_id="ahmed-hamed/emotion-image-model",
                filename="model.onnx",
                repo_type="model"
            )

            logger.info(
                "📦 Downloaded model: %s",
                path
            )

            self.session = ort.InferenceSession(
                path,
                providers=["CPUExecutionProvider"]
            )

            model_input = (
                self.session
                .get_inputs()[0]
            )

            self.input_name = model_input.name
            self.input_shape = model_input.shape

            logger.info(
                "✅ ONNX image model loaded."
            )

            logger.info(
                "🔌 Input name: %s",
                self.input_name
            )

            logger.info(
                "🧠 Model input shape: %s",
                self.input_shape
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

            x, y, w, h = max(
                faces,
                key=lambda f: f[2] * f[3]
            )

            face = image[
                y:y+h,
                x:x+w
            ]

            logger.info(
                "👀 Face detected"
            )

            if face.size > 0:
                return face

        logger.info(
            "🌚 No face detected → full image"
        )

        return image

    def _preprocess(self, face):

        gray = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.resize(
            gray,
            (48, 48)
        )

        gray = gray.astype(
            np.float32
        ) / 255.0

        # dynamic shape
        if self.input_shape[-1] == 1:

            img = gray.reshape(
                1,
                48,
                48,
                1
            )

        elif self.input_shape[-1] == 3:

            img = np.stack(
                [gray, gray, gray],
                axis=-1
            )

            img = img.reshape(
                1,
                48,
                48,
                3
            )

        else:

            img = gray.reshape(
                1,
                48,
                48,
                1
            )

        logger.info(
            "📐 Final input shape: %s",
            img.shape
        )

        return img.astype(np.float32)

    def predict(self, image_input):

        if self.session is None:

            logger.error(
                "❌ Model not loaded"
            )

            return {
                "label": "neutral",
                "confidence": 0.0
            }

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

                return {
                    "label": "neutral",
                    "confidence": 0.0
                }

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

            # top predictions
            top_indices = np.argsort(
                probs
            )[::-1]

            top1 = int(top_indices[0])
            top2 = int(top_indices[1])

            conf1 = float(probs[top1])
            conf2 = float(probs[top2])

            logger.info(
                "🎯 Top1=class_%s %.4f | Top2=class_%s %.4f",
                top1,
                conf1,
                top2,
                conf2
            )

            return {
                "label": f"class_{top1}",
                "confidence": round(conf1, 4),
                "top2": f"class_{top2}",
                "top2_confidence": round(conf2, 4),
                "probs": probs.tolist()
            }

        except Exception as e:

            logger.error(
                "❌ Prediction error: %s",
                e
            )

            return {
                "label": "neutral",
                "confidence": 0.0
            }


image_model_service = ImageModelService()