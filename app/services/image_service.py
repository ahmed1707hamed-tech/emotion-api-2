import cv2
import numpy as np
import logging

from deepface import DeepFace

logger = logging.getLogger(__name__)


class ImageModelService:

    def __init__(self):

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

        logger.info(
            "✅ DeepFace model ready"
        )

    def predict(self, image_input):

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

            result = DeepFace.analyze(
                image,
                actions=["emotion"],
                enforce_detection=False
            )

            if isinstance(result, list):
                result = result[0]

            emotion = result["dominant_emotion"]

            logger.info(
                "🎯 DeepFace emotion: %s",
                emotion
            )

            return emotion.lower()

        except Exception as e:

            logger.error(
                "❌ Image prediction error: %s",
                e
            )

            return "neutral"


image_model_service = ImageModelService()