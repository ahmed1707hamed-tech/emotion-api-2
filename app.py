"""
Chat Router
Smart multimodal emotion chat.
"""

import os
import uuid
import logging
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form

from app.services.image_service import image_model_service
from app.services.audio_service import audio_model_service
from app.services.text_service import text_model_service
from app.services.fusion_service import fuse_emotions
from app.services.groq_service import generate_response

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chat"])

# كلمات تدل على مشاعر
EMOTION_KEYWORDS = [
    "sad", "happy", "angry", "fear", "depressed",
    "anxious", "upset", "cry", "lonely",
    "حزين", "مبسوط", "زعلان", "مضايق",
    "مكتئب", "متوتر", "قلقان", "خايف"
]


@router.post("/chat")
async def chat(
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
):
    image_emotion = None
    audio_emotion = None
    text_emotion = None

    clean_text = text.strip() if text else ""

    # ==================================================
    # IMAGE → AUTO ANALYSIS
    # ==================================================
    if image is not None:
        try:
            image_bytes = await image.read()

            if len(image_bytes) > 0:
                image_emotion = image_model_service.predict(
                    image_bytes
                )

                logger.info(
                    "🖼️ Image → %s",
                    image_emotion
                )

        except Exception as e:
            logger.error(
                "Image processing failed: %s",
                e
            )

    # ==================================================
    # AUDIO → AUTO ANALYSIS
    # ==================================================
    if audio is not None:
        try:
            audio_bytes = await audio.read()

            if len(audio_bytes) > 0:
                temp_path = f"temp_{uuid.uuid4().hex}.wav"

                try:
                    with open(temp_path, "wb") as f:
                        f.write(audio_bytes)

                    audio_emotion = (
                        audio_model_service.predict(
                            temp_path
                        )
                    )

                    logger.info(
                        "🎵 Audio → %s",
                        audio_emotion
                    )

                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

        except Exception as e:
            logger.error(
                "Audio processing failed: %s",
                e
            )

    # ==================================================
    # TEXT → ONLY IF EMOTIONAL
    # ==================================================
    if clean_text:

        lower_text = clean_text.lower()

        contains_emotion = any(
            keyword in lower_text
            for keyword in EMOTION_KEYWORDS
        )

        if contains_emotion:
            text_emotion = (
                text_model_service
                .detect_emotion(clean_text)
            )

            logger.info(
                "📝 Text emotion → %s",
                text_emotion
            )

    # ==================================================
    # IF ANY EMOTION EXISTS → FUSE
    # ==================================================
    if (
        image_emotion
        or audio_emotion
        or text_emotion
    ):

        final_emotion = fuse_emotions(
            image_emotion=image_emotion,
            audio_emotion=audio_emotion,
            text_emotion=text_emotion,
        )

        logger.info(
            "🎯 Final emotion → %s",
            final_emotion
        )

        llm_response = generate_response(
            emotion=final_emotion,
            user_text=clean_text
        )

        return {
            "emotion": final_emotion,
            "response": llm_response,
            "modalities": {
                "image": image_emotion,
                "audio": audio_emotion,
                "text": text_emotion,
            },
        }

    # ==================================================
    # NORMAL CHAT → GROQ
    # ==================================================
    llm_response = generate_response(
        emotion="neutral",
        user_text=clean_text
    )

    return {
        "emotion": None,
        "response": llm_response,
        "modalities": {
            "image": None,
            "audio": None,
            "text": None,
        },
    }