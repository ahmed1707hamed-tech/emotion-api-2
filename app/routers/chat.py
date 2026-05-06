"""
Chat Router
Unified POST /chat endpoint that accepts text, image, and audio,
runs emotion detection on each modality, fuses them, and returns
an empathetic LLM response.
"""

import os
import uuid
import logging
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from app.services.image_service import image_model_service
from app.services.audio_service import audio_model_service
from app.services.text_service import text_model_service
from app.services.fusion_service import fuse_emotions
from app.services.groq_service import generate_response

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chat"])


@router.post("/chat", summary="Multimodal emotion chat", response_description="Emotion analysis and empathetic response")
async def chat(
    text: Optional[str] = Form(None, description="User text message"),
    image: Optional[UploadFile] = File(None, description="Face image (jpg/png)"),
    audio: Optional[UploadFile] = File(None, description="Voice recording (.wav)"),
):
    """
    Analyze emotion from one or more modalities (text, image, audio),
    fuse the results, and return an empathetic AI response.

    At least one modality must be provided.
    """
    image_emotion = None
    audio_emotion = None
    text_emotion = None

    # ── IMAGE ──────────────────────────────────────────────
    if image is not None:
        try:
            image_bytes = await image.read()
            if len(image_bytes) > 0:
                image_emotion = image_model_service.predict(image_bytes)
                logger.info("🖼️  Image → %s", image_emotion)
        except Exception as e:
            logger.error("Image processing failed: %s", e)

    # ── AUDIO ──────────────────────────────────────────────
    if audio is not None:
        try:
            audio_bytes = await audio.read()
            if len(audio_bytes) > 0:
                temp_path = f"temp_{uuid.uuid4().hex}.wav"
                try:
                    with open(temp_path, "wb") as f:
                        f.write(audio_bytes)
                    audio_emotion = audio_model_service.predict(temp_path)
                    logger.info("🎵 Audio → %s", audio_emotion)
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
        except Exception as e:
            logger.error("Audio processing failed: %s", e)

    # ── TEXT ───────────────────────────────────────────────
    clean_text = ""
    if text and text.strip() and text.strip().lower() != "string":
        clean_text = text.strip()
        text_emotion = text_model_service.detect_emotion(clean_text)
        logger.info("📝 Text → %s", text_emotion)

    # ── VALIDATE ──────────────────────────────────────────
    if image_emotion is None and audio_emotion is None and text_emotion is None:
        raise HTTPException(
            status_code=400,
            detail="At least one modality (text, image, or audio) must be provided.",
        )

    # ── FUSION ────────────────────────────────────────────
    final_emotion = fuse_emotions(
        image_emotion=image_emotion,
        audio_emotion=audio_emotion,
        text_emotion=text_emotion,
    )
    logger.info("🎯 Fused emotion: %s", final_emotion)

    # ── GROQ LLM ─────────────────────────────────────────
    llm_response = generate_response(emotion=final_emotion, user_text=clean_text)

    return {
        "emotion": final_emotion,
        "response": llm_response,
        "modalities": {
            "image": image_emotion,
            "audio": audio_emotion,
            "text": text_emotion,
        },
    }
