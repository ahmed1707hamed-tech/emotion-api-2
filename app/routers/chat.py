import os
import uuid
import logging
import time
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from app.services.image_service import image_model_service
from app.services.audio_service import audio_model_service
from app.services.text_service import text_model_service
from app.services.fusion_service import fuse_emotions
from app.services.groq_service import generate_response, client

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chat"])

# --- EMOTION MEMORY CONFIG ---
EMOTION_MEMORY = {}  # session_id -> {"emotion": str, "timestamp": float, "counter": int}
MEMORY_TIMEOUT = 15 * 60  # 15 minutes
MAX_UNRELATED_MESSAGES = 3

FOLLOW_UP_PHRASES = [
    "what should i do", "help me", "how do i feel better",
    "how to fix this", "what can i do", "what is the solution",
    "ايه الحل", "اعمل ايه", "ساعدني", "اعمل ايه عشان اتحسن",
    "ازاي احسن نفسي", "مش عارف اعمل ايه"
]

def is_emotional_follow_up(text: str) -> bool:
    """Check if the user message is seeking emotional help."""
    t = text.lower().strip()
    return any(phrase in t for phrase in FOLLOW_UP_PHRASES)

def check_emotional_intent(text: str) -> bool:
    """Use Groq to classify message as EMOTIONAL or NORMAL."""
    try:
        prompt = (
            "Determine if this message expresses the user's personal emotional or psychological state.\n"
            "Include feelings like anxiety, worry, nervousness, shock, and personal experiences.\n\n"
            "Return ONLY:\n"
            "EMOTIONAL (if expressing feelings, moods, or mental state)\n"
            "NORMAL (if purely factual, informational, or general conversation)\n\n"
            f"Text: \"{text}\""
        )
        
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5
        )
        
        answer = completion.choices[0].message.content.strip().upper()
        logger.info("🧠 LLM Intent Classifier: %s → %s", text, answer)
        return "EMOTIONAL" in answer
    except Exception as e:
        logger.error("LLM intent check failed: %s", e)
        return False


@router.post("/chat")
async def chat(
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    session_id: Optional[str] = Form("default"),
):
    clean_text = text.strip() if text else ""
    
    image_emotion = None
    audio_emotion = None
    text_emotion = None
    
    # Flag to determine if we should return an emotional response
    emotion_mode = False

    # --- SESSION MEMORY HANDLING ---
    now = time.time()
    session = EMOTION_MEMORY.get(session_id)
    
    # Cleanup stale memory
    if session:
        if now - session["timestamp"] > MEMORY_TIMEOUT or session["counter"] >= MAX_UNRELATED_MESSAGES:
            logger.info("🕒 Session memory expired for %s", session_id)
            EMOTION_MEMORY.pop(session_id, None)
            session = None

    context_emotion = None
    if session and is_emotional_follow_up(clean_text):
        context_emotion = session["emotion"]
        emotion_mode = True
        logger.info("🧠 Follow-up detected. Context emotion: %s", context_emotion)

    # 1. IMAGE PROCESSING
    if image is not None:
        try:
            image_bytes = await image.read()
            if len(image_bytes) > 0:
                image_emotion = image_model_service.predict(image_bytes)
                emotion_mode = True
                logger.info("🖼️ Image detected → %s", image_emotion)
        except Exception as e:
            logger.error("Image processing failed: %s", e)

    # 2. AUDIO PROCESSING
    if audio is not None:
        try:
            audio_bytes = await audio.read()
            if len(audio_bytes) > 0:
                temp_path = f"temp_{uuid.uuid4().hex}.wav"
                try:
                    with open(temp_path, "wb") as f:
                        f.write(audio_bytes)
                    audio_emotion = audio_model_service.predict(temp_path)
                    emotion_mode = True
                    logger.info("🎵 Audio detected → %s", audio_emotion)
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
        except Exception as e:
            logger.error("Audio processing failed: %s", e)

    # 3. TEXT PROCESSING
    if clean_text:
        # Step 1: Always run the text emotion model
        detected_label = text_model_service.detect_emotion(clean_text)
        
        if emotion_mode:
            # Media present or Follow-up detected
            # If this is a follow-up, the context_emotion takes precedence for text
            text_emotion = context_emotion if context_emotion else detected_label
        else:
            # Text-only: Use LLM as final gatekeeper
            if check_emotional_intent(clean_text):
                if detected_label != "neutral":
                    text_emotion = detected_label
                    emotion_mode = True
                else:
                    text_emotion = None
            else:
                text_emotion = None

    # 4. RESPONSE GENERATION
    final_emotion = "neutral"
    if emotion_mode:
        # 4. FUSION
        logger.info("DEBUG: text=%s, image=%s, audio=%s", text_emotion, image_emotion, audio_emotion)
        
        final_emotion = fuse_emotions(
            image_emotion=image_emotion,
            audio_emotion=audio_emotion,
            text_emotion=text_emotion,
        )
        
        # If this is a follow-up, the context_emotion MUST override the final fused result
        if context_emotion:
            final_emotion = context_emotion

        logger.info("FINAL_EMOTION: %s", final_emotion)

        # Empathetic Groq response with memory context
        llm_response = generate_response(
            emotion=final_emotion,
            user_text=clean_text,
            context_emotion=context_emotion
        )

        # Update memory if a strong emotion is detected
        if final_emotion != "neutral":
            EMOTION_MEMORY[session_id] = {
                "emotion": final_emotion,
                "timestamp": now,
                "counter": 0
            }
        elif session:
            # Increment counter for neutral messages
            session["counter"] += 1

        return {
            "emotion": final_emotion,
            "response": llm_response,
            "modalities": {
                "image": image_emotion,
                "audio": audio_emotion,
                "text": text_emotion,
            },
        }
    else:
        # NORMAL CHAT MODE
        if session:
            session["counter"] += 1 # Increment for unrelated chat
            
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