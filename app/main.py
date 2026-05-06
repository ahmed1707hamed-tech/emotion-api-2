import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.services.audio_service import audio_model_service
from app.services.text_service import text_model_service
from app.services.image_service import image_model_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(BASE, "emotion-models")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Loading models...")

    audio_model_service.load_model(
        os.path.join(MODELS, "onnx/model_quant.onnx"),
        os.path.join(MODELS, "encoder_stable.pkl")
    )

    text_model_service.load_model(
        os.path.join(MODELS, "vectorizer.pkl"),
        os.path.join(MODELS, "text_model.pkl")
    )

    image_model_service.load_model()

    logger.info("✅ READY")
    yield


app = FastAPI(lifespan=lifespan)

from app.routers.chat import router
app.include_router(router)