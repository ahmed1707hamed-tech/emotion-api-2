import os
import cv2
import numpy as np
import librosa
import onnxruntime as ort
import joblib
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Unified Emotion API 🎭",
    description="A production-ready API for Image, Audio, and Text Emotion Detection.",
    version="1.0.0"
)

# ==========================================
# 📂 PATH CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_MODEL_PATH = os.path.join(BASE_DIR, "image", "model.onnx")
IMAGE_ENCODER_PATH = os.path.join(BASE_DIR, "image", "encoder.pkl")

AUDIO_MODEL_PATH = os.path.join(BASE_DIR, "audio", "audio_model.onnx")
AUDIO_SCALER_PATH = os.path.join(BASE_DIR, "audio", "scaler.pkl")

TEXT_MODEL_PATH = os.path.join(BASE_DIR, "text", "text_model.pkl")
TEXT_VECTORIZER_PATH = os.path.join(BASE_DIR, "text", "vectorizer.pkl")

# ==========================================
# 🧠 MODEL LOADING
# ==========================================

# 1. Image Model
try:
    image_session = ort.InferenceSession(IMAGE_MODEL_PATH)
    image_input_name = image_session.get_inputs()[0].name
    image_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    logger.info("✅ Image model loaded successfully.")
except Exception as e:
    logger.error(f"❌ Failed to load image model: {e}")
    image_session = None

# 2. Audio Model
try:
    audio_session = ort.InferenceSession(AUDIO_MODEL_PATH)
    audio_input_name = audio_session.get_inputs()[0].name
    audio_output_name = audio_session.get_outputs()[0].name
    
    if os.path.exists(AUDIO_SCALER_PATH):
        audio_scaler = joblib.load(AUDIO_SCALER_PATH)
    else:
        audio_scaler = None
        
    # We use the encoder moved to 'image/' folder as requested by structure
    if os.path.exists(IMAGE_ENCODER_PATH):
        audio_encoder = joblib.load(IMAGE_ENCODER_PATH)
    else:
        audio_encoder = None
    logger.info("✅ Audio model and artifacts loaded successfully.")
except Exception as e:
    logger.error(f"❌ Failed to load audio model: {e}")
    audio_session = None

# 3. Text Model
try:
    text_model = joblib.load(TEXT_MODEL_PATH)
    text_vectorizer = joblib.load(TEXT_VECTORIZER_PATH)
    logger.info("✅ Text model and vectorizer loaded successfully.")
except Exception as e:
    logger.error(f"❌ Failed to load text model: {e}")
    text_model = None

# ==========================================
# 🛠️ PREPROCESSING HELPERS
# ==========================================

def preprocess_image(image_bytes):
    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image file")

    # Detect face (basic haar cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_full, 1.3, 5)

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
        face = img[y:y+h, x:x+w]
    else:
        face = img # Fallback to full image

    # Final preprocessing
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.resize(gray, (48, 48))
    
    # Stack to 3 channels if model expects it (typical for some converted models)
    # Check model input shape
    expected_shape = image_session.get_inputs()[0].shape
    if len(expected_shape) == 4 and expected_shape[-1] == 3:
        img_input = np.stack([gray, gray, gray], axis=-1)
    else:
        img_input = gray.reshape(48, 48, 1)

    img_input = img_input.astype("float32")
    # Normalize if needed (0-255 is standard for models with Rescaling layer)
    # img_input /= 255.0 # Uncomment if model doesn't have internal rescaling
    
    img_input = np.expand_dims(img_input, axis=0)
    return img_input

def preprocess_audio(file_path):
    y, sr = librosa.load(file_path, duration=3)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    features = mfcc.reshape(1, -1).astype(np.float32)
    
    if audio_scaler:
        features = audio_scaler.transform(features).astype(np.float32)
        
    return features

# ==========================================
# 🚀 API ENDPOINTS
# ==========================================

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    if not image_session:
        raise HTTPException(status_code=503, detail="Image model not loaded")
    
    try:
        contents = await file.read()
        img_input = preprocess_image(contents)
        
        preds = image_session.run(None, {image_input_name: img_input})[0]
        idx = int(np.argmax(preds))
        label = image_labels[idx]
        confidence = float(np.max(preds))
        
        return {"emotion": label, "confidence": confidence, "method": "image"}
    except Exception as e:
        logger.error(f"Image prediction error: {e}")
        return {"error": str(e)}

@app.post("/predict/audio")
async def predict_audio_endpoint(file: UploadFile = File(...)):
    if not audio_session:
        raise HTTPException(status_code=503, detail="Audio model not loaded")
    
    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        
        features = preprocess_audio(temp_path)
        
        preds = audio_session.run([audio_output_name], {audio_input_name: features})[0]
        idx = int(np.argmax(preds))
        
        if audio_encoder:
            label = audio_encoder.inverse_transform([idx])[0]
        else:
            label = str(idx) # Fallback to index
            
        confidence = float(np.max(preds))
        return {"emotion": label, "confidence": confidence, "method": "audio"}
    except Exception as e:
        logger.error(f"Audio prediction error: {e}")
        return {"error": str(e)}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/predict/text")
async def predict_text_endpoint(text: str = Form(...)):
    if not text_model or not text_vectorizer:
        raise HTTPException(status_code=503, detail="Text model not loaded")
    
    try:
        vec = text_vectorizer.transform([text])
        prediction = text_model.predict(vec)[0]
        return {"emotion": str(prediction), "method": "text"}
    except Exception as e:
        logger.error(f"Text prediction error: {e}")
        return {"error": str(e)}

@app.get("/")
def health_check():
    return {
        "status": "online",
        "models_loaded": {
            "image": image_session is not None,
            "audio": audio_session is not None,
            "text": text_model is not None
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)