---
title: emotion-api-2
emoji: 🎭
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---
# 🎭 Multimodal Emotion Detection API

A production-ready FastAPI application that provides emotion analysis across three modalities: Image, Audio, and Text. This project is structured for easy deployment to Hugging Face Spaces or any cloud provider.

## 📂 Project Structure

```
emotion-api/
│
├── image/               # Image emotion model (ONNX)
│   ├── model.onnx       # Primary ONNX model
│   ├── model_quant.onnx # Quantized version (optional)
│   └── encoder.pkl      # Label encoder
│
├── audio/               # Audio emotion model (ONNX)
│   ├── audio_model.onnx # ONNX inference model
│   └── scaler.pkl       # Feature scaler
│
├── text/                # Text emotion model (Scikit-learn)
│   ├── text_model.pkl   # Classifier model
│   ├── vectorizer.pkl   # TF-IDF or Count Vectorizer
│   ├── tokenizer.json   # Tokenizer config (if using transformers)
│   └── vocab.txt        # Vocabulary file
│
├── app.py               # Unified FastAPI application
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## 🚀 How to Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the API**:
   ```bash
   python app.py
   ```
   Or using uvicorn:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

## 🛠️ API Endpoints

### 1. Image Prediction
- **Endpoint**: `/predict/image`
- **Method**: `POST`
- **Payload**: `file` (Upload an image file)
- **Output**: Returns predicted emotion and confidence.

### 2. Audio Prediction
- **Endpoint**: `/predict/audio`
- **Method**: `POST`
- **Payload**: `file` (Upload an audio file, e.g., .wav)
- **Output**: Returns predicted emotion based on MFCC features.

### 3. Text Prediction
- **Endpoint**: `/predict/text`
- **Method**: `POST`
- **Payload**: `text` (Form data)
- **Output**: Returns predicted emotion using the ML fallback or rules.

## 🚢 Deploying to Hugging Face

This repository is ready to be pushed to a Hugging Face Space (Docker or Static). 

1. Create a new Space on Hugging Face.
2. Select **Docker** or **Python** (SDK).
3. Push these files to the repository.
4. Hugging Face will automatically build and serve the API.

---
*Created with ❤️ for Multimodal AI.*
