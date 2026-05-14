"""
Text Emotion Service
Robust negation + context-aware rules + ML fallback
"""

import re
import logging
import joblib
import numpy as np
from app.services.groq_service import client

logger = logging.getLogger(__name__)


class TextModelService:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.allowed_emotions = [
            "happy", "sad", "angry", "fear", "surprise", "love", "neutral"
        ]

    def load_model(self, vectorizer_path: str, model_path: str):
        try:
            self.vectorizer = joblib.load(vectorizer_path)
            self.model = joblib.load(model_path)
            logger.info("✅ Text model loaded")
        except Exception as e:
            logger.error("❌ Failed to load text model: %s", e)
            self.vectorizer = None
            self.model = None

    def detect_emotion(self, text: str) -> str:
        text = text.strip()
        if not text or text.lower() == "string":
            return "neutral"

        # 1) Try Rules First (for very obvious cases or negations)
        rule_label = self._advanced_rules(text)
        if rule_label:
            logger.info("🧠 Rule-based result: %s", rule_label)
            return self._normalize_label(rule_label)

        # 2) ML Prediction with Confidence
        if self.vectorizer and self.model:
            try:
                vec = self.vectorizer.transform([text])
                if hasattr(self.model, "predict_proba"):
                    probas = self.model.predict_proba(vec)[0]
                    idx = np.argmax(probas)
                    confidence = probas[idx]
                    label = str(self.model.classes_[idx]).lower()
                else:
                    label = str(self.model.predict(vec)[0]).lower()
                    confidence = 1.0
                
                logger.info("RAW_MODEL: %s confidence=%.2f", label, confidence)
                
                normalized_label = self._normalize_label(label)
                logger.info("NORMALIZED: %s", normalized_label)

                # 3) Semantic Refinement (Groq)
                tr_text = text.lower()
                is_surprise_trigger = any(w in tr_text for w in ["wow", "believe", "shock", "unexpected", "surprise", "unbelievable"])
                is_fear_trigger = any(w in tr_text for w in ["anxious", "nervous", "worry", "panic", "scared"])
                
                should_refine = (
                    confidence < 0.7 or 
                    normalized_label not in self.allowed_emotions or
                    (normalized_label == "love" and is_surprise_trigger) or
                    (normalized_label == "sad" and is_surprise_trigger) or
                    (normalized_label == "neutral" and (is_surprise_trigger or is_fear_trigger))
                )

                if should_refine:
                    refined_label = self._semantic_refinement(text, normalized_label)
                    logger.info("GROQ_SEMANTIC: %s", refined_label)
                    if refined_label:
                        return refined_label
                
                return normalized_label
            except Exception as e:
                logger.warning("ML prediction failed: %s", e)

        return "neutral"

    def _normalize_label(self, label: str) -> str:
        """Standardize labels and map synonyms."""
        mapping = {
            "joy": "happy",
            "happiness": "happy",
            "sadness": "sad",
            "anger": "angry",
            "fearful": "fear",
            "surprised": "surprise",
            "shocked": "surprise",
            "affection": "love",
            "romantic": "love",
            "excited": "happy", # User wants excitement to be distinguished, but allowed list maps to happy/surprise? 
                               # Actually user said "distinguish: surprise, love, excitement, happiness correctly".
                               # But their allowed list is: happy, sad, angry, fear, surprise, love, neutral.
                               # I'll map excitement to happy for now or surprise depending on context.
        }
        
        normalized = mapping.get(label.lower(), label.lower())
        
        # NEVER map surprise to love
        if "surprise" in label.lower() and normalized == "love":
            return "surprise"
            
        return normalized

    def _semantic_refinement(self, text: str, ml_label: str) -> str:
        """Use Groq to classify the emotional state accurately and prevent shock/surprise from being labeled as sad."""
        try:
            prompt = (
                "Classify the emotional state expressed in this text.\n"
                "Return ONLY ONE of these labels: [happy, sad, angry, fear, surprise, love, neutral].\n\n"
                "STRICT SEMANTIC RULES:\n"
                "- SURPRISE: shock, disbelief, unexpected events, amazement, stunned, 'no way', 'can't believe'. (High Priority)\n"
                "- SAD: emotional pain, emptiness, loneliness, depression, hopelessness, grief, crying.\n"
                "- FEAR: anxiety, worry, nervousness, panic.\n"
                "- LOVE: affection, romance, attachment.\n"
                "- ANGRY: frustration, rage, irritation.\n"
                "- HAPPY: joy, excitement, gratitude.\n"
                "- NEUTRAL: no clear emotional signal.\n\n"
                "CRITICAL: Shock and disbelief (surprise) MUST NEVER be classified as sad.\n"
                "If text expresses 'I can't believe it' or 'Wow', it is SURPRISE.\n"
                "Return ONLY the word itself.\n\n"
                f"Text: \"{text}\"\n"
                "Result:"
            )
            
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10
            )
            
            raw_response = completion.choices[0].message.content.strip().lower()
            # Clean response to just alpha characters
            clean_word = re.sub(r'[^a-z]', '', raw_response)
            
            if clean_word in self.allowed_emotions:
                return clean_word
            
            # Fallback extraction if verbose
            # Prioritize surprise over sad to fix the collapse
            if "surprise" in raw_response:
                return "surprise"
            if "sad" in raw_response:
                return "sad"
                
            for emotion in self.allowed_emotions:
                if emotion in raw_response:
                    return emotion
            
            return "neutral"

        except Exception as e:
            logger.error("Groq refinement failed: %s", e)
            return "neutral"

    def _advanced_rules(self, text: str) -> str:
        t = text.lower()

        # Negation handling (Strong priority)
        neg_patterns = [
            (r"(not|don't|dont|never|no)\s+(feel\s+)?sad", "happy"),
            (r"(not|don't|dont|never|no)\s+(feel\s+)?happy", "sad"),
        ]

        for pattern, result in neg_patterns:
            if re.search(pattern, t):
                return result

        # Basic synonym fallback for extremely short inputs that confuse ML
        # (This is safe normalization as requested)
        quick_map = {
            "joy": "happy",
            "joyful": "happy",
            "happiness": "happy",
            "sadness": "sad",
            "anger": "angry",
            "scared": "fear",
            "fearful": "fear",
            "surprised": "surprise",
            "shocked": "surprise",
            "amazing": "surprise",
        }
        
        if t in quick_map:
            return quick_map[t]

        # Very strong indicators
        if any(w in t for w in ["i love you", "i love her", "i love him", "بحبك", "بعشقك"]):
            return "love"
            
        return None


# singleton
text_model_service = TextModelService()