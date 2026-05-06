"""
Text Emotion Service
Robust negation + context-aware rules + ML fallback
"""

import re
import logging
import joblib

logger = logging.getLogger(__name__)


class TextModelService:
    def __init__(self):
        self.vectorizer = None
        self.model = None

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

        # ======================
        # 1) RULES FIRST (أهم)
        # ======================
        rule_label = self._advanced_rules(text)
        if rule_label:
            logger.info("🧠 Rule-based result: %s", rule_label)
            return rule_label

        # ======================
        # 2) ML fallback
        # ======================
        if self.vectorizer and self.model:
            try:
                vec = self.vectorizer.transform([text])
                pred = self.model.predict(vec)
                label = str(pred[0]).lower()
                logger.info("🧠 ML result: %s", label)
                return label
            except Exception as e:
                logger.warning("ML failed: %s", e)

        return "neutral"

    # =========================================
    # 🔥 ADVANCED RULE ENGINE
    # =========================================
    def _advanced_rules(self, text: str) -> str:
        t = text.lower()

        # =====================
        # 1) EMPTY / NUMB
        # =====================
        if any(w in t for w in ["empty", "numb", "lost", "hopeless", "nothing"]):
            return "sad"

        # =====================
        # 2) NEGATION (STRONG)
        # =====================
        neg_patterns = [
            (r"(not|don't|dont|never|no)\s+(feel\s+)?sad", "happy"),
            (r"(not|don't|dont|never|no)\s+(feel\s+)?happy", "sad"),
            (r"(not|don't|dont|never|no)\s+(feel\s+)?angry", "neutral"),
            (r"(not|don't|dont|never|no)\s+(feel\s+)?scared", "neutral"),
        ]

        for pattern, result in neg_patterns:
            if re.search(pattern, t):
                return result

        # =====================
        # 3) CONTRAST (but / actually)
        # =====================
        if "but" in t:
            parts = t.split("but")
            return self._advanced_rules(parts[-1]) or "neutral"

        if "actually" in t:
            parts = t.split("actually")
            return self._advanced_rules(parts[-1]) or "neutral"

        # =====================
        # 4) DIRECT EMOTION
        # =====================
        if any(w in t for w in ["happy", "excited", "great", "good", "love"]):
            return "happy"
        if any(w in t for w in ["sad", "depressed", "upset", "tired"]):
            return "sad"
        if any(w in t for w in ["angry", "mad", "furious"]):
            return "angry"
        if any(w in t for w in ["fear", "scared", "afraid", "anxious"]):
            return "fear"

        # =====================
        # 5) SPECIAL CASES
        # =====================
        if "i don't feel anything" in t:
            return "sad"

        if "not sad" in t:
            return "happy"

        if "not happy" in t:
            return "sad"

        return None


# singleton
text_model_service = TextModelService()