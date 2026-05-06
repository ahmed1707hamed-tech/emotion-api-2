"""
Groq LLM Service
Uses the Groq API to generate dynamic, empathetic responses
based on the detected emotion and user text.
"""

import os
import logging
import requests

logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_PROMPT = """You are an empathetic AI assistant.

User emotion: {emotion}
User message: {text}

Respond naturally, emotionally, and differently each time.
Avoid generic responses."""


def generate_response(emotion: str, user_text: str = "") -> str:
    """
    Call the Groq API to generate a dynamic empathetic LLM response.
    """
    api_key = GROQ_API_KEY or os.getenv("GROQ_API_KEY")

    if not api_key:
        logger.error("GROQ_API_KEY not set")
        return "API key is not configured. Please set GROQ_API_KEY."

    # Use the exact prompt required by the user
    prompt = SYSTEM_PROMPT.format(
        emotion=emotion, 
        text=user_text.strip() if user_text else "[No text provided, only voice/image emotion]"
    )

    logger.info(f"--- GROQ PROMPT ---\n{prompt}\n-------------------")

    try:
        response = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.9,
                "max_tokens": 300,
            },
            timeout=15,
        )

        if response.status_code != 200:
            logger.error("Groq API error %d: %s", response.status_code, response.text)
            return f"LLM service error (status {response.status_code}). Please try again."

        data = response.json()

        if "choices" not in data or not data["choices"]:
            logger.error("Invalid Groq response format: %s", data)
            return "Unexpected response from LLM. Please try again."

        content = data["choices"][0]["message"]["content"]
        logger.info(f"--- GROQ RESPONSE ---\n{content}\n---------------------")
        return content

    except requests.exceptions.Timeout:
        logger.error("Groq API request timed out")
        return "LLM request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        logger.error("Groq API request failed: %s", e)
        return f"LLM request failed: {e}"
    except Exception as e:
        logger.error("Unexpected error in Groq service: %s", e)
        return f"Unexpected error: {e}"