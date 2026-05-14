import logging
import os
from groq import Groq

logger = logging.getLogger(__name__)

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def generate_response(
    emotion: str,
    user_text: str = "",
    context_emotion: str = None
):
    """
    Generate an empathetic response based on detected emotion.
    """
    logger.info("LLM_EMOTION_CONTEXT=%s", emotion)
    
    try:
        # Guidance for specific emotions
        guidance = {
            "sad": {
                "tone": "deeply compassionate, comforting, and supportive. Use a soothing and gentle voice.",
                "recs": "relaxing audio, specialized breathing exercises, or Mood Booster videos in the V8 app"
            },
            "angry": {
                "tone": "calming, grounding, and de-escalating. Stay extremely patient and peaceful.",
                "recs": "guided meditation, calming sounds, or deep breathing techniques in the V8 app"
            },
            "fear": {
                "tone": "reassuring, protective, and calming. Focus on safety and grounding.",
                "recs": "slow breathing patterns, mindfulness content, or relaxing nature videos in the V8 app"
            },
            "happy": {
                "tone": "positive, high-energy, and enthusiastic! Celebrate with the user.",
                "recs": "keeping this amazing positive energy flowing"
            },
            "surprise": {
                "tone": "curious, supportive, and engaged. Respond with genuine interest.",
                "recs": "grounding exercises to process the unexpected"
            },
            "love": {
                "tone": "warm, affectionate, and positive. Focus on connection.",
                "recs": "positive journaling or connection activities"
            }
        }

        info = guidance.get(emotion, {"tone": "casual and natural", "recs": None})
        
        system_prompt = (
            f"You are a helpful and empathetic AI assistant. "
            f"The user sounds {emotion}. "
            f"Your response tone MUST be {info['tone']}. "
        )

        if info['recs']:
            if emotion in ["sad", "angry", "fear"]:
                system_prompt += f"You MUST suggest that they try {info['recs']}. Mention that these are available in the V8 app. "
            else:
                system_prompt += f"Encourage them by mentioning {info['recs']}. "

        system_prompt += (
            "Reply in 1-2 short, natural, human-like sentences. "
            "Do not sound like a robot. Use emojis where appropriate."
        )

        # Handle follow-up memory context if different from current
        if context_emotion and context_emotion != "neutral" and context_emotion != emotion:
            system_prompt += f" Also acknowledge that they previously felt {context_emotion}."

        if not user_text:
            user_text = f"I am feeling {emotion}."

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ],
            temperature=0.7,
            max_tokens=100
        )

        response = completion.choices[0].message.content.strip()
        return response

    except Exception as e:
        logger.error("❌ Groq error: %s", e)
        fallback = {
            "happy": "You sound happy today 😄 Keep that positive energy going!",
            "sad": "I'm sorry you're feeling this way 😔 Try some breathing exercises or relaxing audio in V8.",
            "angry": "It sounds like you're frustrated 😠. Take a deep breath and maybe try some calming sounds in V8.",
            "fear": "You're not alone 😟. Try slow breathing and some calming content in V8.",
            "surprise": "What happened? You sound quite surprised 😲",
            "neutral": "I'm here for you. How can I help today? 🙂",
            "love": "It's wonderful to feel that warmth 💖"
        }
        return fallback.get(emotion, "I'm here for you 🙂")