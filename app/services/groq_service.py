import os
from groq import Groq

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def generate_response(
    emotion: str,
    user_text: str = "",
    context_emotion: str = None
):
    """
    Generate an empathetic response.
    If context_emotion is provided, it means this is a follow-up about emotional help.
    """
    try:
        emotion_prompts = {
            "happy": (
                "The user looks happy. "
                "Reply in a warm, friendly, natural way "
                "using 1-2 short sentences."
            ),
            "sad": (
                "The user seems sad. "
                "Reply with empathy and emotional support "
                "using 1-2 natural sentences."
            ),
            "angry": (
                "The user seems angry. "
                "Reply calmly and help ease the tension "
                "using short natural sentences."
            ),
            "fear": (
                "The user seems anxious or fearful. "
                "Reply reassuringly and gently "
                "using 1-2 sentences."
            ),
            "surprise": (
                "The user looks surprised. "
                "Reply with curiosity and engagement "
                "using short natural sentences."
            ),
            "neutral": (
                "The user appears neutral. "
                "Reply casually and naturally."
            ),
            "love": (
                "The user feels love or affection. "
                "Reply warmly and supportively."
            )
        }

        # Recommendations based on emotion (Dynamic context for Groq)
        recommendations = {
            "sad": "relaxing audio, Mood Booster videos in V8, and breathing exercises",
            "angry": "meditation, calming music, and breathing techniques",
            "fear": "mindfulness practices, breathing exercises, and relaxing videos",
            "happy": "motivation videos and gratitude activities",
            "love": "journaling and positive content",
            "surprise": "reflective prompts and grounding exercises"
        }

        system_prompt = emotion_prompts.get(
            emotion,
            emotion_prompts["neutral"]
        )

        # If it's a follow-up, override system prompt to be more guidance-oriented
        if context_emotion and context_emotion != "neutral":
            rec_text = recommendations.get(context_emotion, "simple self-care activities")
            system_prompt = (
                f"The user is seeking help regarding their earlier feeling of {context_emotion}. "
                "Provide a deeply empathetic, supportive response. "
                f"Suggest these specific activities: {rec_text}. "
                "Mention that they can find more in the V8 app. "
                "Keep it natural, human, and concise (2-3 sentences)."
            )

        if not user_text:
            user_text = f"My detected emotion is {emotion}"

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ],
            temperature=0.8,
            max_tokens=100
        )

        response = completion.choices[0].message.content.strip()
        return response

    except Exception as e:
        logger.error("❌ Groq error: %s", e)
        fallback = {
            "happy": "You seem really happy today 😄",
            "sad": "I hope things get better soon 💙. Try some breathing exercises.",
            "angry": "Take it easy, everything will be okay 🙏. Maybe some calming music?",
            "fear": "Don't worry, you're safe 💙. Try a grounding exercise.",
            "surprise": "Wow, that looks surprising 😮",
            "neutral": "Hope you're doing well 🙂",
            "love": "It's wonderful to feel loved 💖"
        }
        return fallback.get(emotion, "Hello 🙂")