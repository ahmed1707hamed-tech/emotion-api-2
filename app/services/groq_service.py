import os
from groq import Groq

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def generate_response(
    emotion: str,
    user_text: str = ""
):

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

            "disgust": (
                "The user seems uncomfortable. "
                "Reply sympathetically and naturally."
            )
        }

        system_prompt = emotion_prompts.get(
            emotion,
            emotion_prompts["neutral"]
        )

        if not user_text:
            user_text = (
                f"My detected emotion is {emotion}"
            )

        completion = client.chat.completions.create(

            model="llama-3.1-8b-instant",

            messages=[

                {
                    "role": "system",
                    "content": system_prompt
                },

                {
                    "role": "user",
                    "content": user_text
                }
            ],

            temperature=0.9,
            max_tokens=60
        )

        response = (
            completion
            .choices[0]
            .message.content
            .strip()
        )

        return response

    except Exception as e:

        print("❌ Groq error:", e)

        fallback = {
            "happy": "You seem really happy today 😄",
            "sad": "I hope things get better soon 💙",
            "angry": "Take it easy, everything will be okay 🙏",
            "fear": "Don't worry, you're safe 💙",
            "surprise": "Wow, that looks surprising 😮",
            "neutral": "Hope you're doing well 🙂",
            "disgust": "That doesn't seem very pleasant 😕"
        }

        return fallback.get(
            emotion,
            "Hello 🙂"
        )