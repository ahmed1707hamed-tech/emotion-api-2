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
                "Respond warmly and cheerfully."
            ),

            "sad": (
                "Respond with empathy."
            ),

            "angry": (
                "Respond calmly and softly."
            ),

            "fear": (
                "Respond reassuringly."
            ),

            "surprise": (
                "Respond with excitement."
            ),

            "neutral": (
                "Respond naturally."
            ),

            "disgust": (
                "Respond carefully and sympathetically."
            )
        }

        system_prompt = emotion_prompts.get(
            emotion,
            "Respond naturally."
        )

        if not user_text:
            user_text = (
                f"My emotion is {emotion}"
            )

        completion = client.chat.completions.create(

            model="llama3-8b-8192",

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

            temperature=0.8,
            max_tokens=80
        )

        return (
            completion
            .choices[0]
            .message.content
        )

    except Exception as e:

        print("❌ Groq error:", e)

        fallback = {
            "happy": "You seem happy today 😄",
            "sad": "I hope things get better ❤️",
            "angry": "Take it easy, everything will be okay 🙏",
            "fear": "Don't worry, you're safe 💙",
            "surprise": "Wow, that's surprising 😮",
            "neutral": "Hope you're doing well 🙂",
            "disgust": "That doesn't seem pleasant 😕"
        }

        return fallback.get(
            emotion,
            "Hello 🙂"
        )