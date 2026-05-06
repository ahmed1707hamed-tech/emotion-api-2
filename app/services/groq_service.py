import os
from groq import Groq

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def generate_response(
    emotion: str,
    user_text: str = ""
):

    emotion_prompts = {

        "happy": (
            "The user looks happy and positive. "
            "Respond warmly, cheerfully, and naturally."
        ),

        "sad": (
            "The user seems sad. "
            "Respond with empathy and emotional support."
        ),

        "angry": (
            "The user seems angry or frustrated. "
            "Respond calmly and try to de-escalate."
        ),

        "fear": (
            "The user seems anxious or fearful. "
            "Respond reassuringly and gently."
        ),

        "surprise": (
            "The user looks surprised or shocked. "
            "Respond with curiosity and engagement."
        ),

        "neutral": (
            "The user appears neutral. "
            "Respond naturally and conversationally."
        ),

        "disgust": (
            "The user seems uncomfortable or disgusted. "
            "Respond carefully and sympathetically."
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

        model="llama3-70b-8192",

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
        max_tokens=120
    )

    return (
        completion
        .choices[0]
        .message.content
    )