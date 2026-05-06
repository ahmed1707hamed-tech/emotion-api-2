"""
Fusion Service
Combines emotions from multiple modalities using majority voting.
Tie-breaking priority: image > audio > text.
"""

import logging
from collections import Counter
from typing import Optional, Union, Tuple

logger = logging.getLogger(__name__)

# Priority order for tie-breaking (lower index = higher priority)
PRIORITY = ["image", "audio", "text"]

def fuse_emotions(
    image_emotion: Optional[Union[str, Tuple[str, float]]] = None,
    audio_emotion: Optional[Union[str, Tuple[str, float]]] = None,
    text_emotion: Optional[Union[str, Tuple[str, float]]] = None,
) -> str:
    """
    Combine emotions from available modalities using majority voting.
    Handles both raw emotion strings and (label, confidence) tuples.
    """
    
    def extract_label(val):
        if isinstance(val, tuple):
            return val[0]
        return val

    votes = {}
    if image_emotion:
        votes["image"] = extract_label(image_emotion)
    if audio_emotion:
        votes["audio"] = extract_label(audio_emotion)
    if text_emotion:
        votes["text"] = extract_label(text_emotion)

    if not votes:
        return "neutral"

    # If only one modality, return it directly
    if len(votes) == 1:
        emotion = list(votes.values())[0]
        logger.info("🎯 Single modality → %s", emotion)
        return emotion

    # Count votes
    emotion_counts = Counter(votes.values())
    max_count = max(emotion_counts.values())
    winners = [e for e, c in emotion_counts.items() if c == max_count]

    # Clear majority
    if len(winners) == 1:
        logger.info("🎯 Majority vote → %s (votes: %s)", winners[0], dict(emotion_counts))
        return winners[0]

    # Tie → use priority order (image > audio > text)
    for source in PRIORITY:
        if source in votes and votes[source] in winners:
            logger.info(
                "🎯 Tie broken by %s priority → %s (votes: %s)",
                source, votes[source], dict(emotion_counts),
            )
            return votes[source]

    return winners[0]
