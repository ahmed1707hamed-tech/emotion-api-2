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
# New Requirement: audio > image > text
PRIORITY = ["audio", "image", "text"]

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
    if audio_emotion:
        votes["audio"] = extract_label(audio_emotion)
    if image_emotion:
        votes["image"] = extract_label(image_emotion)
    if text_emotion:
        votes["text"] = extract_label(text_emotion)

    if not votes:
        return "neutral"

    # If only one modality, return it directly
    if len(votes) == 1:
        source = list(votes.keys())[0]
        emotion = votes[source]
        logger.info("🎯 Selected modality: %s | Emotion: %s", source, emotion)
        return emotion

    # Count votes
    emotion_counts = Counter(votes.values())
    max_count = max(emotion_counts.values())
    winners = [e for e, c in emotion_counts.items() if c == max_count]

    # Clear majority
    if len(winners) == 1:
        # Find which modality provided this majority
        source = [s for s, e in votes.items() if e == winners[0]][0]
        logger.info("🎯 Selected modality: %s (Majority) | Emotion: %s", source, winners[0])
        return winners[0]

    # Tie → use priority order (audio > image > text)
    for source in PRIORITY:
        if source in votes and votes[source] in winners:
            logger.info(
                "🎯 Selected modality: %s (Priority Tie-break) | Emotion: %s",
                source, votes[source]
            )
            return votes[source]

    return winners[0]
