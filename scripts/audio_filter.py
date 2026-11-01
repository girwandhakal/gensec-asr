"""Shared screening for suspicious clip/reference pairs."""

from config import clip_seconds


def audio_filter(utterance_id: str, reference: str, config: dict) -> str:
    """Return an exclusion reason, or an empty string when the pair passes."""
    seconds = clip_seconds(utterance_id)
    if seconds is None:
        return "unparseable_clip_interval"
    if seconds < config.get("min_clip_seconds", 0.5):
        return "point_bullet_timestamp" if seconds <= 0.1 + 1e-9 else "clip_too_short"
    words = len(reference.split())
    if words / seconds > config.get("max_reference_words_per_second", 4.0):
        return "impossible_speech_rate"
    if seconds > config.get("long_single_word_seconds", 5.0) and words <= 1:
        return "long_clip_short_reference"
    return ""
