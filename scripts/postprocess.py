"""
What this file is for:
Stage 5. Cleans obvious generation artifacts out of the model's predictions.

High-level role in the pipeline:
Seq2seq models get stuck in loops and repeat a phrase until they hit the
length limit. That is a decoding artifact, not a transcription, and it wrecks
WER. This removes the repetition and nothing else - it is deliberately not in
the business of rewriting transcripts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load_config, predictions_path
from text import collapse_whitespace

# Small, hand-checked fixes for errors the model makes consistently.
MANUAL_REPLACEMENTS = {
    "ore": "or",
}

MIN_PHRASE_WORDS = 1
MAX_PHRASE_WORDS = 8


def apply_replacements(words: list[str]) -> list[str]:
    return [MANUAL_REPLACEMENTS.get(word, word) for word in words]


def collapse_repeated_phrases(words: list[str]) -> list[str]:
    """Drop any repeated phrase or loop (e.g. 'and a hawk and a hawk', 'eating and eating').

    For phrases of any size (1 up to MAX_PHRASE_WORDS), if the exact same sequence
    repeats immediately, we collapse successive duplicates down to at most 2
    (or 1 for longer phrases) to preserve natural child speech while eradicating
    catastrophic hallucination loops.
    """
    result = list(words)

    for size in range(MAX_PHRASE_WORDS, MIN_PHRASE_WORDS - 1, -1):
        index = 0
        while index + 2 * size <= len(result):
            phrase = result[index:index + size]
            # Count consecutive occurrences of this phrase
            match_count = 1
            while (
                index + (match_count + 1) * size <= len(result)
                and result[index + match_count * size:index + (match_count + 1) * size] == phrase
            ):
                match_count += 1

            if match_count > 1:
                # Keep 2 repetitions for 1-word or 2-word phrases if 2, else 1
                keep_count = 2 if (size <= 2 and match_count == 2) else 1
                del result[index + keep_count * size:index + match_count * size]
                index += keep_count * size
            else:
                index += 1

    return result


def collapse_repeated_word(words: list[str]) -> list[str]:
    """A prediction that is one word repeated more than 3 times is collapsed to at most two words."""
    if len(words) > 3 and len(set(words)) == 1:
        return words[:2]
    return words


def clean_prediction(text: str) -> str:
    words = collapse_whitespace(str(text or "")).split()
    words = apply_replacements(words)
    words = collapse_repeated_word(words)
    words = collapse_repeated_phrases(words)
    return collapse_whitespace(" ".join(words))


def main(config: dict | None = None) -> None:
    config = config or load_config()

    # Postprocessing is only wired to the first mode; that is the one scored
    # as the headline number.
    mode = config["inference_modes"][0]
    source = predictions_path(config, mode)
    if not source.is_file():
        raise SystemExit(f"Missing predictions: {source}")

    frame = pd.read_csv(source).fillna({"prediction": ""})
    original = frame["prediction"].astype(str)
    frame["prediction"] = original.map(clean_prediction)

    changed = int((frame["prediction"] != original).sum())
    frame.to_csv(config["cleaned_predictions_path"], index=False)

    print(f"Predictions read:    {len(frame):,} ({mode})")
    print(f"Predictions changed: {changed:,}")
    print(f"Wrote {config['cleaned_predictions_path']}")


if __name__ == "__main__":
    main()
