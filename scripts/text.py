"""
What this file is for:
Text normalization for Whisper and correction-model outputs only.
Ground-truth references are finalized upstream and are never cleaned here.
"""

from __future__ import annotations

import re
import unicodedata

WHITESPACE_RE = re.compile(r"\s+")
WHISPER_CONTROL_TOKEN_RE = re.compile(r"<\|[^|]+?\|>")

# Reference cleaning belongs to asr-dataset-pipelines. These functions apply
# only to ASR/model outputs; finalized ground truth must be read unchanged.
PUNCTUATION_RE = re.compile(r"[^\w\s']")


MOJIBAKE_MAP = {
    "â€™": "'",
    "â€˜": "'",
    "â€œ": '"',
    "â€\x9d": '"',
    "â€": '"',
    "â€“": "-",
    "â€”": "-",
    "â tms": "'s",
    "â tm": "'",
}


def fix_mojibake(text: str) -> str:
    for broken, fixed in MOJIBAKE_MAP.items():
        text = text.replace(broken, fixed)
    return text


def collapse_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def clean_whisper_text(text: str) -> str:
    """Strip Whisper's control tokens, fix mojibake and tidy the spacing."""
    text = fix_mojibake(unicodedata.normalize("NFKC", str(text or ""))).replace("’", "'")
    return collapse_whitespace(WHISPER_CONTROL_TOKEN_RE.sub(" ", text))


# One nasal hum, spelled many ways. CHAT transcribers and Whisper do not agree
# on which spelling to use - the test set has `mhm` 74 times in the references
# and 1,216 times in the hypotheses - and no listener can reliably tell them
# apart anyway. Collapsing them scores the vocalization rather than the
# spelling convention, the same way Whisper's own English normalizer does.
#
# Deliberately excluded: `uhhuh` and `uhuh`, which mean yes and no. Merging
# those would erase a real distinction to flatter the metric.
HUM_FORMS = {"mm", "mmm", "mhm", "mmhm", "mmhmm", "hm", "hmm", "hmhm", "mhmm"}
HUM_CANONICAL = "mm"


def canonicalize_hums(text: str) -> str:
    return " ".join(HUM_CANONICAL if word in HUM_FORMS else word for word in text.split())


def normalize_prediction_for_scoring(text: str, canonicalize_fillers: bool = True) -> str:
    """Normalize a predicted transcript to the upstream reference conventions."""
    text = fix_mojibake(unicodedata.normalize("NFKC", str(text or ""))).replace("’", "'")
    text = WHISPER_CONTROL_TOKEN_RE.sub(" ", text).lower()
    text = collapse_whitespace(PUNCTUATION_RE.sub(" ", text))
    return canonicalize_hums(text) if canonicalize_fillers else text
