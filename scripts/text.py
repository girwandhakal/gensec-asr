"""
What this file is for:
The text cleaning shared by the pipeline. Reference transcripts arrive as
CHILDES CHAT markup and Whisper output arrives with control tokens, so both
need reducing to plain words before they can be compared.
"""

from __future__ import annotations

import re
import unicodedata

WHITESPACE_RE = re.compile(r"\s+")
WHISPER_CONTROL_TOKEN_RE = re.compile(r"<\|[^|]+?\|>")

# CHAT annotation, removed in this order.
BRACKET_CODE_RE = re.compile(r"\[[^\[\]]*\]")     # [/] retrace, [?] unclear, [% comment]
EVENT_CODE_RE = re.compile(r"[&+]\S+")            # &=laughs, &-uh, +... terminators
SPECIAL_FORM_RE = re.compile(r"@\S+")             # Mummy@f family form marker
UNINTELLIGIBLE_RE = re.compile(r"\b(?:xxx|yyy|www)\b")
UNSPOKEN_RE = re.compile(r"\b0\S*")               # 0det marks something not said
PUNCTUATION_RE = re.compile(r"[^\w\s']")


def collapse_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def normalize_chat(text: str, remove_unintelligible: bool = True) -> str:
    """Reduce one CHAT-formatted utterance to plain lowercase words."""
    text = unicodedata.normalize("NFKC", str(text or "")).replace("’", "'")

    # Bracketed codes can nest, so keep stripping until nothing changes.
    while True:
        stripped = BRACKET_CODE_RE.sub(" ", text)
        if stripped == text:
            break
        text = stripped

    text = EVENT_CODE_RE.sub(" ", text)
    text = SPECIAL_FORM_RE.sub(" ", text)
    text = UNSPOKEN_RE.sub(" ", text)

    # <these> mark the scope of a following code; the words inside are real.
    # (be)cause marks a clipped pronunciation; keep the full word.
    text = text.replace("<", " ").replace(">", " ")
    text = text.replace("(", "").replace(")", "")
    text = text.replace("_", " ")

    text = text.lower()
    if remove_unintelligible:
        text = UNINTELLIGIBLE_RE.sub(" ", text)

    text = PUNCTUATION_RE.sub(" ", text)
    return collapse_whitespace(text)


def clean_whisper_text(text: str) -> str:
    """Strip Whisper's control tokens and tidy the spacing."""
    return collapse_whitespace(WHISPER_CONTROL_TOKEN_RE.sub(" ", str(text or "")))


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


def normalize_for_scoring(text: str, canonicalize_fillers: bool = True) -> str:
    """Lowercase, drop punctuation - applied to both sides before WER."""
    text = unicodedata.normalize("NFKC", str(text or "")).replace("’", "'")
    text = WHISPER_CONTROL_TOKEN_RE.sub(" ", text).lower()
    text = collapse_whitespace(PUNCTUATION_RE.sub(" ", text))
    return canonicalize_hums(text) if canonicalize_fillers else text
