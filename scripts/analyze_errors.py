"""
What this file is for:
A diagnostic, not a pipeline stage. Prints what the remaining errors actually
are - which words get substituted for which, what goes missing, what gets
invented - so effort goes where the errors are rather than where they are
assumed to be.

High-level role in the pipeline:
Nothing downstream reads this. Run it when a WER number needs explaining:

    python -u scripts/analyze_errors.py

It answers, concretely, whether the remaining loss is an orthography problem
that a normalization rule could fix for free, or an acoustic one that needs a
better ASR. On the 2026-08-27 run the answer was the second: 23,219
substitutions spread over 16,052 distinct pairs, the most common accounting for
0.5% of them.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load_config, predictions_path
from evaluate import align
from text import normalize_for_scoring

TOP = 25


def tally(pairs) -> tuple[Counter, Counter, Counter]:
    """Substituted pairs, deleted words and inserted words across every utterance."""
    substitutions: Counter = Counter()
    deletions: Counter = Counter()
    insertions: Counter = Counter()

    for reference_text, hypothesis_text in pairs:
        reference = normalize_for_scoring(reference_text).split()
        hypothesis = normalize_for_scoring(hypothesis_text).split()

        # align() walks backwards, so track both cursors to recover the words.
        i, j = len(reference), len(hypothesis)
        for operation, _ in align(reference, hypothesis):
            if operation in ("M", "S"):
                if operation == "S":
                    substitutions[(reference[i - 1], hypothesis[j - 1])] += 1
                i, j = i - 1, j - 1
            elif operation == "D":
                deletions[reference[i - 1]] += 1
                i -= 1
            else:
                insertions[hypothesis[j - 1]] += 1
                j -= 1

    return substitutions, deletions, insertions


def concentration(counter: Counter) -> str:
    """How much of the loss the worst offenders carry.

    A short head means a normalization rule can win it back. A flat tail means
    the errors are real and no amount of text munging will help.
    """
    total = sum(counter.values())
    if not total:
        return "nothing to report"

    head = sum(count for _, count in counter.most_common(TOP))
    return (
        f"{total:,} across {len(counter):,} distinct types; "
        f"the top {TOP} cover {head / total:.1%}"
    )


def main(config: dict | None = None) -> None:
    config = config or load_config()

    mode = config["inference_modes"][0]
    source = predictions_path(config, mode)
    if not source.is_file():
        raise SystemExit(f"Missing predictions: {source}")

    frame = pd.read_csv(source).fillna("")
    rows = frame.to_dict("records")
    print(f"Utterances: {len(rows):,} ({mode})\n")

    substitutions, deletions, insertions = tally(
        (row["truth"], row["prediction"]) for row in rows
    )

    print(f"SUBSTITUTIONS  {concentration(substitutions)}")
    print("-" * 60)
    for (reference, hypothesis), count in substitutions.most_common(TOP):
        print(f"  {count:>6}  {reference!r:>18} -> {hypothesis!r}")

    for name, counter in (("DELETIONS", deletions), ("INSERTIONS", insertions)):
        print(f"\n{name}  {concentration(counter)}")
        print("-" * 60)
        for word, count in counter.most_common(TOP):
            print(f"  {count:>6}  {word!r}")

    # A prediction whose last word ends in an apostrophe was cut mid-word, which
    # means the generation cap bound too tightly on that clip.
    truncated = [
        row for row in rows
        if str(row["prediction"]).split() and str(row["prediction"]).split()[-1].endswith("'")
    ]
    print(f"\nTRUNCATED MID-WORD  {len(truncated):,} of {len(rows):,} "
          f"({len(truncated) / len(rows):.2%}) - raise generation_min_tokens if this grows")
    for word, count in Counter(str(r["prediction"]).split()[-1] for r in truncated).most_common(5):
        print(f"  {count:>6}  {word!r}")


if __name__ == "__main__":
    main()
