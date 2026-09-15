"""
What this file is for:
Reads an n-best corpus and reports how much candidate disagreement it actually
contains. Not part of the pipeline - this is the go/no-go check before
committing days of GPU to regenerating stage 2.

High-level role in the pipeline:
GenSEC can only arbitrate between candidates that differ. A corpus where most
clips return one string gives the corrector nothing to do on those clips, so
the distinct-hypothesis rate is the number that decides whether a decoding
change is worth the regeneration cost.

Compare two corpora directly:
    python scripts/check_diversity.py \
        --nbest data_probe/utterance_id_to_nbest.json \
        --baseline data/utterance_id_to_nbest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load_config

# CHSER (Interspeech 2025) reached 5 distinct hypotheses on 91.8% of their
# training utterances using beam-search multinomial sampling with 50 draws
# filtered to 5. Their speech is MyST classroom recordings - longer and more
# varied than CHILDES child utterances - so this is a reference point, not a
# target the same decoding should be expected to hit here.
CHSER_TRAIN_DISTRIBUTION = {1: 0.006, 2: 0.017, 3: 0.027, 4: 0.033, 5: 0.918}


def distinct_counts(nbest: dict) -> Counter:
    """How many distinct hypotheses each clip ended up with."""
    counts = Counter()
    for entry in nbest.values():
        texts = {
            (item.get("text") or "").strip()
            for item in entry.get("nbest", [])
            if (item.get("text") or "").strip()
        }
        counts[len(texts)] += 1
    return counts


def describe(counts: Counter, total: int, label: str) -> list[str]:
    lines = [f"{label}  ({total:,} clips)", "-" * 56]
    lines.append(f"{'distinct':>10}{'clips':>12}{'share':>10}")
    for size in sorted(counts):
        share = counts[size] / total if total else 0.0
        lines.append(f"{size:>10}{counts[size]:>12,}{share:>10.1%}")

    # The single number to decide on: a clip with one hypothesis is a clip the
    # corrector cannot do anything with.
    usable = sum(count for size, count in counts.items() if size >= 3)
    single = counts.get(1, 0)
    lines.append("")
    lines.append(f"  3+ distinct (usable):  {usable:,} ({usable / total:.1%})" if total else "")
    lines.append(f"  1 distinct (inert):    {single:,} ({single / total:.1%})" if total else "")
    return lines


def mean_candidates(nbest: dict) -> float:
    if not nbest:
        return 0.0
    return sum(len(entry.get("nbest", [])) for entry in nbest.values()) / len(nbest)


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser(description="Report n-best candidate diversity.")
    parser.add_argument(
        "--nbest",
        type=Path,
        default=config["nbest_path"],
        help="The n-best corpus to inspect.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="A second corpus to compare against, usually the current one.",
    )
    arguments = parser.parse_args()

    if not arguments.nbest.is_file():
        raise SystemExit(f"No such corpus: {arguments.nbest}")

    nbest = json.loads(arguments.nbest.read_text(encoding="utf-8"))
    counts = distinct_counts(nbest)

    lines = ["", *describe(counts, len(nbest), str(arguments.nbest))]
    lines.append(f"  mean candidates/clip:  {mean_candidates(nbest):.2f}")

    if arguments.baseline and arguments.baseline.is_file():
        other = json.loads(arguments.baseline.read_text(encoding="utf-8"))
        lines += ["", *describe(distinct_counts(other), len(other), str(arguments.baseline))]

    lines += [
        "",
        "CHSER train, for reference (MyST classroom speech, not CHILDES)",
        "-" * 56,
        f"{'distinct':>10}{'share':>22}",
    ]
    for size, share in sorted(CHSER_TRAIN_DISTRIBUTION.items()):
        lines.append(f"{size:>10}{share:>22.1%}")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
