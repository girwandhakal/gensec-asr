"""
What this file is for:
Stage 3. Joins Whisper's candidates to the answer key and writes the
supervised correction dataset.

High-level role in the pipeline:
This is the step that turns speech recognition into a text-to-text problem:
input is a short list of competing hypotheses, target is the true transcript.
Utterances whose candidates all agree carry no information for the corrector,
so they are dropped here - and the drop log is worth reading, because a high
drop rate means the decoding wasn't diverse enough.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load_config


def collect_hypotheses(entry: dict, limit: int) -> list[str]:
    """The 1-best first, then the remaining distinct candidates."""
    hypotheses: list[str] = []

    one_best = (entry.get("1best_text") or "").strip()
    if one_best:
        hypotheses.append(one_best)

    for item in entry.get("nbest", []):
        text = (item.get("text") or "").strip()
        if text and text not in hypotheses:
            hypotheses.append(text)

    return hypotheses[:limit]


def build_dataset(config: dict) -> tuple[list[dict], list[dict]]:
    nbest = json.loads(config["nbest_path"].read_text(encoding="utf-8"))
    references = json.loads(config["reference_map_path"].read_text(encoding="utf-8"))

    kept: list[dict] = []
    dropped: list[dict] = []

    for utterance_id, entry in nbest.items():
        hypotheses = collect_hypotheses(entry, config["max_hypotheses"])
        reference = (references.get(utterance_id) or "").strip()

        if utterance_id not in references:
            reason = "no_reference_transcript"
        elif not reference:
            reason = "empty_reference"
        elif len(hypotheses) < config["min_hypotheses"]:
            reason = "fewer_than_min_unique_hypotheses"
        else:
            kept.append({"id": utterance_id, "input": hypotheses, "output": reference})
            continue

        dropped.append({
            "id": utterance_id,
            "reason": reason,
            "unique_hypotheses": len(hypotheses),
            "reference": reference,
        })

    return kept, dropped


def main(config: dict | None = None) -> None:
    config = config or load_config()

    for path in (config["nbest_path"], config["reference_map_path"]):
        if not path.is_file():
            raise SystemExit(f"Missing input: {path}")

    kept, dropped = build_dataset(config)
    total = len(kept) + len(dropped)

    config["processed_path"].parent.mkdir(parents=True, exist_ok=True)
    config["processed_path"].write_text(
        json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    config["dropped_path"].write_text(
        json.dumps(dropped, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Utterances seen: {total:,}")
    print(f"Kept:            {len(kept):,} ({len(kept) / total:.1%})" if total else "Kept: 0")
    print(f"Dropped:         {len(dropped):,}")
    for reason, count in Counter(d["reason"] for d in dropped).most_common():
        print(f"  {reason}: {count:,}")

    # How much disagreement the corrector actually gets to work with.
    sizes = Counter(len(example["input"]) for example in kept)
    print("Hypotheses per kept example:")
    for size in sorted(sizes):
        print(f"  {size}: {sizes[size]:,}")

    print(f"Wrote {config['processed_path']}")


if __name__ == "__main__":
    main()
