"""
What this file is for:
Stage 3. Joins Whisper's candidates to the answer key and writes the
supervised correction dataset.

High-level role in the pipeline:
This is the step that turns speech recognition into a text-to-text problem:
input is a short list of competing hypotheses, target is the true transcript.
Single-hypothesis examples are retained when min_hypotheses is 1.
Ground-truth strings are copied unchanged from the upstream reference map.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_filter import audio_filter
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
    upstream_drops_path = config["reference_map_path"].parent / "reference_alignment_drops.json"
    upstream_drops = (json.loads(upstream_drops_path.read_text(encoding="utf-8"))
                      if upstream_drops_path.is_file() else {})

    kept: list[dict] = []
    dropped: list[dict] = []

    for utterance_id, entry in nbest.items():
        hypotheses = collect_hypotheses(entry, config["max_hypotheses"])
        reference = references.get(utterance_id) or ""

        if utterance_id not in references:
            reason = upstream_drops.get(utterance_id, {}).get("reason", "no_reference_transcript")
        elif not reference:
            reason = "empty_reference"
        elif (alignment_issue := audio_filter(utterance_id, reference, config)):
            reason = alignment_issue
        elif len(hypotheses) < config["min_hypotheses"]:
            reason = "fewer_than_min_unique_hypotheses"
        else:
            kept.append({"id": utterance_id, "input": hypotheses, "output": reference})
            continue

        dropped.append({
            "id": utterance_id,
            "reason": reason,
            "group": upstream_drops.get(utterance_id, {}).get("group", ""),
            "corpus": upstream_drops.get(utterance_id, {}).get("corpus", ""),
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

    metadata = json.loads(config["metadata_path"].read_text(encoding="utf-8"))
    audit = Counter()
    for item in dropped:
        meta = metadata.get(item["id"], {})
        audit[(item.get("group") or meta.get("group", "unknown"),
               item.get("corpus") or meta.get("corpus", "unknown"), item["reason"])] += 1
    audit_path = config["data_dir"] / "alignment_exclusions.json"
    audit_path.write_text(json.dumps([
        {"group": group, "corpus": corpus, "reason": reason, "count": count}
        for (group, corpus, reason), count in sorted(audit.items())
    ], indent=2), encoding="utf-8")
    print(f"Wrote {audit_path}")

    # How much disagreement the corrector actually gets to work with.
    sizes = Counter(len(example["input"]) for example in kept)
    print("Hypotheses per kept example:")
    for size in sorted(sizes):
        print(f"  {size}: {sizes[size]:,}")

    print(f"Wrote {config['processed_path']}")


if __name__ == "__main__":
    main()
