"""
What this file is for:
Stage 1. Builds the answer key: one JSON mapping each audio clip's utterance
id to the ground-truth transcript of what is said in it.

High-level role in the pipeline:
media_download_report.csv is a download log, not a label file. Most of its
rows describe clips that were never cut, some clips on disk have no row at
all, and the transcripts it does carry are raw CHAT markup. This reconciles
all of that once, so the training target and the scoring reference can never
disagree.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path, PureWindowsPath

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load_config
from text import normalize_chat

# One row in this file has a very long utterance field.
csv.field_size_limit(10_000_000)


def build_reference_map(config: dict) -> tuple[dict[str, str], dict[str, dict]]:
    report_path = config["reference_report_csv"]
    if not report_path.is_file():
        raise SystemExit(f"Reference report not found: {report_path}")

    clips_on_disk = {p.stem for p in config["media_dir"].rglob("*" + config["audio_extension"])}

    references: dict[str, str] = {}
    # Corpus and group travel with the reference so results can be broken down
    # by them later. `group` is the late-talker / typically-developing label,
    # which is the distinction this dataset has and the published child GenSEC
    # work does not - losing it in a single corpus-level WER wastes it.
    metadata: dict[str, dict] = {}
    described_clips: set[str] = set()
    no_clip = 0
    missing_file = 0
    empty_after_cleaning = 0

    with report_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            output = (row.get("output") or "").strip()
            if not output:
                no_clip += 1  # utterance was never cut into a clip
                continue

            # `output` is a Windows absolute path baked in when the report was
            # generated on a Windows laptop. `Path` on Linux won't split on
            # backslashes, so this must use PureWindowsPath (it accepts both
            # `\` and `/`) rather than the platform `Path`.
            utterance_id = PureWindowsPath(output).stem
            described_clips.add(utterance_id)
            if utterance_id not in clips_on_disk:
                missing_file += 1
                continue

            transcript = normalize_chat(row.get("utterance"), config["remove_unintelligible"])
            if not transcript:
                empty_after_cleaning += 1  # noise markers only, nothing said
                continue

            references[utterance_id] = transcript
            metadata[utterance_id] = {
                "corpus": (row.get("corpus") or "").strip(),
                "group": (row.get("group") or "").strip(),
                "child_id": (row.get("child_id") or "").strip(),
            }

    orphan_clips = clips_on_disk - described_clips

    print(f"Report:                   {report_path}")
    print(f"Clips on disk:            {len(clips_on_disk):,}")
    print(f"Rows with no clip:        {no_clip:,}")
    print(f"Rows whose clip is gone:  {missing_file:,}")
    print(f"Clips with no report row: {len(orphan_clips):,}")
    print(f"Empty after cleaning:     {empty_after_cleaning:,}")
    print(f"References kept:          {len(references):,}")

    groups = Counter(entry["group"] for entry in metadata.values())
    print(f"Group split:              {dict(groups.most_common())}")

    if not references:
        raise SystemExit("No references were built; check the media path in baseline.yaml.")

    return references, metadata


def main(config: dict | None = None) -> None:
    config = config or load_config()
    references, metadata = build_reference_map(config)

    config["reference_map_path"].parent.mkdir(parents=True, exist_ok=True)
    with config["reference_map_path"].open("w", encoding="utf-8") as handle:
        json.dump(references, handle, ensure_ascii=False, indent=2)
    print(f"Wrote {config['reference_map_path']}")

    with config["metadata_path"].open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(f"Wrote {config['metadata_path']}")


if __name__ == "__main__":
    main()
