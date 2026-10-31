"""Import finalized ground truth from the upstream media report without cleaning it."""
from __future__ import annotations
import csv
import json
import os
from collections import Counter
from pathlib import Path


def alignment_reason(utterance_id: str, reference: str, config: dict) -> str:
    """Apply the same interval checks before scheduling expensive ASR."""
    parts = utterance_id.split("_")
    try:
        seconds = (int(parts[-2]) - int(parts[-3])) / 1000
    except (IndexError, ValueError):
        return "unparseable_clip_interval"
    if seconds <= 0:
        return "unparseable_clip_interval"
    if seconds < config.get("min_clip_seconds", 0.5):
        return "point_bullet_timestamp" if seconds <= 0.1 + 1e-9 else "clip_too_short"
    words = len(reference.split())
    if words / seconds > config.get("max_reference_words_per_second", 4.0):
        return "impossible_speech_rate"
    if seconds > config.get("long_single_word_seconds", 5.0) and words <= 1:
        return "long_clip_short_reference"
    return ""

REQUIRED_COLUMNS = {"utterance_id", "utterance", "audio_path", "raw_utterance",
                    "corpus", "group", "child_id"}

def build_reference_map(config: dict) -> tuple[dict[str, str], dict[str, dict]]:
    report_path = config["reference_report_csv"]
    media_dir = config["media_dir"].resolve()
    csv.field_size_limit(10_000_000)
    references, metadata = {}, {}
    exclusions = Counter()
    alignment_audit = Counter()
    alignment_drops = {}
    seen = set()
    directories = {}
    with report_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Media report is not finalized (missing {sorted(missing)}). "
                             "Run asr-dataset-pipelines/src/media_pipeline/ground_truth.py first.")
        for line, row in enumerate(reader, 2):
            if not row["utterance"].strip():
                exclusions["empty_cleaned_reference"] += 1
                continue
            if not row["audio_path"]:
                exclusions["no_audio_path"] += 1
                continue
            uid, reference = row["utterance_id"], row["utterance"]
            if not uid or uid in seen:
                raise ValueError(f"Missing or duplicate utterance ID on report row {line}: {uid}")
            seen.add(uid)
            relative = Path(row["audio_path"])
            if not row["audio_path"] or relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Audio path escapes media directory on report row {line}")
            if relative.parent not in directories:
                parent = (media_dir / relative.parent).resolve()
                if not parent.is_relative_to(media_dir):
                    raise ValueError(f"Audio path escapes media directory on report row {line}")
                files = {}
                if parent.is_dir():
                    with os.scandir(parent) as entries:
                        files = {entry.name: entry.stat().st_size for entry in entries
                                 if entry.is_file(follow_symlinks=False)}
                directories[relative.parent] = files
            if relative.stem != uid or not directories[relative.parent].get(relative.name, 0):
                raise ValueError(f"Missing audio or ID/path mismatch on report row {line}: {relative}")
            issue = alignment_reason(uid, reference, config)
            if issue:
                exclusions[issue] += 1
                alignment_audit[(row["group"], row["corpus"], issue)] += 1
                alignment_drops[uid] = {"reason": issue, "group": row["group"],
                                        "corpus": row["corpus"]}
                continue
            # Ground truth is copied exactly as a string, never normalized here.
            references[uid] = reference
            metadata[uid] = {key: row[key] for key in ("corpus", "group", "child_id", "audio_path")}
    if not references:
        raise ValueError("No ready audio/reference pairs in the finalized media report")
    print(f"Finalized reference/audio pairs: {len(references):,}")
    print(f"Upstream exclusions: {dict(exclusions)}")
    if "data_dir" in config:
        audit_path = config["data_dir"] / "reference_alignment_exclusions.json"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps([
            {"group": group, "corpus": corpus, "reason": reason, "count": count}
            for (group, corpus, reason), count in sorted(alignment_audit.items())
        ], indent=2), encoding="utf-8")
        (config["data_dir"] / "reference_alignment_drops.json").write_text(
            json.dumps(alignment_drops, indent=2), encoding="utf-8")
        print(f"Wrote {audit_path}")
    return references, metadata

def main(config: dict | None = None) -> None:
    if config is None:
        from config import load_config
        config = load_config()
    references, metadata = build_reference_map(config)
    config["reference_map_path"].parent.mkdir(parents=True, exist_ok=True)
    config["reference_map_path"].write_text(json.dumps(references, ensure_ascii=False, indent=2), encoding="utf-8")
    config["metadata_path"].write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
