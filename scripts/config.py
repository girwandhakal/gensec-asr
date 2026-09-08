"""
What this file is for:
Loads configs/baseline.yaml and resolves every path the pipeline uses,
preferring the local dataset copy when one exists so the same code runs
unchanged on a laptop and on the HPC.
"""

from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path | None = None) -> dict:
    path = path or PROJECT_ROOT / "configs" / "baseline.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["config_path"] = path

    for key in ("data_dir", "work_dir", "results_dir"):
        config[key] = PROJECT_ROOT / config[key]

    # Use the local dataset when it is there, otherwise the HPC copy.
    local_media = (PROJECT_ROOT / config["local_media_dir"]).resolve()
    config["media_dir"] = local_media if local_media.is_dir() else Path(config["hpc_media_dir"])

    # Labels sit at the root of the media directory, so audio and transcripts
    # can never come from two different copies of the dataset.
    config["reference_report_csv"] = config["media_dir"] / config["reference_report_filename"]

    data_dir = config["data_dir"]
    config["reference_map_path"] = data_dir / "utterance_id_to_reference.json"
    config["metadata_path"] = data_dir / "utterance_metadata.json"
    config["nbest_path"] = data_dir / "utterance_id_to_nbest.json"
    config["processed_path"] = data_dir / "processed_gensec.json"
    config["dropped_path"] = data_dir / "dropped_gensec.json"
    config["splits_dir"] = data_dir / "splits"
    config["predictions_dir"] = data_dir / "predictions"

    return config


def clip_seconds(utterance_id: str) -> float | None:
    """Clip duration, read straight out of the id.

    Clips are named `<transcript_hash>_<start_ms>_<end_ms>_<ordinal>`, so how
    long the audio was is known before it is opened. Both the ASR decode budget
    and the correction decode budget depend on it.
    """
    parts = utterance_id.split("_")
    if len(parts) < 4:
        return None

    try:
        start, end = int(parts[-3]), int(parts[-2])
    except ValueError:
        return None

    return (end - start) / 1000 if end > start else None


def transcript_id(utterance_id: str) -> str:
    """The source transcript a clip was cut from, read straight out of the id.

    Clips are named `<transcript_hash>_<start_ms>_<end_ms>_<ordinal>`, so every
    utterance from one recording session shares a prefix. This is the unit the
    train/test split has to respect: utterances from the same session share a
    child, a recording setup and a vocabulary, so splitting below this level
    lets the corrector memorize a session it is later scored on.
    """
    return utterance_id.split("_")[0]


def predictions_path(config: dict, mode: str) -> Path:
    """Where one inference mode's raw predictions live."""
    return config["predictions_dir"] / f"test_predictions_{mode}.csv"


if __name__ == "__main__":
    for key, value in load_config().items():
        print(f"{key}: {value}")
