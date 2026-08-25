"""
What this file is for:
The single entry point. Runs all six stages in order, so nothing has to be
run by hand.

High-level role in the pipeline:
Archives the previous evaluation, then walks the stages from the audio on
disk to the WER report. Every stage skips itself when its output already
exists, which is what lets a job that hit the walltime just be resubmitted.
"""

from __future__ import annotations

import shutil
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_gensec_dataset
import build_reference_map
import evaluate
import generate_nbest
import postprocess
import train
from config import load_config, predictions_path

TOTAL_STAGES = 6


def archive_previous_results(results_dir: Path, history_dir: Path) -> None:
    """Move the last run aside so its numbers stay readable."""
    if not results_dir.is_dir() or not any(results_dir.iterdir()):
        return

    destination = history_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(results_dir), str(destination))
    print(f"Archived previous results to {destination}")


@contextmanager
def stage(number: int, name: str):
    print(f"\n===== STAGE {number}/{TOTAL_STAGES}: {name} =====")
    started = time.perf_counter()
    yield
    print(f"----- Stage {number}/{TOTAL_STAGES} finished in {time.perf_counter() - started:.1f}s -----")


def main() -> None:
    config = load_config()

    if not config["media_dir"].is_dir():
        raise SystemExit(f"Media directory not found: {config['media_dir']}")
    print(f"Media:  {config['media_dir']}")
    print(f"Output: {config['data_dir']}")

    print("\n===== HOUSEKEEPING: ARCHIVE PREVIOUS RESULTS =====")
    archive_previous_results(config["results_dir"], config["results_dir"].parent / "evaluation_history")

    # Stages 1, 3, 5 and 6 take seconds and read data that can still be
    # growing, so they always rerun rather than caching a stale answer. Only
    # the two expensive stages are skipped once they have output.
    with stage(1, "BUILD REFERENCE MAP"):
        build_reference_map.main(config)

    with stage(2, "GENERATE WHISPER N-BEST"):
        # Resumes internally, so it always runs; it exits quickly if complete.
        generate_nbest.main(config)

    with stage(3, "BUILD GENSEC DATASET"):
        build_gensec_dataset.main(config)

    with stage(4, "TRAIN AND RUN INFERENCE"):
        # The costly one. Delete the predictions to retrain on a grown dataset.
        first_mode = config["inference_modes"][0]
        if predictions_path(config, first_mode).is_file():
            print(f"Using existing {predictions_path(config, first_mode)}")
        else:
            train.main(config)

    with stage(5, "POSTPROCESS PREDICTIONS"):
        postprocess.main(config)

    with stage(6, "EVALUATE"):
        evaluate.main(config)

    # Keep the settings next to the numbers they produced.
    shutil.copy(config["config_path"], config["results_dir"] / "config_used.yaml")

    print("\n===== GENSEC PIPELINE COMPLETE =====")


if __name__ == "__main__":
    main()
