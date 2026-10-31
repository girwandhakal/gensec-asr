"""Run or resume a shared Whisper dataset comparison with FLAN-T5 and Qwen3."""

from __future__ import annotations

import json
import os
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
import generate_figures
import generate_nbest
import run_analysis
import train
import train_qwen
from config import PROJECT_ROOT, load_config, predictions_path
from reference_io import read_prediction_rows

TOTAL_STAGES = 12


@contextmanager
def stage(number: int, name: str):
    started = time.perf_counter()
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"\n===== STAGE {number}/{TOTAL_STAGES} START at {now}: {name} =====", flush=True)
    try:
        yield
    except BaseException as error:
        elapsed = time.perf_counter() - started
        now = datetime.now().astimezone().isoformat(timespec="seconds")
        print(
            f"===== STAGE {number}/{TOTAL_STAGES} FAILED at {now} after {elapsed:.1f}s: "
            f"{type(error).__name__}: {error} =====",
            flush=True,
        )
        raise
    elapsed = time.perf_counter() - started
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"===== STAGE {number}/{TOTAL_STAGES} DONE at {now} in {elapsed:.1f}s: {name} =====", flush=True)


def fresh_run_directory(work_root: Path) -> Path:
    requested = os.environ.get("GENSEC_RUN_DIR")
    destination = Path(requested).resolve() if requested else work_root / "comparison"
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / "comparison_complete.json"
    if marker.is_file():
        # A completed run is replaceable; a partial run must retain its
        # checkpoints and predictions so it can resume after a walltime.
        default_run = work_root / "comparison"
        model_root = work_root if destination.resolve() == default_run.resolve() else destination / "models"
        for stale in (destination / "predictions", model_root / "flan_t5", model_root / "qwen3_0.6b"):
            if stale.is_dir():
                shutil.rmtree(stale)
        marker.unlink()
        print(f"Reusing comparison workspace for a new run: {destination}", flush=True)
    else:
        print(f"Using comparison workspace: {destination}", flush=True)
    return destination


def configure_run(base: dict, run_root: Path) -> dict:
    config = dict(base)
    data_dir = run_root / "shared_data"
    default_run = base["work_dir"] / "comparison"
    model_root = base["work_dir"] if run_root.resolve() == default_run.resolve() else run_root / "models"
    config.update({
        "data_dir": data_dir,
        "reference_map_path": data_dir / "utterance_id_to_reference.json",
        "metadata_path": data_dir / "utterance_metadata.json",
        "nbest_path": data_dir / "utterance_id_to_nbest_greedy_sample10.json",
        "whisper_samples_path": data_dir / "utterance_id_to_whisper_samples.json",
        "processed_path": data_dir / "processed_gensec.json",
        "dropped_path": data_dir / "dropped_gensec.json",
        "splits_dir": data_dir / "splits",
        "predictions_dir": run_root / "predictions",
        "work_dir": model_root / "flan_t5",
        "qwen_work_dir": model_root / "qwen3_0.6b",
        "run_root": run_root,
    })
    return config


def reuse_whisper_artifacts(config: dict) -> None:
    """Seed a new run with prior transcripts made using the same ASR settings."""
    destinations = [config["nbest_path"], config["whisper_samples_path"]]
    if any(path.exists() for path in destinations):
        return

    current_signature = generate_nbest.decode_signature(config)
    artifact_names = (
        "utterance_id_to_nbest_greedy_sample10.json",
        "utterance_id_to_whisper_samples.json",
    )
    # Old comparison folders can supply a matching cache on the first run;
    # they are read only and remain where they are.
    runs_root = PROJECT_ROOT / "runs"
    candidates = sorted(
        (
            path for path in runs_root.glob("comparison_*")
            if path.is_dir() and path.resolve() != config["run_root"].resolve()
        ),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )

    for previous_run in candidates:
        previous_data = previous_run / "shared_data"
        sources = [previous_data / name for name in artifact_names]
        signatures = [path.with_suffix(".signature.json") for path in sources]
        if not all(path.is_file() for path in (*sources, *signatures)):
            continue
        if any(path.read_text(encoding="utf-8") != current_signature for path in signatures):
            continue

        config["data_dir"].mkdir(parents=True, exist_ok=True)
        for source, destination in zip(sources, destinations):
            shutil.copy2(source, destination)
            shutil.copy2(source.with_suffix(".signature.json"), destination.with_suffix(".signature.json"))
        print(
            f"Reusing Whisper outputs from {previous_run} "
            f"(decode settings match: {current_signature})",
            flush=True,
        )
        return

    print("No prior Whisper outputs with matching decode settings; transcribing audio.", flush=True)


def model_config(shared: dict, name: str) -> dict:
    config = dict(shared)
    config["predictions_dir"] = shared["run_root"] / "predictions" / name
    config["results_dir"] = shared["results_root"] / name
    config["figure_dir"] = config["results_dir"] / "figures"
    config["model_display_name"] = "Whisper + FLAN-T5" if name == "flan_t5" else "Whisper + Qwen3-0.6B"
    return config


def archive_previous_comparison_outputs(results_root: Path, history_root: Path) -> Path | None:
    """Copy prior comparison reports aside before refreshing the active results.

    The existing results remain in place until this run writes replacements.
    Model checkpoints and run inputs are not part of this archive.
    """
    named_outputs = ("flan_t5", "qwen3_0.6b", "model_comparison.json", "model_comparison.txt")
    existing = [results_root / name for name in named_outputs if (results_root / name).exists()]
    if not existing:
        return None

    latest_write = max(path.stat().st_mtime for path in existing)
    stamp = datetime.fromtimestamp(latest_write).strftime("comparison_%Y%m%d_%H%M%S")
    destination = history_root / stamp
    suffix = 1
    while destination.exists():
        suffix += 1
        destination = history_root / f"{stamp}_{suffix}"

    destination.mkdir(parents=True, exist_ok=False)
    for source in existing:
        target = destination / source.name
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
    print(f"Copied previous comparison results to {destination}")
    return destination


def run_model_pipeline(config: dict, model_name: str, first_stage: int, base: dict | None = None) -> dict:
    config["predictions_dir"].mkdir(parents=True, exist_ok=True)
    config["results_dir"].mkdir(parents=True, exist_ok=True)
    if model_name == "flan_t5":
        with stage(first_stage, "FLAN-T5 training and inference"):
            if base is not None:
                copy_current_flan_checkpoint(config, base)
            train.main(config)
    else:
        with stage(first_stage, "Qwen3-0.6B training and inference"):
            train_qwen.main(config)

    with stage(first_stage + 1, f"{model_name} WER evaluation"):
        evaluate.main(config)
    with stage(first_stage + 2, f"{model_name} child-level analysis"):
        run_analysis.main(config)
    with stage(first_stage + 3, f"{model_name} publication figures"):
        generate_figures.main(config)
    shutil.copy(config["config_path"], config["results_dir"] / "config_used.yaml")
    (config["results_dir"] / "run_config.json").write_text(
        json.dumps({
            "run_root": str(config["run_root"]),
            "model": config["gensec_model_id"] if model_name == "flan_t5" else config["qwen_model_id"],
            "display_name": config["model_display_name"],
            "inference_modes": config["inference_modes"],
        }, indent=2),
        encoding="utf-8",
    )
    return json.loads((config["results_dir"] / "metrics.json").read_text(encoding="utf-8"))


def copy_current_flan_checkpoint(config: dict, base: dict) -> None:
    """Reuse the current FLAN weights only when their saved data signature matches."""
    source = base["work_dir"] / "final_checkpoint"
    signature_file = source / "training_signature.txt"
    if not signature_file.is_file():
        print("No signed FLAN checkpoint found; this comparison will train FLAN.")
        return
    if signature_file.read_text(encoding="utf-8") != train.training_signature(config):
        print("The existing FLAN checkpoint does not match the regenerated training data; it will be retrained.")
        return

    destination = config["work_dir"] / "final_checkpoint"
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    print(f"Copied matching FLAN checkpoint into this run: {destination}")


def write_comparison(output_root: Path, flan: dict, qwen: dict, flan_config: dict, qwen_config: dict) -> None:
    mode = flan_config["inference_modes"][0]
    flan_rows = read_prediction_rows(predictions_path(flan_config, mode))
    qwen_rows = read_prediction_rows(predictions_path(qwen_config, mode))
    flan_pairs = {(row["id"], row["truth"]) for row in flan_rows}
    qwen_pairs = {(row["id"], row["truth"]) for row in qwen_rows}
    if len(flan_pairs) != len(flan_rows) or len(qwen_pairs) != len(qwen_rows) or flan_pairs != qwen_pairs:
        raise ValueError("FLAN and Qwen predictions do not cover the same test utterances and references.")
    corrected = f"gensec_{mode}"
    if flan["whisper_1best"]["utterances"] != qwen["whisper_1best"]["utterances"]:
        raise ValueError("The two model runs were evaluated on different test utterances.")
    report = {
        "test_utterances": flan["whisper_1best"]["utterances"],
        "systems": {
            "whisper_1best": flan["whisper_1best"],
            "flan_t5": flan[corrected],
            "qwen3_0.6b": qwen[corrected],
        },
    }
    report["wer_difference_qwen_minus_flan"] = (
        report["systems"]["qwen3_0.6b"]["wer"] - report["systems"]["flan_t5"]["wer"]
    )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "model_comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "Shared test set model comparison",
        f"Test utterances: {report['test_utterances']:,}",
        f"Whisper 1-best WER: {report['systems']['whisper_1best']['wer']:.2%}",
        f"Whisper + FLAN-T5 WER: {report['systems']['flan_t5']['wer']:.2%}",
        f"Whisper + Qwen3-0.6B WER: {report['systems']['qwen3_0.6b']['wer']:.2%}",
        f"Qwen minus FLAN WER: {report['wer_difference_qwen_minus_flan']:+.2%}",
    ]
    (output_root / "model_comparison.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    base = load_config()
    base["results_root"] = base["results_dir"]
    run_root = fresh_run_directory(base["work_dir"])
    shared = configure_run(base, run_root)
    archive_previous_comparison_outputs(
        base["results_root"], PROJECT_ROOT / "evaluation_history"
    )
    print(f"Comparison run: {run_root}")
    print(f"Audio and references: {shared['media_dir']}")
    reuse_whisper_artifacts(shared)

    with stage(1, "import finalized audio/reference pairs"):
        build_reference_map.main(shared)
    with stage(2, "generate Whisper greedy transcript and ten raw samples"):
        generate_nbest.main(shared)
    with stage(3, "build shared correction dataset"):
        build_gensec_dataset.main(shared)

    flan_config = model_config(shared, "flan_t5")
    flan_metrics = run_model_pipeline(flan_config, "flan_t5", 4, base)
    qwen_config = model_config(shared, "qwen3_0.6b")
    qwen_metrics = run_model_pipeline(qwen_config, "qwen3_0.6b", 8)
    with stage(12, "compare FLAN-T5 and Qwen3-0.6B on the same test set"):
        write_comparison(base["results_root"], flan_metrics, qwen_metrics, flan_config, qwen_config)
    shutil.copy(base["config_path"], run_root / "config_used.yaml")
    (run_root / "comparison_complete.json").write_text(
        json.dumps({"results_dir": str(base["results_root"])}, indent=2), encoding="utf-8"
    )
    print(f"\n===== COMPARISON COMPLETE: {run_root} =====")


if __name__ == "__main__":
    main()
