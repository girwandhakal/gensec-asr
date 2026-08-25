"""
What this file is for:
Stage 6. Scores word error rate for every system on the same test utterances.

High-level role in the pipeline:
This is the result. Raw Whisper 1-best is the baseline to beat; the whole
project is the gap between it and the corrected output. Everything is scored
on the identical set of utterances, or the comparison means nothing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load_config, predictions_path
from text import normalize_for_scoring

WORST_EXAMPLES = 10


def count_errors(reference: list[str], hypothesis: list[str]) -> tuple[int, int, int]:
    """Levenshtein alignment over words, returning substitutions/deletions/insertions."""
    rows, columns = len(reference) + 1, len(hypothesis) + 1
    costs = [[0] * columns for _ in range(rows)]
    operations = [[""] * columns for _ in range(rows)]

    for i in range(1, rows):
        costs[i][0], operations[i][0] = i, "D"
    for j in range(1, columns):
        costs[0][j], operations[0][j] = j, "I"

    for i in range(1, rows):
        for j in range(1, columns):
            if reference[i - 1] == hypothesis[j - 1]:
                costs[i][j], operations[i][j] = costs[i - 1][j - 1], "M"
                continue

            substitution = costs[i - 1][j - 1] + 1
            deletion = costs[i - 1][j] + 1
            insertion = costs[i][j - 1] + 1
            best = min(substitution, deletion, insertion)

            costs[i][j] = best
            operations[i][j] = "S" if best == substitution else ("D" if best == deletion else "I")

    substitutions = deletions = insertions = 0
    i, j = len(reference), len(hypothesis)
    while i > 0 or j > 0:
        operation = operations[i][j]
        if operation in ("M", "S"):
            i, j = i - 1, j - 1
            substitutions += operation == "S"
        elif operation == "D":
            i -= 1
            deletions += 1
        else:
            j -= 1
            insertions += 1

    return substitutions, deletions, insertions


def score(pairs: list[tuple[str, str, str]]) -> dict:
    """pairs is (utterance_id, reference, hypothesis)."""
    totals = {"substitutions": 0, "deletions": 0, "insertions": 0, "reference_words": 0}
    exact = 0
    worst = []

    for utterance_id, reference_text, hypothesis_text in pairs:
        reference = normalize_for_scoring(reference_text).split()
        hypothesis = normalize_for_scoring(hypothesis_text).split()

        substitutions, deletions, insertions = count_errors(reference, hypothesis)
        totals["substitutions"] += substitutions
        totals["deletions"] += deletions
        totals["insertions"] += insertions
        totals["reference_words"] += len(reference)
        exact += reference == hypothesis

        if reference:
            errors = substitutions + deletions + insertions
            worst.append((errors / len(reference), utterance_id,
                          " ".join(reference), " ".join(hypothesis)))

    errors = totals["substitutions"] + totals["deletions"] + totals["insertions"]
    return {
        **totals,
        "utterances": len(pairs),
        "total_errors": errors,
        "wer": errors / totals["reference_words"] if totals["reference_words"] else 0.0,
        "exact_match_rate": exact / len(pairs) if pairs else 0.0,
        "worst": sorted(worst, key=lambda item: -item[0])[:WORST_EXAMPLES],
    }


def collect_systems(config: dict) -> dict[str, list[tuple[str, str, str]]]:
    """Build (id, truth, prediction) triples for each system, on the same utterances."""
    mode = config["inference_modes"][0]
    frame = pd.read_csv(predictions_path(config, mode)).fillna("")
    nbest = json.loads(config["nbest_path"].read_text(encoding="utf-8"))

    systems = {
        "whisper_1best": [
            (row["id"], row["truth"], nbest.get(row["id"], {}).get("1best_text", ""))
            for row in frame.to_dict("records")
        ],
        f"gensec_{mode}": [
            (row["id"], row["truth"], row["prediction"]) for row in frame.to_dict("records")
        ],
    }

    if config["cleaned_predictions_path"].is_file():
        cleaned = pd.read_csv(config["cleaned_predictions_path"]).fillna("")
        systems[f"gensec_{mode}_cleaned"] = [
            (row["id"], row["truth"], row["prediction"]) for row in cleaned.to_dict("records")
        ]

    return systems


def main(config: dict | None = None) -> None:
    config = config or load_config()

    mode = config["inference_modes"][0]
    if not predictions_path(config, mode).is_file():
        raise SystemExit(f"Missing predictions: {predictions_path(config, mode)}")

    results = {name: score(pairs) for name, pairs in collect_systems(config).items()}
    config["results_dir"].mkdir(parents=True, exist_ok=True)

    lines = [
        "GenSEC evaluation",
        "=" * 72,
        f"Test utterances: {next(iter(results.values()))['utterances']:,}",
        "",
        f"{'system':<28}{'WER':>10}{'exact':>10}{'S':>9}{'D':>9}{'I':>9}",
        "-" * 72,
    ]
    for name, result in results.items():
        lines.append(
            f"{name:<28}{result['wer']:>9.2%}{result['exact_match_rate']:>10.2%}"
            f"{result['substitutions']:>9,}{result['deletions']:>9,}{result['insertions']:>9,}"
        )

    # Positive means fewer errors than the raw ASR baseline - the whole point.
    baseline = results["whisper_1best"]["wer"]
    for name, result in results.items():
        if name != "whisper_1best" and baseline:
            reduction = (baseline - result["wer"]) / baseline
            verdict = "better" if reduction > 0 else "worse"
            lines.append(
                f"\n{name} vs whisper_1best: {reduction:+.2%} relative WER reduction ({verdict})"
            )

    for name, result in results.items():
        lines.append(f"\n\nWorst utterances - {name}")
        lines.append("-" * 72)
        for sentence_wer, utterance_id, reference, hypothesis in result["worst"]:
            lines.append(f"{utterance_id}  sentence_wer={sentence_wer:.2f}")
            lines.append(f"  ref: {reference}")
            lines.append(f"  hyp: {hypothesis}")

    report = "\n".join(lines)
    print(report)
    (config["results_dir"] / "wer_report.txt").write_text(report, encoding="utf-8")

    metrics = {
        name: {key: value for key, value in result.items() if key != "worst"}
        for name, result in results.items()
    }
    (config["results_dir"] / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {config['results_dir'] / 'wer_report.txt'}")


if __name__ == "__main__":
    main()
