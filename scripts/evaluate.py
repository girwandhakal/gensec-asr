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
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load_config, predictions_path
from text import normalize_for_scoring

WORST_EXAMPLES = 10


def align(reference: list[str], hypothesis: list[str]) -> list[tuple[str, int]]:
    """Levenshtein backtrace as (operation, reference index) steps.

    Operations are M/S/D/I; the index is the reference word the step consumed,
    or -1 for an insertion, which consumes none.
    """
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

    steps: list[tuple[str, int]] = []
    i, j = len(reference), len(hypothesis)
    while i > 0 or j > 0:
        operation = operations[i][j]
        if operation in ("M", "S"):
            steps.append((operation, i - 1))
            i, j = i - 1, j - 1
        elif operation == "D":
            steps.append((operation, i - 1))
            i -= 1
        else:
            steps.append((operation, -1))
            j -= 1

    return steps


def count_errors(reference: list[str], hypothesis: list[str]) -> tuple[int, int, int]:
    """Substitutions, deletions and insertions between two word sequences."""
    steps = align(reference, hypothesis)
    return (
        sum(operation == "S" for operation, _ in steps),
        sum(operation == "D" for operation, _ in steps),
        sum(operation == "I" for operation, _ in steps),
    )


def oracle_best_hypothesis(reference_text: str, hypotheses: list[str]) -> str:
    """The candidate with the fewest errors - the ceiling for picking, not composing.

    A correction model that could only ever return one of its inputs verbatim
    would score exactly this. The gap between it and the raw 1-best is how much
    the n-best list is worth at all; the gap between it and the model is how
    much of that the model captured.
    """
    reference = normalize_for_scoring(reference_text).split()

    best_text, fewest = "", None
    for text in hypotheses:
        errors = sum(count_errors(reference, normalize_for_scoring(text).split()))
        if fewest is None or errors < fewest:
            best_text, fewest = text, errors

    return best_text


def score_compositional_oracle(triples: list[tuple[str, str, list[str]]]) -> dict:
    """Lower bound on WER when any word may be taken from any hypothesis.

    A reference word counts as reachable if at least one candidate aligned to
    it exactly. This is optimistic - it does not require the kept words to form
    one consistent path through the candidates, and it charges nothing for
    insertions - so read it as a floor nobody can beat, not a target.
    """
    unreachable = 0
    reference_words = 0

    for _, reference_text, hypotheses in triples:
        reference = normalize_for_scoring(reference_text).split()
        reference_words += len(reference)

        reachable: set[int] = set()
        for text in hypotheses:
            steps = align(reference, normalize_for_scoring(text).split())
            reachable.update(index for operation, index in steps if operation == "M")

        unreachable += len(reference) - len(reachable)

    return {
        "substitutions": 0,
        "deletions": unreachable,
        "insertions": 0,
        "reference_words": reference_words,
        "utterances": len(triples),
        "total_errors": unreachable,
        "wer": unreachable / reference_words if reference_words else 0.0,
        "exact_match_rate": 0.0,
        "worst": [],
    }


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


def utterance_errors(pairs: list[tuple[str, str, str]]) -> list[int]:
    """Total errors per utterance, in the order given."""
    return [
        sum(count_errors(
            normalize_for_scoring(reference).split(),
            normalize_for_scoring(hypothesis).split(),
        ))
        for _, reference, hypothesis in pairs
    ]


def bootstrap_reductions(baseline: list[int], system: list[int], iterations: int, rng) -> list[float]:
    """Relative WER reductions over resampled utterance sets.

    Resampling utterances rather than words is what makes this paired: both
    systems are scored on the same draw every time, so the shared difficulty of
    an utterance cancels. Reference words are identical across systems, so the
    ratio of error counts is the relative reduction directly.

    Returned in draw order, NOT sorted. Two of these get differenced elementwise
    to compare groups, and that is only a resample-by-resample difference while
    the order is preserved - differencing two sorted lists compares quantile
    against quantile, which collapses the spread and reports a false certainty.
    """
    size = len(baseline)
    if size < 2:
        return []

    indices = range(size)
    reductions = []
    for _ in range(iterations):
        draw = rng.choices(indices, k=size)
        baseline_errors = sum(baseline[i] for i in draw)
        system_errors = sum(system[i] for i in draw)
        # Keep one entry per iteration either way, so elementwise differencing
        # between two groups stays aligned.
        reductions.append(
            (baseline_errors - system_errors) / baseline_errors if baseline_errors else 0.0
        )

    return reductions


def interval(values: list[float]) -> tuple[float, float]:
    """95% percentile interval. Sorts internally; callers keep draw order."""
    if not values:
        return (0.0, 0.0)
    ordered = sorted(values)
    return (
        ordered[int(.025 * len(ordered))],
        ordered[min(int(.975 * len(ordered)), len(ordered) - 1)],
    )


def candidates(nbest: dict, utterance_id: str) -> list[str]:
    """Every distinct hypothesis the ASR offered for one utterance."""
    entry = nbest.get(utterance_id, {})
    texts = [entry.get("1best_text", "")]
    texts += [item.get("text", "") for item in entry.get("nbest", [])]
    return list(dict.fromkeys(text for text in texts if text))


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

    # The ceiling. Without it a WER reduction has no scale: 4% of a reachable
    # 5% is most of what there was, 4% of a reachable 40% is barely a start.
    systems["oracle_nbest"] = [
        (row["id"], row["truth"], oracle_best_hypothesis(row["truth"], candidates(nbest, row["id"])))
        for row in frame.to_dict("records")
    ]

    return systems


LENGTH_BUCKETS = [
    ("1 word", 1, 1),
    ("2 words", 2, 2),
    ("3-5 words", 3, 5),
    ("6-10 words", 6, 10),
    ("11+ words", 11, 10**6),
]


def significance(systems: dict, metadata: dict, config: dict) -> tuple[list[str], dict]:
    """Confidence intervals on the WER reduction, overall and per group.

    The LT-vs-TD gap is the finding this project exists to report, and it rests
    on the smaller group. Without an interval there is no way to tell a real
    difference from 1,466 utterances of noise.
    """
    mode = config["inference_modes"][0]
    system_name = f"gensec_{mode}"
    if system_name not in systems:
        return [], {}

    iterations = config.get("bootstrap_iterations", 1000)
    rng = random.Random(config["seed"])

    baseline_pairs = systems["whisper_1best"]
    system_pairs = systems[system_name]
    baseline_all = utterance_errors(baseline_pairs)
    system_all = utterance_errors(system_pairs)

    lines = [
        f"\n\nSignificance ({iterations:,} bootstrap resamples of the utterance set)",
        "-" * 72,
        f"{system_name} vs whisper_1best",
    ]
    summary: dict[str, dict] = {}
    per_group: dict[str, list[float]] = {}

    subsets = {"overall": list(range(len(baseline_all)))}
    for group in sorted({metadata.get(uid, {}).get("group", "") for uid, _, _ in baseline_pairs} - {""}):
        subsets[group] = [
            i for i, (uid, _, _) in enumerate(baseline_pairs)
            if metadata.get(uid, {}).get("group") == group
        ]

    for label, rows in subsets.items():
        if len(rows) < 2:
            continue
        base = [baseline_all[i] for i in rows]
        syst = [system_all[i] for i in rows]
        point = (sum(base) - sum(syst)) / sum(base) if sum(base) else 0.0
        draws = bootstrap_reductions(base, syst, iterations, rng)
        low, high = interval(draws)
        per_group[label] = draws
        summary[label] = {"utterances": len(rows), "reduction": point, "ci_low": low, "ci_high": high}
        lines.append(
            f"  {label:<12}{len(rows):>8}{point:>10.2%}   95% CI [{low:+.2%}, {high:+.2%}]"
        )

    # Does correction help one group more than the other? Differencing the two
    # bootstrap distributions gives the interval; the share of draws on the
    # wrong side of zero is the two-sided p-value.
    if "LT" in per_group and "TD" in per_group:
        paired = min(len(per_group["LT"]), len(per_group["TD"]))
        gaps = [per_group["TD"][i] - per_group["LT"][i] for i in range(paired)]
        low, high = interval(gaps)
        point = summary["TD"]["reduction"] - summary["LT"]["reduction"]

        wrong_side = sum(1 for g in gaps if (g <= 0) == (point > 0))
        p_value = min(2 * wrong_side / len(gaps), 1.0) if gaps else 1.0
        # No draw landed on the wrong side, so all this resolution can say is
        # that p is below the smallest value it could have measured.
        floor = 2 / len(gaps) if gaps else 1.0
        shown = f"p = {p_value:.3f}" if wrong_side else f"p < {floor:.3f}"
        verdict = "significant at p<0.05" if p_value < 0.05 else "NOT significant at p<0.05"

        summary["TD_minus_LT"] = {"difference": point, "ci_low": low, "ci_high": high,
                                  "p_value": p_value, "p_below": None if wrong_side else floor,
                                  "significant": p_value < 0.05}
        lines += [
            "",
            f"  {'TD - LT':<12}{'':>8}{point:>10.2%}   95% CI [{low:+.2%}, {high:+.2%}]",
            f"  {shown} - {verdict}",
        ]

    return lines, summary


def breakdown(systems: dict, keep, label: str) -> list[str]:
    """Score every system over one subset of the utterances."""
    subsets = {name: [t for t in triples if keep(t)] for name, triples in systems.items()}
    if not next(iter(subsets.values())):
        return []

    lines = [label]
    for name, triples in subsets.items():
        result = score(triples)
        lines.append(
            f"  {name:<26}{result['utterances']:>8}"
            f"{result['wer']:>10.2%}{result['exact_match_rate']:>10.2%}"
        )
    return lines


def main(config: dict | None = None) -> None:
    config = config or load_config()

    mode = config["inference_modes"][0]
    if not predictions_path(config, mode).is_file():
        raise SystemExit(f"Missing predictions: {predictions_path(config, mode)}")

    systems = collect_systems(config)
    results = {name: score(pairs) for name, pairs in systems.items()}

    # The compositional floor needs every candidate, not one chosen string, so
    # it is scored separately and folded into the table afterwards.
    nbest = json.loads(config["nbest_path"].read_text(encoding="utf-8"))
    results["oracle_compositional"] = score_compositional_oracle(
        [(uid, truth, candidates(nbest, uid)) for uid, truth, _ in systems["whisper_1best"]]
    )

    metadata = {}
    if config["metadata_path"].is_file():
        metadata = json.loads(config["metadata_path"].read_text(encoding="utf-8"))

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

    significance_lines, significance_summary = significance(systems, metadata, config)
    lines += significance_lines

    # Late-talker vs typically-developing. This split is the reason the dataset
    # is worth running GenSEC over at all, so it gets its own table rather than
    # being averaged away into the corpus number.
    if metadata:
        groups = sorted({metadata.get(uid, {}).get("group", "") for uid, _, _ in systems["whisper_1best"]} - {""})
        if groups:
            lines.append("\n\nBy group (LT = late talker, TD = typically developing)")
            lines.append("-" * 72)
            lines.append(f"{'':<28}{'utts':>8}{'WER':>10}{'exact':>10}")
            for group in groups:
                lines += breakdown(
                    systems, lambda t, g=group: metadata.get(t[0], {}).get("group") == g, group
                )

    # Reference length. A one-word reference makes sentence WER unbounded, so a
    # handful of them can dominate a corpus number; this shows how much.
    lines.append("\n\nBy reference length")
    lines.append("-" * 72)
    lines.append(f"{'':<28}{'utts':>8}{'WER':>10}{'exact':>10}")
    for label, low, high in LENGTH_BUCKETS:
        lines += breakdown(
            systems,
            lambda t, lo=low, hi=high: lo <= len(normalize_for_scoring(t[1]).split()) <= hi,
            label,
        )

    # CHSER (Shankar et al., Interspeech 2025) drops references under 3 words
    # before scoring. This subset is the row that can be compared with theirs.
    lines.append("\n\nReferences of 3+ words only (comparable to published child GenSEC work)")
    lines.append("-" * 72)
    lines.append(f"{'':<28}{'utts':>8}{'WER':>10}{'exact':>10}")
    lines += breakdown(
        systems, lambda t: len(normalize_for_scoring(t[1]).split()) >= 3, "3+ words"
    )

    for name, result in results.items():
        if not result["worst"]:
            continue
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
    if significance_summary:
        metrics["significance"] = significance_summary
    (config["results_dir"] / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {config['results_dir'] / 'wer_report.txt'}")


if __name__ == "__main__":
    main()
