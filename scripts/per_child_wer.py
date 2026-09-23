"""
What this file is for:
One WER per child per system - the unit every test in run_anova.py requires.

High-level role in the pipeline:
Pipeline stage 6. Reads the predictions stage 4 already wrote and regroups
them by child; no retraining, no re-inference. Run it from run_analysis.py:

    python -u scripts/run_analysis.py

The statistics Dr. Xiong asked for compare children, not utterances. The
bootstrap in wer_report.txt resamples utterances, which treats 75 utterances
from one child as 75 independent observations; per-child is the conservative
unit and the one the ANOVA needs.

Writes only into results_dir/analysis/, so it cannot disturb wer_report.txt.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load_config, predictions_path
from evaluate import count_errors
from text import normalize_for_scoring

# A child with two test utterances produces a WER whose denominator is a
# handful of words - it moves by tens of points on a single misheard token.
# Five is a cheap floor; how many children it drops is reported rather than
# left implicit, because it is a choice a reviewer is entitled to question.
MIN_UTTERANCES = 5

# The two levels of the model factor: baseline Whisper against Whisper + LLM
# correction.
SYSTEMS = {"whisper": "one_best", "gensec": "prediction"}


def load_metadata(config: dict) -> dict:
    return json.loads(config["metadata_path"].read_text(encoding="utf-8"))


def accumulate(frame: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """Total errors and reference words per child, per system.

    Errors are accumulated and divided once at the end, rather than averaging
    per-utterance WERs. The mean of per-utterance rates weights a one-word
    utterance the same as a twenty-word one, which lets a child's shortest
    utterances decide their own score.
    """
    totals: dict[str, dict] = {}

    for row in frame.to_dict("records"):
        meta = metadata.get(row["id"])
        if not meta:
            continue

        child = (meta.get("child_id") or "").strip()
        group = (meta.get("group") or "").strip()
        if not child or not group:
            continue

        reference = normalize_for_scoring(row["truth"]).split()
        if not reference:
            continue

        entry = totals.setdefault(
            child,
            {
                "child_id": child,
                "group": group,
                # A child recorded across sessions can span corpora; keep the
                # set so the matched-subset analysis can filter on it later.
                "corpora": set(),
                "n_utterances": 0,
                "ref_words": 0,
                **{f"{name}_errors": 0 for name in SYSTEMS},
            },
        )

        entry["corpora"].add((meta.get("corpus") or "").strip())
        entry["n_utterances"] += 1
        entry["ref_words"] += len(reference)

        for name, column in SYSTEMS.items():
            hypothesis = normalize_for_scoring(row.get(column) or "").split()
            entry[f"{name}_errors"] += sum(count_errors(reference, hypothesis))

    rows = []
    for entry in totals.values():
        entry = dict(entry)
        entry["corpus"] = "|".join(sorted(c for c in entry.pop("corpora") if c))
        for name in SYSTEMS:
            entry[f"{name}_wer"] = entry[f"{name}_errors"] / entry["ref_words"]
        rows.append(entry)

    return pd.DataFrame(rows)


def per_child_wer(config: dict, mode: str = "zero_shot") -> pd.DataFrame:
    path = predictions_path(config, mode)
    if not path.is_file():
        raise SystemExit(f"Missing predictions: {path}")

    frame = pd.read_csv(path).fillna("")
    metadata = load_metadata(config)

    print(f"Predictions: {len(frame):,} test utterances")

    children = accumulate(frame, metadata)
    if children.empty:
        raise SystemExit("No child_id survived the join - check utterance_metadata.json")

    kept = children[children["n_utterances"] >= MIN_UTTERANCES].copy()
    dropped = len(children) - len(kept)
    print(
        f"Children: {len(children)} total, {dropped} dropped below "
        f"{MIN_UTTERANCES} utterances, {len(kept)} kept"
    )

    # The improvement score is what the interaction test permutes, so it is
    # computed once here rather than re-derived in run_anova.py.
    kept["improvement"] = kept["whisper_wer"] - kept["gensec_wer"]
    kept = kept.sort_values(["group", "child_id"]).reset_index(drop=True)

    for group, rows in kept.groupby("group"):
        print(
            f"  {group}: {len(rows)} children, "
            f"median {rows['n_utterances'].median():.0f} utterances each"
        )

    return kept[
        [
            "child_id",
            "group",
            "corpus",
            "n_utterances",
            "ref_words",
            "whisper_wer",
            "gensec_wer",
            "improvement",
        ]
    ]


def reconcile(children: pd.DataFrame) -> None:
    """Check the regrouping against the group WERs already in wer_report.txt.

    Per-child WERs weighted by reference words must reproduce the corpus-level
    group WER, because both are the same errors over the same words. They will
    not match exactly - the utterance floor drops a few children - but a gap of
    more than a point or so means the join or the normalization is wrong, and
    that is worth catching before any of it reaches a p-value.
    """
    print("\nReconciliation against wer_report.txt (word-weighted)")
    for group, rows in children.groupby("group"):
        words = rows["ref_words"].sum()
        for name in SYSTEMS:
            weighted = (rows[f"{name}_wer"] * rows["ref_words"]).sum() / words
            print(f"  {group:<3} {name:<8} {weighted:.2%}")


def main() -> None:
    config = load_config()
    children = per_child_wer(config)
    reconcile(children)

    output_dir = config["results_dir"] / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "per_child_wer.csv"
    children.to_csv(output_path, index=False)

    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
