"""
What this file is for:
The group x model statistics Dr. Xiong asked for, on per-child WER.

High-level role in the pipeline:
Pipeline stage 6. Reads analysis/per_child_wer.csv and writes
analysis/anova_report.txt and anova_metrics.json. Run it from run_analysis.py.

The mixed model describes the factorial design. The interaction is also
checked on child-level improvement scores, with the assumptions of each test
reported alongside its result.

A note on AnovaRM, so nobody spends an afternoon on it: the design is mixed -
model is within-child, group is between-child - and statsmodels cannot fit
that. AnovaRM raises "Between subject effect not yet supported!" the moment a
between-subject factor is passed. A mixed-effects model with a random
intercept per child and child-level tests of improvement run instead.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load_config

PERMUTATIONS = 10_000
STUDENTIZED_PERMUTATIONS = 20_000
SEED = 20260908
ALPHA = 0.05


def load_children(config: dict) -> pd.DataFrame:
    path = config["results_dir"] / "analysis" / "per_child_wer.csv"
    if not path.is_file():
        raise SystemExit(f"Missing {path} - run per_child_wer.py first")

    children = pd.read_csv(path)
    groups = sorted(children["group"].unique())
    if len(groups) != 2:
        raise SystemExit(f"Expected two groups, found {groups}")

    return children


def to_long(children: pd.DataFrame) -> pd.DataFrame:
    """One row per child per model - the format the mixed model needs."""
    long = children.melt(
        id_vars=["child_id", "group", "corpus", "n_utterances", "ref_words"],
        value_vars=["whisper_wer", "gensec_wer"],
        var_name="model",
        value_name="wer",
    )
    long["model"] = long["model"].str.replace("_wer", "", regex=False)
    return long


def cell_means(children: pd.DataFrame) -> list[dict]:
    rows = []
    for group, group_rows in children.groupby("group"):
        rows.append(
            {
                "group": group,
                "n": len(group_rows),
                "whisper": group_rows["whisper_wer"].mean(),
                "gensec": group_rows["gensec_wer"].mean(),
                "improvement": group_rows["improvement"].mean(),
            }
        )
    return rows


def mixed_model(long: pd.DataFrame) -> dict:
    """Random intercept per child; fixed effects for group, model, interaction.

    This is a joint model for the unbalanced mixed design. Its coefficient
    p-values are model-based Wald tests.
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        raise SystemExit("statsmodels is required: pip install statsmodels")

    frame = long.copy()
    # Treatment coding against the baseline cell, so each coefficient reads as
    # a difference from baseline Whisper on the reference group.
    frame["group"] = pd.Categorical(frame["group"])
    frame["model"] = pd.Categorical(frame["model"], categories=["whisper", "gensec"])

    fitted = smf.mixedlm(
        "wer ~ group * model", frame, groups=frame["child_id"]
    ).fit(reml=False)

    terms = {}
    for name in fitted.params.index:
        if name == "Intercept" or name.startswith("Group Var"):
            continue
        terms[name] = {
            "coefficient": float(fitted.params[name]),
            "p": float(fitted.pvalues[name]),
        }

    return {"terms": terms, "summary": str(fitted.summary())}


def permutation_interaction(children: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Compare mean improvement by permuting child-level group labels.

    The interaction in this design is exactly "is the per-child improvement
    larger in one group than the other", so it can be tested directly by
    shuffling the group labels on the improvement scores. The unstudentized
    test requires exchangeability, which unequal group variances can violate.
    """
    groups = sorted(children["group"].unique())
    first = children.loc[children["group"] == groups[0], "improvement"].to_numpy()
    second = children.loc[children["group"] == groups[1], "improvement"].to_numpy()

    observed = second.mean() - first.mean()
    pooled = np.concatenate([first, second])
    cut = len(first)

    extreme = 0
    for _ in range(PERMUTATIONS):
        shuffled = rng.permutation(pooled)
        if abs(shuffled[cut:].mean() - shuffled[:cut].mean()) >= abs(observed):
            extreme += 1

    # +1 in both places: a permutation p of exactly zero would claim more
    # certainty than PERMUTATIONS draws can support.
    return {
        "difference": float(observed),
        "p": (extreme + 1) / (PERMUTATIONS + 1),
        "groups": groups,
    }


def studentized_permutation_interaction(children: pd.DataFrame) -> dict:
    """Permutation check using a Welch statistic under unequal variances.

    This is asymptotically valid for equality of means under heterogeneous
    variances; it is not an exact finite-sample test here.
    """
    groups = sorted(children["group"].unique())
    first = children.loc[children["group"] == groups[0], "improvement"].to_numpy()
    second = children.loc[children["group"] == groups[1], "improvement"].to_numpy()

    def statistic(a: np.ndarray, b: np.ndarray) -> float:
        return float((b.mean() - a.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))

    observed = statistic(first, second)
    pooled = np.concatenate([first, second])
    rng = np.random.default_rng(20260924)
    extreme = 0
    for _ in range(STUDENTIZED_PERMUTATIONS):
        shuffled = rng.permutation(pooled)
        extreme += abs(statistic(shuffled[:len(first)], shuffled[len(first):])) >= abs(observed)

    return {"statistic": observed, "p": (extreme + 1) / (STUDENTIZED_PERMUTATIONS + 1)}


def disagreement_check(children: pd.DataFrame) -> dict:
    """Show sensitivity of the interaction to different test assumptions."""
    groups = sorted(children["group"].unique())
    first = children.loc[children["group"] == groups[0], "improvement"].to_numpy()
    second = children.loc[children["group"] == groups[1], "improvement"].to_numpy()

    tests = {
        "student_t": stats.ttest_ind(first, second, equal_var=True).pvalue,
        "welch_t": stats.ttest_ind(first, second, equal_var=False).pvalue,
        "mann_whitney": stats.mannwhitneyu(first, second).pvalue,
        "yuen_trimmed": stats.ttest_ind(first, second, equal_var=False, trim=0.2).pvalue,
    }
    tests = {name: float(p) for name, p in tests.items()}

    normality = {}
    for group, values in ((groups[0], first), (groups[1], second)):
        # Shapiro is only meaningful above a handful of observations.
        if len(values) >= 3:
            normality[group] = float(stats.shapiro(values).pvalue)

    split = min(tests.values()) < ALPHA <= max(tests.values())

    return {
        "tests": tests,
        "shapiro": normality,
        "sd": {groups[0]: float(first.std(ddof=1)), groups[1]: float(second.std(ddof=1))},
        "disagree": bool(split),
    }


def simple_effects(children: pd.DataFrame) -> dict:
    """Did correction help within each group? Holm-corrected across groups.

    She asked for these as post-hoc tests conditional on a significant
    interaction. They are computed either way - they are the within-group
    comparison she named separately - but the report says plainly whether the
    interaction licensed them.
    """
    results = {}
    for group, rows in children.groupby("group"):
        whisper = rows["whisper_wer"].to_numpy()
        gensec = rows["gensec_wer"].to_numpy()
        test = stats.ttest_rel(whisper, gensec)
        differences = whisper - gensec
        # Cohen's dz: the paired effect size, mean difference over its own SD.
        spread = differences.std(ddof=1)
        results[group] = {
            "n": int(len(rows)),
            "p_uncorrected": float(test.pvalue),
            "dz": float(differences.mean() / spread) if spread else float("nan"),
        }

    # Holm: sort ascending, multiply by the number of tests remaining, and
    # enforce monotonicity so a later p can never fall below an earlier one.
    ordered = sorted(results, key=lambda g: results[g]["p_uncorrected"])
    running = 0.0
    for position, group in enumerate(ordered):
        adjusted = results[group]["p_uncorrected"] * (len(ordered) - position)
        running = max(running, min(adjusted, 1.0))
        results[group]["p_holm"] = running

    return results


def render(children: pd.DataFrame, results: dict) -> str:
    lines = [
        "Group x model statistics on per-child WER",
        "=" * 72,
        "",
        f"Children: {len(children)} "
        + ", ".join(f"{row['group']} {row['n']}" for row in results["cell_means"]),
        f"Retained utterances: {int(children['n_utterances'].sum()):,}",
        "",
        "Per-child cell means",
        "-" * 72,
        f"{'group':<8}{'n':>5}{'whisper':>12}{'gensec':>12}{'improvement':>14}",
    ]
    for row in results["cell_means"]:
        lines.append(
            f"{row['group']:<8}{row['n']:>5}{row['whisper']:>11.2%}"
            f"{row['gensec']:>12.2%}{row['improvement'] * 100:>13.2f}pp"
        )

    lines += ["", "Corpus composition (children)", "-" * 72]
    for (corpus, group), count in children.groupby(["corpus", "group"]).size().items():
        lines.append(f"  {corpus:<20} {group:<3} {count:>3}")
    lines.append("  Group comparisons may reflect differences in corpus composition.")

    lines += ["", "Mixed-effects model (random intercept per child)", "-" * 72]
    for name, term in results["mixed_model"]["terms"].items():
        flag = "significant" if term["p"] < ALPHA else "not significant"
        lines.append(f"  {name:<40} b={term['coefficient']:+.4f}  p={term['p']:.4f}  {flag}")
    lines.append("  group coefficient: TD-LT difference under Whisper only.")
    lines.append("  model coefficient: GenSEC-Whisper difference within LT only.")

    interaction = results["permutation"]
    studentized = results["studentized_permutation"]
    lines += [
        "",
        "Interaction: does improvement differ by group?",
        "-" * 72,
        f"  label permutation ({PERMUTATIONS:,} resamples)  p = {interaction['p']:.4f}",
        f"  Welch-statistic permutation ({STUDENTIZED_PERMUTATIONS:,} resamples)"
        f"  p = {studentized['p']:.4f}",
        f"  difference in mean improvement: {interaction['difference'] * 100:+.2f}pp "
        f"({interaction['groups'][1]} - {interaction['groups'][0]})",
        "",
        "  Other tests, for comparison:",
    ]
    for name, p in results["disagreement"]["tests"].items():
        mark = "  <- under .05" if p < ALPHA else ""
        lines.append(f"    {name:<16} p = {p:.4f}{mark}")

    for group, p in results["disagreement"]["shapiro"].items():
        lines.append(f"    Shapiro {group:<8} p = {p:.3g}")

    for group, sd in results["disagreement"]["sd"].items():
        lines.append(f"    improvement SD {group:<8} = {sd:.4f}")

    if results["disagreement"]["disagree"]:
        lines += [
            "",
            "  WARNING: these tests disagree across alpha = .05.",
            "  The groups have unequal variances and unequal sample sizes.",
            "  The label permutation assumes exchangeable improvement scores;",
            "  that assumption is doubtful here. Welch tests equality of means",
            "  under unequal variances, but is sensitive to large TD values.",
            "  No single p-value resolves this discrepancy. Treat the interaction",
            "  as sensitive to the analysis method, not confirmed or ruled out.",
        ]

    lines += ["", "Simple effects: did correction help within each group?", "-" * 72]
    for group, effect in results["simple_effects"].items():
        verdict = "significant" if effect["p_holm"] < ALPHA else "not significant"
        lines.append(
            f"  {group:<8} n={effect['n']:<5} p(Holm)={effect['p_holm']:.3g}  "
            f"dz={effect['dz']:.2f}  {verdict}"
        )

    lines.append("  These paired tests do not test whether the group gains differ.")

    return "\n".join(lines) + "\n"


def main() -> None:
    config = load_config()
    children = load_children(config)
    rng = np.random.default_rng(SEED)

    results = {
        "cell_means": cell_means(children),
        "mixed_model": mixed_model(to_long(children)),
        "permutation": permutation_interaction(children, rng),
        "studentized_permutation": studentized_permutation_interaction(children),
        "disagreement": disagreement_check(children),
        "simple_effects": simple_effects(children),
    }

    report = render(children, results)
    print(report)

    output_dir = config["results_dir"] / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "anova_report.txt").write_text(report, encoding="utf-8")

    # The full statsmodels summary is useful in the report but noise in JSON.
    metrics = dict(results)
    metrics["mixed_model"] = {"terms": results["mixed_model"]["terms"]}
    (output_dir / "anova_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print(f"Wrote {output_dir / 'anova_report.txt'}")
    print(f"Wrote {output_dir / 'anova_metrics.json'}")


if __name__ == "__main__":
    main()
