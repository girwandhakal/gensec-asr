"""Check corpus matching without fitting a costly mixed model."""

import ast
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TREE = ast.parse((ROOT / "scripts/run_anova.py").read_text(encoding="utf-8"))
FUNCTIONS = [node for node in TREE.body if isinstance(node, ast.FunctionDef)
             and node.name in {"to_long", "cell_means", "corpus_matched_analysis"}]
NS = {"pd": pd, "np": np, "SHARED_CORPORA": ("ENNI", "EllisWeismer", "Rescorla"),
      "SEED": 42, "PERMUTATIONS": 500,
      "mixed_model": lambda frame, adjust_corpus: {"terms": {}, "summary": "fixture"}}
exec(compile(ast.Module(body=FUNCTIONS, type_ignores=[]), "anova", "exec"), NS)


class CorpusMatchedTests(unittest.TestCase):
    def test_unshared_corpus_is_excluded_and_labels_stay_within_corpus(self):
        rows = []
        for corpus in (*NS["SHARED_CORPORA"], "Other"):
            groups = ("LT", "TD") if corpus != "Other" else ("TD",)
            for group in groups:
                for index in range(4):
                    improvement = 0.2 if group == "LT" else 0.0
                    rows.append(dict(child_id=f"{corpus}-{group}-{index}", group=group,
                                     corpus=corpus, n_utterances=10, ref_words=40,
                                     whisper_wer=0.5, gensec_wer=0.5-improvement,
                                     improvement=improvement))
        result = NS["corpus_matched_analysis"](pd.DataFrame(rows))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["children"], 24)
        self.assertEqual(set(result["composition"]), set(NS["SHARED_CORPORA"]))
        self.assertAlmostEqual(result["within_corpus_permutation"]["difference"], -0.2)
        self.assertLess(result["within_corpus_permutation"]["p"], 0.05)
        missing = pd.DataFrame(rows)
        missing = missing[missing["corpus"] != "Rescorla"]
        self.assertEqual(NS["corpus_matched_analysis"](missing)["status"],
                         "insufficient_children_in_shared_corpora")


if __name__ == "__main__":
    unittest.main()
