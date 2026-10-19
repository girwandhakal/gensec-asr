"""Check child isolation without loading models or running training."""

import ast
from pathlib import Path
import random
import tempfile
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
tree = ast.parse((ROOT / "scripts/train.py").read_text(encoding="utf-8"))
function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "make_splits")
namespace = {"pd": pd, "random": random, "transcript_id": lambda uid: uid.split("_")[0]}
exec(compile(ast.Module(body=[function], type_ignores=[]), "child_splits", "exec"), namespace)
make_splits = namespace["make_splits"]


class ChildSplitTests(unittest.TestCase):
    def setUp(self):
        self.metadata = {}
        for group in ["LT", "TD"]:
            for child in range(12):
                for recording in range(3):
                    for turn in range(4):
                        uid = f"{group}{child}t{recording}_0_100_{turn}"
                        self.metadata[uid] = {"child_id": f"X::{group}{child}", "group": group}
        self.examples = pd.DataFrame({"id": list(self.metadata)})
        self.config = {"test_size": 0.2, "seed": 42}

    def children(self, frame):
        return {self.metadata[uid]["child_id"] for uid in frame.id}

    def test_all_recordings_stay_with_child_across_three_partitions(self):
        pool, test = make_splits(self.examples, self.config, self.metadata, save=False)
        train, validation = make_splits(pool, self.config, self.metadata, save=False)
        parts = [train, validation, test]
        for i, left in enumerate(parts):
            self.assertFalse(left.empty)
            self.assertEqual({self.metadata[uid]["group"] for uid in left.id}, {"LT", "TD"})
            for right in parts[i + 1:]:
                self.assertTrue(self.children(left).isdisjoint(self.children(right)))
        self.assertEqual(set().union(*(set(part.id) for part in parts)), set(self.examples.id))
        self.assertEqual(sum(len(part) for part in parts), len(self.examples))

    def test_selection_does_not_depend_on_row_order(self):
        first = make_splits(self.examples, self.config, self.metadata, save=False)
        shuffled = make_splits(self.examples.sample(frac=1, random_state=7), self.config, self.metadata, save=False)
        for left, right in zip(first, shuffled):
            self.assertEqual(set(left.id), set(right.id))

    def test_validation_does_not_overwrite_saved_train_test_splits(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "data") as directory:
            config = {**self.config, "splits_dir": Path(directory)}
            pool, _ = make_splits(self.examples, config, self.metadata)
            saved = {p.name: p.read_bytes() for p in Path(directory).glob("*.csv")}
            make_splits(pool, config, self.metadata, save=False)
            self.assertEqual(saved, {p.name: p.read_bytes() for p in Path(directory).glob("*.csv")})

    def test_missing_child_id_is_rejected(self):
        self.metadata[self.examples.iloc[0].id]["child_id"] = ""
        with self.assertRaisesRegex(ValueError, "Missing child ID"):
            make_splits(self.examples, self.config, self.metadata, save=False)


if __name__ == "__main__":
    unittest.main()
