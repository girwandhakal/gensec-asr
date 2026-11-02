"""Tests for hypothesis_length_cap and inference signature tracking."""

import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load_function(path, name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class DummyTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


class HypothesisCapTests(unittest.TestCase):
    def setUp(self):
        self.hypothesis_length_cap = load_function(
            ROOT / "scripts/train.py", "hypothesis_length_cap", {}
        )
        self.tokenizer = DummyTokenizer()
        self.config = {
            "generation_length_slack": 1.5,
            "generation_token_headroom": 4,
            "max_target_length": 128,
        }

    def test_single_token_input(self):
        # 1 token * 1.5 + 4 = int(1.5) + 4 = 5 tokens
        batch = [{"input": ["you"]}]
        cap = self.hypothesis_length_cap(batch, self.tokenizer, self.config)
        self.assertEqual(cap, 5)

    def test_multi_hypothesis_picks_longest(self):
        # Longest is 4 tokens ("i want juice please") -> int(4 * 1.5) + 4 = 6 + 4 = 10
        batch = [{"input": ["juice", "i want juice please", "want juice"]}]
        cap = self.hypothesis_length_cap(batch, self.tokenizer, self.config)
        self.assertEqual(cap, 10)

    def test_empty_hypothesis_falls_back_to_headroom(self):
        batch = [{"input": [""]}]
        cap = self.hypothesis_length_cap(batch, self.tokenizer, self.config)
        self.assertEqual(cap, 4)

    def test_cap_does_not_exceed_max_target_length(self):
        batch = [{"input": ["word " * 100]}]  # 100 tokens -> 100 * 1.5 + 4 = 154
        cap = self.hypothesis_length_cap(batch, self.tokenizer, self.config)
        self.assertEqual(cap, 128)

    def test_signature_keys_include_hypothesis_cap_settings(self):
        tree = ast.parse((ROOT / "scripts/train.py").read_text(encoding="utf-8"))
        assign = next(
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                target.id == "INFERENCE_SIGNATURE_KEYS"
                for target in node.targets
                if isinstance(target, ast.Name)
            )
        )
        keys = [elt.value for elt in assign.value.elts]
        self.assertIn("generation_length_slack", keys)
        self.assertIn("generation_token_headroom", keys)


if __name__ == "__main__":
    unittest.main()
