import importlib.util
import sys
import types
import unittest
from pathlib import Path


utils_stub = types.ModuleType("utils")
utils_stub.get_section_slug = lambda section: section
sys.modules.setdefault("utils", utils_stub)

MODULE_PATH = Path(__file__).resolve().parents[1] / "playbook_utils.py"
SPEC = importlib.util.spec_from_file_location("playbook_utils_under_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class UnusedBulletPruningTests(unittest.TestCase):
    def test_prunes_only_bullets_without_any_evidence(self):
        playbook = """## FORMULAS
[calc-00001] helpful=2 harmful=0 :: Keep this rule.
[calc-00002] helpful=0 harmful=0 :: Remove this rule.

## STRATEGIES
[strat-00003] helpful=0 harmful=1 :: Keep this audited rule."""

        pruned_playbook, pruned_ids = MODULE.prune_zero_evidence_bullets(playbook)

        self.assertEqual(pruned_ids, ["calc-00002"])
        self.assertIn("[calc-00001] helpful=2 harmful=0", pruned_playbook)
        self.assertIn("[strat-00003] helpful=0 harmful=1", pruned_playbook)
        self.assertNotIn("calc-00002", pruned_playbook)
        self.assertIn("## STRATEGIES", pruned_playbook)


if __name__ == "__main__":
    unittest.main()
