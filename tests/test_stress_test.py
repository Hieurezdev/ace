import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).parents[1] / "ace" / "core" / "stress_test.py"
SPEC = importlib.util.spec_from_file_location("stress_test_under_test", MODULE_PATH)
stress_test_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(stress_test_module)


PLAYBOOK = """## FORMULAS & CALCULATIONS
[formula-00001] helpful=1 harmful=0 :: Convert the discount rate to decimal.
[formula-00002] helpful=0 harmful=0 :: Sum the discounted cash flows.

## OTHERS
[others-00003] helpful=0 harmful=0 :: Check units before answering."""


class StressTestTests(unittest.TestCase):
    def test_replace_is_deterministic_and_does_not_change_bullet_count(self):
        corrupted, manifest = stress_test_module.corrupt_playbook(
            PLAYBOOK, noise_rate=0.5, mode="replace", seed=7
        )
        self.assertEqual(manifest["harmful_bullets"], 1)
        self.assertEqual(manifest["original_bullets"], 3)
        self.assertEqual(corrupted.count("helpful="), PLAYBOOK.count("helpful="))
        self.assertIn("Stress-test rule:", corrupted)
        self.assertNotIn("Stress-test rule:", PLAYBOOK)

    def test_append_preserves_source_bullets_and_adds_noise(self):
        corrupted, manifest = stress_test_module.corrupt_playbook(
            PLAYBOOK, noise_rate=1.0, mode="append", seed=42
        )
        self.assertEqual(manifest["harmful_bullets"], 3)
        self.assertEqual(corrupted.count("helpful="), 6)
        self.assertIn("[stress-00004]", corrupted)


if __name__ == "__main__":
    unittest.main()
