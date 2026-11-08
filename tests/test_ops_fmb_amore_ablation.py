"""Execute ablation dry-runs and check feature isolation without model calls."""
import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FEATURES = {"ops_fmb": (True, True, False), "ops_amore": (True, False, True),
            "fmb_amore": (False, True, True), "all": (True, True, True)}


class ScriptTests(unittest.TestCase):
    def dry_run(self, variants="ops_fmb ops_amore fmb_amore all"):
        with tempfile.TemporaryDirectory() as output:
            result = subprocess.run(["bash", str(ROOT / "run_ops_fmb_amore_ablation.sh")],
                env={**os.environ, "DRY_RUN": "1", "VARIANTS": variants,
                     "MAX_TRAIN_SAMPLES": "5", "OUTPUT_ROOT": output,
                     "APPWORLD_PROJECT_PATH": str(ROOT)},
                capture_output=True, text=True, timeout=15)
            self.assertEqual(list(Path(output).iterdir()), [])
        return result

    def test_matrix_commands_select_the_requested_components(self):
        result = self.dry_run()
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [shlex.split(line) for line in result.stdout.splitlines() if line.startswith("uv ")]
        self.assertEqual(len(commands), 8)
        paths = set()
        memories = set()
        for command in commands:
            path = command[command.index("--save_path") + 1]
            variant = Path(path).parent.name
            ops, fmb, amore = FEATURES[variant]
            self.assertEqual("--use_lifecycle_curator" in command, ops)
            self.assertEqual("--use_dbscan_merge_candidates" in command, ops)
            self.assertEqual("--use_verified_failure_memory" in command, fmb)
            self.assertEqual("--use_amore" in command, amore)
            self.assertNotIn("--use_failure_memory", command)
            self.assertNotIn("--use_budgeted_curator", command)
            self.assertNotIn("--use_adversarial", command)
            self.assertNotIn("--use_rae", command)
            self.assertEqual(command[command.index("--playbook_token_budget") + 1], "8000")
            self.assertEqual(command[command.index("--max_train_samples") + 1], "5")
            paths.add(path)
            if fmb:
                self.assertEqual(command[command.index("--failure_memory_top_k") + 1], "10")
            if amore:
                self.assertEqual(command[command.index("--amore_top_k") + 1], "4")
                memories.add(command[command.index("--amore_memory_path") + 1])
        self.assertEqual(len(paths), 8)
        self.assertEqual(len(memories), 6)

    def test_can_select_only_the_full_combination(self):
        result = self.dry_run("all")
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [line for line in result.stdout.splitlines() if line.startswith("uv ")]
        self.assertEqual(len(commands), 2)

    def test_invalid_variant_stops_before_any_run(self):
        result = self.dry_run("ops_fmb typo")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("uv run", result.stdout)


if __name__ == "__main__":
    unittest.main()
