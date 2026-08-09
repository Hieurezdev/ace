import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import os

from ace.ace import ACE


class OfflineResumeRunPathTests(unittest.TestCase):
    def test_offline_train_writes_current_playbook_when_resuming(self):
        ace = ACE.__new__(ACE)
        ace.playbook = "initial playbook"
        ace.max_tokens = 128
        ace.failure_memory = None

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "resume_run"
            save_path.mkdir()
            playbook_dir = save_path / "intermediate_playbooks"
            playbook_dir.mkdir()
            log_dir = save_path / "detailed_llm_logs"
            log_dir.mkdir()

            result = ace._offline_train(
                train_samples=[],
                val_samples=[],
                data_processor=object(),
                config={
                    "task_name": "task",
                    "num_epochs": 1,
                    "eval_steps": 1,
                    "save_steps": 1,
                    "test_workers": 1,
                    "json_mode": False,
                    "curator_frequency": 1,
                },
                save_path=str(save_path),
                usage_log_path=str(save_path / "bullet_usage_log.jsonl"),
                playbook_dir=str(playbook_dir),
                log_dir=str(log_dir),
                resume_run_path=str(save_path),
            )

            self.assertEqual(result["best_validation_accuracy"], 0.0)
            current_playbook_path = save_path / "current_playbook.txt"
            self.assertTrue(current_playbook_path.exists())
            self.assertEqual(current_playbook_path.read_text(), "initial playbook")

    def test_run_forwards_resume_run_path_into_offline_train(self):
        ace = ACE.__new__(ACE)
        ace.playbook = "initial playbook"
        ace.best_playbook = "initial playbook"
        ace.generator = SimpleNamespace(model="generator")
        ace.reflector = SimpleNamespace(model="reflector")
        ace.curator = SimpleNamespace(model="curator")
        ace.adversarial_agent = None
        ace.failure_memory = None

        captured_kwargs = {}

        def fake_setup_paths(save_dir, task_name, mode, resume_run_path=None):
            self.assertEqual(save_dir, "./results")
            self.assertEqual(task_name, "task")
            self.assertEqual(mode, "offline")
            self.assertEqual(resume_run_path, "/tmp/resume-target")
            os.makedirs("/tmp/resume-target", exist_ok=True)
            return (
                "/tmp/resume-target",
                "/tmp/resume-target/bullet_usage_log.jsonl",
                "/tmp/resume-target/intermediate_playbooks",
                "/tmp/resume-target/detailed_llm_logs",
            )

        def fake_offline_train(**kwargs):
            captured_kwargs.update(kwargs)
            return {"best_validation_accuracy": 0.0}

        ace._setup_paths = fake_setup_paths
        ace._offline_train = fake_offline_train

        result = ace.run(
            mode="offline",
            train_samples=[],
            val_samples=[],
            data_processor=object(),
            config={
                "task_name": "task",
                "save_dir": "./results",
                "resume_run_path": "/tmp/resume-target",
            },
        )

        self.assertEqual(result["training_results"]["best_validation_accuracy"], 0.0)
        self.assertEqual(captured_kwargs["resume_run_path"], "/tmp/resume-target")


if __name__ == "__main__":
    unittest.main()