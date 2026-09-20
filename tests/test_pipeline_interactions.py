import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ace.ace import ACE


class PipelineInteractionTests(unittest.TestCase):
    def test_failure_memories_are_source_isolated_without_changing_top_k(self):
        real_memory = MagicMock(name="real_memory")
        adversarial_memory = MagicMock(name="adversarial_memory")

        with patch(
            "ace.ace.initialize_clients", return_value=(None, None, None)
        ), patch(
            "ace.ace.FailureMemoryBank",
            side_effect=[real_memory, adversarial_memory],
        ) as memory_cls:
            ace = ACE(
                api_provider="fake",
                generator_model="generator",
                reflector_model="reflector",
                curator_model="curator",
                use_failure_memory=True,
                failure_memory_top_k=10,
                failure_memory_mode="verified",
            )

        self.assertIs(ace._get_real_failure_memory(), real_memory)
        self.assertIs(ace._get_adversarial_failure_memory(), adversarial_memory)
        self.assertIs(ace.failure_memory, real_memory)
        self.assertEqual(memory_cls.call_count, 2)
        self.assertEqual(memory_cls.call_args_list[0].kwargs["top_k"], 10)
        self.assertEqual(memory_cls.call_args_list[1].kwargs["top_k"], 10)

    def test_correct_sample_updates_counters_but_does_not_call_curator(self):
        ace = ACE.__new__(ACE)
        ace.playbook = (
            "## CALCULATIONS\n"
            "[calc-00001] helpful=0 harmful=0 :: Apply the relevant formula."
        )
        ace.next_global_id = 2
        ace.playbook_retriever = None
        ace.real_failure_memory = None
        ace.adversarial_failure_memory = None
        ace.failure_memory = None
        ace.generator = MagicMock()
        ace.generator.generate.return_value = (
            '{"reasoning":"ok","bullet_ids":["calc-00001"],"final_answer":"42"}',
            ["calc-00001"],
            {},
        )
        ace.reflector = MagicMock()
        ace.reflector.reflect.return_value = (
            '{"bullet_tags":[{"id":"calc-00001","tag":"helpful"}]}',
            [{"id": "calc-00001", "tag": "helpful"}],
            {},
        )
        ace.curator = MagicMock()
        ace.use_bulletpoint_analyzer = False
        ace.bulletpoint_analyzer = None
        ace.use_rae = False
        ace._run_adversarial_episode = MagicMock(return_value=None)

        processor = MagicMock()
        processor.answer_is_correct.return_value = True

        with tempfile.TemporaryDirectory() as directory:
            ace._train_single_sample(
                task_dict={"question": "q", "context": "ctx", "target": "42"},
                data_processor=processor,
                step_id="train_e_1_s_1",
                epoch=1,
                step=1,
                usage_log_path=str(Path(directory) / "usage.jsonl"),
                log_dir=directory,
                config_params={
                    "max_num_rounds": 3,
                    "curator_frequency": 1,
                    "token_budget": 4000,
                    "use_json_mode": False,
                    "no_ground_truth": False,
                },
                total_samples=1,
            )

        ace.curator.curate.assert_not_called()
        self.assertIn("[calc-00001] helpful=1 harmful=0", ace.playbook)

    def test_adversarial_failure_uses_exactly_one_update_path(self):
        for curator_operations, expected_memory_writes in [
            ([{"type": "ADD", "content": "fix"}], 0),
            ([], 1),
        ]:
            with self.subTest(curator_operations=curator_operations):
                ace = ACE.__new__(ACE)
                ace.use_adversarial = True
                ace.playbook = "## CALCULATIONS"
                ace.next_global_id = 1
                ace.playbook_retriever = None
                ace.adversarial_agent = MagicMock()
                ace.adversarial_agent.generate_attack.return_value = (
                    {
                        "question": "attack",
                        "context": "ctx",
                        "target": "right",
                        "candidate_id": "c1",
                        "vulnerability_id": "v1",
                        "attack_category": "edge",
                        "verifier_confidence": 1.0,
                        "selection_score": 0.9,
                    },
                    {"pipeline": "mine-generate-verify-select"},
                )
                ace.generator = MagicMock()
                ace.generator.generate.return_value = (
                    '{"reasoning":"bad","bullet_ids":[],"final_answer":"wrong"}',
                    [],
                    {},
                )
                ace.reflector = MagicMock()
                ace.reflector.reflect.return_value = (
                    '{"error_identification":"e","root_cause_analysis":"r",'
                    '"key_insight":"k","bullet_tags":[]}',
                    [],
                    {},
                )
                ace.curator = MagicMock()
                ace.curator.curate.return_value = (
                    ace.playbook,
                    ace.next_global_id,
                    curator_operations,
                    {},
                )
                ace.curator_allowed_operations = ["ADD"]
                ace.delete_harmful_margin = 4
                ace.delete_min_harmful = 3
                ace._get_curator_merge_candidates = MagicMock(return_value=[])
                ace.use_bulletpoint_analyzer = False
                ace.bulletpoint_analyzer = None
                ace.use_rae = False

                real_memory = MagicMock(name="real_memory")
                adversarial_memory = MagicMock(name="adversarial_memory")
                adversarial_memory.mode = "verified"
                ace.real_failure_memory = real_memory
                ace.adversarial_failure_memory = adversarial_memory
                ace.failure_memory = real_memory

                processor = MagicMock()
                processor.answer_is_correct.return_value = False

                with tempfile.TemporaryDirectory() as directory:
                    result = ace._run_adversarial_episode(
                        step_id="train_e_1_s_10",
                        epoch=1,
                        step=10,
                        usage_log_path=str(Path(directory) / "usage.jsonl"),
                        log_dir=directory,
                        config_params={
                            "adversarial_frequency": 10,
                            "use_json_mode": False,
                            "no_ground_truth": False,
                            "token_budget": 4000,
                            "task_name": "formula",
                        },
                        total_samples=100,
                        base_question="base",
                        base_context="base context",
                        base_target="base target",
                        data_processor=processor,
                    )

                self.assertIsNotNone(result)
                self.assertIs(
                    ace.reflector.reflect.call_args.kwargs["failure_memory"],
                    adversarial_memory,
                )
                self.assertEqual(
                    adversarial_memory.add_verified.call_count,
                    expected_memory_writes,
                )
                real_memory.add_verified.assert_not_called()


if __name__ == "__main__":
    unittest.main()
