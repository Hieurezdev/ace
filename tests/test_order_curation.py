import importlib.util
import itertools
import sys
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("order_curation", Path(__file__).resolve().parents[1] / "ace/core/order_curation.py")
order = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = order
spec.loader.exec_module(order)


class OrderCurationTests(unittest.TestCase):
    def test_merge_preserves_update_independently_of_arrival(self):
        update = {"type": "UPDATE", "target_id": "calc-1", "content": "Include t=0 once."}
        merge = {"type": "MERGE", "source_ids": ["calc-1", "calc-2"], "content": "Convert rate first."}
        unrelated = {"type": "ADD", "section": "formulas", "content": "Check precision."}
        plans = [order.plan_batch(list(items), resolve_overlaps=True)
                 for items in itertools.permutations([update, merge, unrelated])]
        self.assertTrue(all(plan == plans[0] for plan in plans))
        self.assertEqual(plans[0].reconciled_groups, 1)
        edits = [op["content"] for op in plans[0].operations if op["type"] == "MERGE"]
        self.assertEqual(len(edits), 1)
        self.assertIn(update["content"], edits[0])
        self.assertIn(merge["content"], edits[0])

    def test_overlapping_update_and_delete_are_held(self):
        update = {"type": "UPDATE", "bullet_id": "calc-1", "content": "Use simple interest."}
        delete = {"type": "DELETE", "bullet_id": "calc-1", "reason": "Overgeneralized."}
        self.assertEqual(order.plan_batch([update, delete], resolve_overlaps=True).operations, [])
        self.assertEqual(len(order.plan_batch([update, delete], resolve_overlaps=False).operations), 2)

    def test_replay_tracks_repairs_and_regressions_separately(self):
        self.assertEqual(order.replay_counts([False, True, False], [True, False, False]), (1, 1))


if __name__ == "__main__":
    unittest.main()
