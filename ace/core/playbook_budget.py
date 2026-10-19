"""Transactional, write-time token limits for a single canonical Playbook."""

import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal, TypedDict

from playbook_utils import apply_curator_operations, parse_playbook_line


class MaintenanceOperation(TypedDict, total=False):
    type: Literal["UPDATE", "MERGE", "DELETE"]
    target_id: str
    source_ids: list[str]
    section: str
    content: str
    reason: str


Repair = Callable[[str, int, int], list[MaintenanceOperation]]


def load_token_counter(tokenizer_name: str) -> Callable[[str], int]:
    """Load only a tokenizer, never model weights or remote Python code."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=False)

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    return count


@dataclass(frozen=True)
class BudgetDecision:
    playbook: str
    next_global_id: int
    accepted: bool
    maintenance: list[MaintenanceOperation]


def validate_maintenance(playbook: str, operations: list[MaintenanceOperation]) -> None:
    """Reject malformed, conflicting, or ineffective local edit targets."""
    parsed = [parse_playbook_line(line) for line in playbook.splitlines()]
    known_ids = {item["id"] for item in parsed if item}
    sections = {
        line.strip()[2:].strip().lower().replace(" ", "_").replace("&", "and")
        for line in playbook.splitlines() if line.strip().startswith("##")
    }
    touched: set[str] = set()
    for operation in operations:
        kind = operation.get("type")
        if kind not in {"UPDATE", "MERGE", "DELETE"}:
            raise ValueError("Budget maintenance allows only UPDATE, MERGE, DELETE")
        reason = operation.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Budget maintenance requires a short evidence-based reason")
        if kind != "DELETE":
            content = operation.get("content")
            if not isinstance(content, str) or not content.strip() or "\n" in content or "\r" in content:
                raise ValueError("Maintenance content must be a nonempty, single-line rule")
        if kind == "MERGE":
            ids = operation.get("source_ids")
            section = operation.get("section")
            if not isinstance(section, str) or section.strip().lower().replace(" ", "_").replace("&", "and") not in sections:
                raise ValueError("MERGE must name an existing section")
            if not isinstance(ids, list) or len(ids) < 2 or not all(isinstance(item, str) for item in ids):
                raise ValueError("MERGE requires at least two distinct source IDs")
            if len(set(ids)) != len(ids):
                raise ValueError("MERGE source IDs must be distinct")
        else:
            target_id = operation.get("target_id")
            if not isinstance(target_id, str):
                raise ValueError("Maintenance target_id must be a string")
            ids = [target_id]
        if not set(ids) <= known_ids or touched.intersection(ids):
            raise ValueError("Maintenance targets must exist and cannot overlap within a batch")
        touched.update(ids)


class PlaybookBudget:
    """Accept a bounded candidate, repair it locally, or keep the old version.

    This limits serialized Playbook tokens, not the full LLM request. Semantic
    preservation is requested from the Curator, not guaranteed by token checks.
    """

    def __init__(
        self,
        limit: int,
        count_tokens: Callable[[str], int],
        repair_attempts: int,
        log_dir: Path,
        tokenizer_name: str,
    ) -> None:
        if limit <= 0 or repair_attempts < 0:
            raise ValueError("Playbook budget must be positive; repair attempts must be nonnegative")
        self.limit = limit
        self.count_tokens = count_tokens
        self.repair_attempts = repair_attempts
        self.log_dir = log_dir
        self.tokenizer_name = tokenizer_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def require_within_budget(self, playbook: str) -> None:
        tokens = self.count_tokens(playbook)
        if tokens > self.limit:
            raise ValueError(
                f"Playbook has {tokens} tokens, above hard budget {self.limit} "
                f"({self.tokenizer_name}). Supply a smaller initial Playbook or increase the budget."
            )

    def _log(self, event: str, call_id: str, **fields: object) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "call_id": call_id,
            "budget": self.limit,
            "tokenizer": self.tokenizer_name,
            **fields,
        }
        with (self.log_dir / "playbook_budget_events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        print(f"[PLAYBOOK BUDGET] {call_id}: {event} {fields}")

    def enforce(
        self,
        previous: str,
        candidate: str,
        previous_next_id: int,
        candidate_next_id: int,
        repair: Repair,
        call_id: str,
    ) -> BudgetDecision:
        self.require_within_budget(previous)
        before = self.count_tokens(previous)
        tokens = self.count_tokens(candidate)
        self._log("candidate_checked", call_id, previous_tokens=before, candidate_tokens=tokens)
        maintenance: list[MaintenanceOperation] = []
        original_candidate = candidate
        attempted_edits: list[dict[str, object]] = []
        overflow = tokens > self.limit
        for attempt in range(1, self.repair_attempts + 1):
            if tokens <= self.limit:
                break
            try:
                operations = deepcopy(repair(candidate, tokens, attempt))
                validate_maintenance(candidate, operations)
                repaired, repaired_id = apply_curator_operations(candidate, operations, candidate_next_id)
                repaired_tokens = self.count_tokens(repaired)
                attempted_edits.append({"attempt": attempt, "operations": operations, "tokens": repaired_tokens})
                # A failed or expanding edit cannot consume IDs or replace the candidate.
                made_progress = repaired_tokens < tokens
                if made_progress:
                    candidate, candidate_next_id, tokens = repaired, repaired_id, repaired_tokens
                    maintenance.extend(operations)
                self._log("repair_attempt", call_id, attempt=attempt, tokens_after=tokens,
                          proposed_tokens=repaired_tokens, operations=len(operations))
                if not made_progress:
                    # No further deterministic call on unchanged input can help.
                    break
            except (ValueError, KeyError, TypeError) as exc:
                attempted_edits.append({"attempt": attempt, "error": str(exc)})
                self._log("repair_invalid", call_id, attempt=attempt, error=str(exc))
                break
        accepted = tokens <= self.limit
        if overflow:
            # Preserve deferred insights and repair provenance without exposing them to execution.
            with (self.log_dir / "playbook_budget_candidates.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "call_id": call_id, "accepted": accepted, "candidate": original_candidate,
                    "attempts": attempted_edits, "final_candidate": candidate,
                }, ensure_ascii=False) + "\n")
        self._log("committed" if accepted else "rolled_back", call_id,
                  tokens_after=tokens if accepted else before, maintenance_operations=len(maintenance))
        return BudgetDecision(
            candidate if accepted else previous,
            candidate_next_id if accepted else previous_next_id,
            accepted,
            maintenance if accepted else [],
        )
