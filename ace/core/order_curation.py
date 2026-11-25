"""Deterministic planning for batches of validated Curator operations."""

import json
from dataclasses import dataclass

Operation = dict[str, str | list[str]]


@dataclass(frozen=True)
class BatchPlan:
    operations: list[Operation]
    overlapping_groups: int
    reconciled_groups: int
    held_groups: int


def _targets(operation: Operation) -> set[str]:
    if operation["type"] == "MERGE":
        ids = operation.get("source_ids", [])
        return set(ids) if isinstance(ids, list) else set()
    if operation["type"] in {"UPDATE", "DELETE"}:
        target = operation.get("target_id") or operation.get("bullet_id")
        if isinstance(target, str) and target:
            return {target}
    return set()


def plan_batch(proposals: list[Operation], *, resolve_overlaps: bool) -> BatchPlan:
    """Canonicalize a fixed batch; hold overlaps without a conservative composition."""
    # Operation reasons are explanatory metadata, not part of edit identity.
    unique: dict[str, Operation] = {}
    for operation in sorted(proposals, key=lambda op: json.dumps(op, sort_keys=True)):
        significant = {
            key: value for key, value in operation.items()
            if key != "reason" and not key.startswith("_")
        }
        unique.setdefault(json.dumps(significant, sort_keys=True), operation)

    if not resolve_overlaps:
        return BatchPlan(list(unique.values()), 0, 0, 0)

    groups: list[tuple[set[str], list[Operation]]] = []
    for operation in unique.values():
        targets = _targets(operation)
        members = [operation]
        untouched = []
        for known_targets, known_members in groups:
            if targets & known_targets:
                targets |= known_targets
                members.extend(known_members)
            else:
                untouched.append((known_targets, known_members))
        groups = untouched + [(targets, members)]

    accepted, overlaps, reconciled, held = [], 0, 0, 0
    for _, members in groups:
        if len(members) == 1:
            accepted.extend(members)
            continue
        overlaps += 1
        merges = [op for op in members if op["type"] == "MERGE"]
        updates = [op for op in members if op["type"] == "UPDATE"]
        # Preserve the exact update text rather than asking the model to rewrite
        # it again. Behavioral replay decides whether this composition works.
        if len(merges) == len(updates) == 1 and len(members) == 2:
            update = str(updates[0]["content"]).strip()
            base = str(merges[0]["content"]).strip()
            content = base if update in base else f"{base} Additional condition: {update}"
            accepted.append({**merges[0], "content": " ".join(content.splitlines())})
            reconciled += 1
        else:
            held += 1

    # CREATE_META must precede edits that reference the section it creates.
    accepted.sort(key=lambda op: (op["type"] != "CREATE_META", json.dumps(op, sort_keys=True)))
    return BatchPlan(accepted, overlaps, reconciled, held)


def replay_counts(before: list[bool], after: list[bool]) -> tuple[int, int]:
    """Count repaired and regressed tasks on identical replay cases."""
    pairs = list(zip(before, after, strict=True))
    return sum(not a and b for a, b in pairs), sum(a and not b for a, b in pairs)
