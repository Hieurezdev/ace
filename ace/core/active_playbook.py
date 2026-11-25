"""Select paired replay cases before opening any checker outcomes."""

from collections import Counter
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
import random
import re
from typing import TypeVar


@dataclass(frozen=True)
class ProbeCase:
    key: str
    text: str
    prior_success: bool


@dataclass(frozen=True)
class Candidate:
    playbook: str
    next_id: int
    operation_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class Decision:
    selected: int
    checked_cases: tuple[str, ...]
    candidate_scores: tuple[tuple[int, int], ...]
    rollouts: int
    checker_calls: int
    disagreeing_cases: int


Observation = TypeVar("Observation")


def probe_pool(history: Sequence[ProbeCase], limit: int) -> list[ProbeCase]:
    """Keep the trigger and a known success, then prefer locally related cases."""
    # Last occurrence is authoritative when the same training task reappears.
    if not history:
        return []
    latest = history[-1]
    recent = list({case.key: case for case in history}.values())
    by_recency = {case.key: index for index, case in enumerate(history)}
    anchor = max(
        (case for case in recent if case.prior_success and case.key != latest.key),
        key=lambda case: by_recency[case.key], default=None,
    )
    selected = [latest] if anchor is None else [latest, anchor]
    terms = set(re.findall(r"[a-z][a-z0-9]+", latest.text.lower()))

    def priority(case: ProbeCase) -> tuple[int, int]:
        overlap = terms & set(re.findall(r"[a-z][a-z0-9]+", case.text.lower()))
        return -len(overlap), -by_recency[case.key]

    remaining = sorted((case for case in recent if case not in selected), key=priority)
    return (selected + remaining)[:limit]


def candidate_versions(
    base: str,
    next_id: int,
    operation_count: int,
    limit: int,
    apply: Callable[[tuple[int, ...]], tuple[str, int]],
    fits: Callable[[str], bool],
) -> list[Candidate]:
    """Build HOLD, the full proposal, and distinct single-operation alternatives."""
    candidates = [Candidate(base, next_id)]
    seen = {base}
    if not operation_count:
        return candidates
    indices = [tuple(range(operation_count)), *((i,) for i in range(operation_count))]
    for edit in dict.fromkeys(indices):
        proposal, proposed_id = apply(edit)
        if proposal in seen or not fits(proposal):
            continue
        seen.add(proposal)
        candidates.append(Candidate(proposal, proposed_id, edit))
        if len(candidates) == limit:
            break
    return candidates


def select_edit(
    candidates: Sequence[Candidate],
    cases: Sequence[ProbeCase],
    rollout: Callable[[ProbeCase, Candidate], Observation],
    behavior_key: Callable[[Observation], Hashable],
    grade: Callable[[ProbeCase, Observation], bool],
    *,
    policy: str,
    checker_cases: int,
    seed: int,
) -> Decision:
    """Select labelled checks using only behavior, then compare against HOLD."""
    if len(candidates) < 2 or not cases:
        return Decision(0, (), (), 0, 0, 0)
    observations = [[rollout(case, candidate) for candidate in candidates] for case in cases]
    distinct = [Counter(behavior_key(item) for item in row) for row in observations]
    indices = list(range(len(cases)))
    random.Random(seed).shuffle(indices)
    # In both ablations reserve a previously solved case for preservation.
    anchor = next((i for i in indices if cases[i].prior_success), None)
    if anchor is not None:
        indices.remove(anchor)
    if policy == "disagreement":
        indices.sort(key=lambda i: len(candidates) - max(distinct[i].values()), reverse=True)
    elif policy != "random":
        raise ValueError(f"Unknown active case selection policy: {policy}")
    checked = ([] if anchor is None else [anchor]) + indices
    checked = checked[:checker_cases]
    # Each selected case is paired across all candidate versions.
    scores = [(0, 0) for _ in candidates]
    for index in checked:
        outcomes = [grade(cases[index], observation) for observation in observations[index]]
        base = outcomes[0]
        scores = [
            (repairs + int(not base and correct), regressions + int(base and not correct))
            for (repairs, regressions), correct in zip(scores, outcomes, strict=True)
        ]
    eligible = [i for i, (repairs, regressions) in enumerate(scores) if repairs and not regressions]
    winner = max(eligible, key=lambda i: (scores[i][0], -i), default=0)
    return Decision(
        winner, tuple(cases[i].key for i in checked), tuple(scores),
        len(cases) * len(candidates), len(checked) * len(candidates),
        sum(len(values) > 1 for values in distinct),
    )
