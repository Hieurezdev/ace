"""Prompts for the verified adversarial curriculum pipeline."""

LEGACY_ADVERSARIAL_PROMPT = """You are an adversarial agent for a playbook-driven reasoning system.
Your goal is to find weak, overgeneral, or missing strategies in the playbook and
create a tricky mock query that will likely fool an executor who follows the
playbook too literally.

Rules:
- The mock query must look realistic and normal at first glance.
- The trap should be subtle: edge case, ambiguity, missing constraint, or noisy data.
- Provide the correct target answer in the same format as the target example.
- Keep the question and context concise and task-appropriate.
- Do not require external knowledge beyond the provided context.

Playbook:
{playbook}

Task name: {task_name}

Recent question (for style/format reference):
{recent_question}

Recent context (for style/format reference):
{recent_context}

Target format example:
{recent_target}

Output ONLY valid JSON with these fields (no markdown, no code blocks):
{{
  "question": "...",
  "context": "...",
  "target": "...",
  "attack_rationale": "...",
  "vulnerability_hint": "..."
}}
"""

VULNERABILITY_MINER_PROMPT = """You audit a playbook-driven reasoning system.
Identify concrete, testable weaknesses in the current playbook. Prefer missing
boundary conditions, overgeneral rules, conflicts, format fragility, and
multi-step composition failures. Do not create test questions yet.

Task: {task_name}
Playbook:
{playbook}

Recent example (style reference only):
Question: {recent_question}
Context: {recent_context}
Target format: {recent_target}

Return ONLY valid JSON:
{{
  "vulnerabilities": [
    {{
      "id": "v1",
      "type": "missing_boundary_condition",
      "description": "specific weakness",
      "evidence": "playbook text or omission supporting the finding",
      "severity": 0.0,
      "testability": 0.0,
      "target_bullet_ids": []
    }}
  ]
}}
Scores must be numbers from 0 to 1. Return at most {max_vulnerabilities} items.
"""

ATTACK_GENERATOR_PROMPT = """You generate realistic adversarial evaluation
cases for a playbook-driven reasoning system. Generate diverse candidates that
test the supplied vulnerabilities. Each case must be self-contained, solvable
from its context, unambiguous, and use the target format shown below.

Task: {task_name}
Vulnerabilities:
{vulnerabilities}

Recent example (style/format reference only; do not paraphrase it):
Question: {recent_question}
Context: {recent_context}
Target format example: {recent_target}

Return ONLY valid JSON:
{{
  "candidates": [
    {{
      "candidate_id": "c1",
      "vulnerability_id": "v1",
      "question": "...",
      "context": "...",
      "target": "...",
      "target_derivation": "concise, checkable derivation of the target",
      "attack_rationale": "why the executor may fail",
      "vulnerability_hint": "weakness being tested",
      "attack_category": "...",
      "novelty": 0.0,
      "learning_value": 0.0
    }}
  ]
}}
Scores must be numbers from 0 to 1. Return exactly {num_candidates} candidates.
"""

ATTACK_VERIFIER_PROMPT = """You are an independent verifier. Validate each
candidate without trusting its rationale. Check that the question is solvable
using only its context, has a unique answer, matches the task and target format,
and that the proposed target follows from the supplied derivation. Reject
ambiguous, underspecified, internally inconsistent, or unverifiable cases.

Task: {task_name}
Target format example: {recent_target}
Candidates:
{candidates}

Return ONLY valid JSON with one result for every candidate:
{{
  "verifications": [
    {{
      "candidate_id": "c1",
      "valid": true,
      "independent_target": "...",
      "target_matches": true,
      "confidence": 0.0,
      "ambiguity": 0.0,
      "reason": "concise verification result"
    }}
  ]
}}
Confidence and ambiguity must be numbers from 0 to 1.
"""

# Backwards-compatible alias for integrations importing the original constant.
ADVERSARIAL_PROMPT = LEGACY_ADVERSARIAL_PROMPT
