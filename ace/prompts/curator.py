"""
Curator prompts for ACE system.
"""

import json


BUDGET_OPERATION_EXAMPLES: dict[str, dict[str, str | list[str]]] = {
    "UPDATE": {
        "type": "UPDATE", "target_id": "<existing_bullet_id>",
        "content": "<complete concise rule preserving its conditions>",
        "reason": "<why this removes redundant wording without changing scope>",
    },
    "MERGE": {
        "type": "MERGE", "source_ids": ["<existing_id_1>", "<existing_id_2>"],
        "section": "<existing_section>", "content": "<complete consolidated rule>",
        "reason": "<why the useful conditions of both sources are preserved>",
    },
    "DELETE": {
        "type": "DELETE", "target_id": "<eligible_delete_candidate_id>",
        "reason": "<evidence that this counter-qualified rule is harmful or superseded>",
    },
}


def build_budget_response_schema(allowed_operations: list[str]) -> dict[str, object]:
    """Use one operation contract for prompt examples and constrained decoding."""
    variants: list[dict[str, object]] = []
    for kind in allowed_operations:
        if kind not in BUDGET_OPERATION_EXAMPLES:
            raise ValueError(f"Unsupported budget maintenance operation: {kind}")
        properties: dict[str, object] = {}
        for field in BUDGET_OPERATION_EXAMPLES[kind]:
            if field == "type":
                properties[field] = {"type": "string", "enum": [kind]}
            elif field == "source_ids":
                properties[field] = {"type": "array", "items": {"type": "string"}, "minItems": 2}
            else:
                properties[field] = {"type": "string"}
        variants.append({
            "type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False,
        })
    if not variants:
        raise ValueError("Budget maintenance requires at least one enabled operation")
    return {
        "type": "object", "additionalProperties": False,
        "required": ["reasoning", "operations"],
        "properties": {
            "reasoning": {"type": "string"},
            "operations": {"type": "array", "items": {"anyOf": variants}},
        },
    }


def build_budget_response_examples(allowed_operations: list[str]) -> str:
    """Show each permitted operation in a complete, unambiguous JSON envelope."""
    examples = [
        f"{kind} example:\n" + json.dumps({
            "reasoning": "Short evidence-based justification.",
            "operations": [BUDGET_OPERATION_EXAMPLES[kind]],
        }, indent=2)
        for kind in allowed_operations
    ]
    examples.append('No safe edit:\n{"reasoning":"No safe local reduction.","operations":[]}')
    return "\n\n".join(examples)


# Curator prompt for intelligent playbook management
BUDGET_MAINTENANCE_PROMPT = """Maintain ONE canonical Playbook, not a query-specific subset.
The proposed update is {actual_tokens} tokens; its hard limit is {token_budget}.
Reduce at least {excess_tokens} tokens using LOCAL operations on the candidate.
Do not output a replacement Playbook, truncate text, or add new rules/sections.
UPDATE: remove verbose explanations while preserving the rule's applicability,
conditions, exceptions, units, formulas, and output constraints.
MERGE: consolidate truly redundant rules; preserve all distinct valid conditions.
Semantic similarity alone does not prove equivalence. Do not merge contradictions
into one unconditional rule. Do not invent evidence or alter execution counters.
DELETE is allowed ONLY if enabled below AND the target is in the supplied audit
list AND there is an evidence-based reason. Budget pressure is not such a reason.
Prefer preserving useful rare rules over fitting the budget. If no safe local
maintenance is available, return an empty operations list; code will roll back.

{operation_schema}

DELETE audit candidates:
{delete_candidates}

Recent task evidence (may be empty for counter or hygiene maintenance):
{recent_reflection}

Candidate Playbook (including the newly proposed insights):
{candidate}

Return only JSON with "reasoning" (a SHORT evidence-based rationale) and
"operations". Never output a full Playbook or a detailed reasoning trace.
Every item in "operations" MUST contain the literal key "type" with an enabled
uppercase operation name. Do not substitute "action", "operation", or an object
key such as "UPDATE" for "type". Use the exact field names in the examples.
Return an array of operation objects, not an array of strings or a keyed map.
The examples show alternative responses, not one combined batch. Replace all
placeholder IDs, sections, and content with values from the candidate; never
copy placeholder IDs or add fields such as "helpful" or "harmful".

{response_examples}
"""

CURATOR_PROMPT = """You are a master curator of knowledge. Your job is to identify what new insights should be added to an existing playbook based on a reflection from a previous attempt.

**Context:**
- The playbook you created will be used to help answering similar questions. 
- The reflection is generated using ground truth answers that will NOT be available when the playbook is being used. So you need to come up with content that can aid the playbook user to create predictions that likely align with ground truth. 

**CRITICAL: You MUST respond with valid JSON only. Do not use markdown formatting or code blocks.**

**Instructions:**
- Review the existing playbook and the reflection from the previous attempt
- Identify ONLY the NEW insights, strategies, or mistakes that are MISSING from the current playbook
- Avoid redundancy - if similar advice already exists, only add new content that is a perfect complement to the existing playbook
- Do NOT regenerate the entire playbook - only provide the additions needed
- Focus on quality over quantity - a focused, well-organized playbook is better than an exhaustive one
- Format your response as a PURE JSON object with specific sections
- For any operation if no new content to add, return an empty list for the operations field
- Be concise and specific - each addition should be actionable
- `helpful` and `harmful` are execution counters maintained by the Reflector;
  they are not editable Curator metadata. Do not use UPDATE to relabel or
  repair those counters. UPDATE may only replace the textual rule content.


**Training Context:**
- Total token budget: {token_budget} tokens
- Training progress: Sample {current_step} out of {total_samples}

**Current Playbook Stats:**
{playbook_stats}

**Recent Reflection:**
{recent_reflection}

**Current Playbook:**
{current_playbook}

**Question Context:**
{question_context}

**Your Task:**
Output ONLY a valid JSON object with `reasoning` and `operations`. Each
operation must follow the enabled operation schema below.

{operation_schema}

**RESPONSE FORMAT - Output ONLY this JSON structure (no markdown, no code blocks):**
{{
  "reasoning": "[Your chain of thought / reasoning / thinking process, detailed analysis and calculations here]",
  "operations": [
    {{
      "type": "[One enabled operation type]", 
      "section": "formulas_and_calculations",
      "content": "[New calculation method...]"
    }}
  ]
}}

---
"""

CURATOR_PROMPT_NO_GT = """You are a master curator of knowledge. Your job is to identify what new insights should be added to an existing playbook based on a reflection from a previous attempt.

**Context:**
- The playbook you created will be used to help answering similar questions. 
- The reflection is generated using environment feedback that will NOT be available when the playbook is being used.

**CRITICAL: You MUST respond with valid JSON only. Do not use markdown formatting or code blocks.**

**Instructions:**
- Review the existing playbook and the reflection from the previous attempt
- Identify ONLY the NEW insights, strategies, or mistakes that are MISSING from the current playbook
- Avoid redundancy - if similar advice already exists, only add new content that is a perfect complement to the existing playbook
- Do NOT regenerate the entire playbook - only provide the additions needed
- Focus on quality over quantity - a focused, well-organized playbook is better than an exhaustive one
- Format your response as a PURE JSON object with specific sections
- For any operation if no new content to add, return an empty list for the operations field
- Be concise and specific - each addition should be actionable
- `helpful` and `harmful` are execution counters maintained by the Reflector;
  they are not editable Curator metadata. Do not use UPDATE to relabel or
  repair those counters. UPDATE may only replace the textual rule content.


**Training Context:**
- Total token budget: {token_budget} tokens
- Training progress: Sample {current_step} out of {total_samples}

**Current Playbook Stats:**
{playbook_stats}

**Recent Reflection:**
{recent_reflection}

**Current Playbook:**
{current_playbook}

**Question Context:**
{question_context}

**Your Task:**
Output ONLY a valid JSON object with `reasoning` and `operations`. Each
operation must follow the enabled operation schema below.

{operation_schema}

**RESPONSE FORMAT - Output ONLY this JSON structure (no markdown, no code blocks):**
{{
  "reasoning": "[Your chain of thought / reasoning / thinking process, detailed analysis and calculations here]",
  "operations": [
    {{
      "type": "[One enabled operation type]", 
      "section": "formulas_and_calculations",
      "content": "[New calculation method...]"
    }}
  ]
}}

---
"""


LIFECYCLE_OPERATION_INSTRUCTIONS = {
"ADD": """
1. ADD: Create one new bullet with a fresh system-assigned ID.
   - section: destination section
   - content: actionable, reusable rule; do not include an ID yourself
""",
"UPDATE": """
2. UPDATE: Correct or clarify one existing bullet.
   - target_id: existing bullet ID
   - content: complete replacement content
   - reason: concrete error, ambiguity, or missing condition
""",
"DELETE": """
3. DELETE: Remove a demonstrably harmful, obsolete, or superseded bullet.
   - target_id: existing bullet ID
   - reason: evidence-based reason for removal (required)
   - evidence_failure_ids: optional verified failure IDs
""",
"MERGE": """
4. MERGE: Consolidate two or more redundant or complementary bullets.
   - source_ids: list of at least two existing bullet IDs
   - section: destination section
   - content: complete merged replacement content
   - reason: why consolidation preserves the useful information
""",
"CREATE_META": """
5. CREATE_META: Create a new organizational section only when the existing
   taxonomy cannot represent a reusable family of rules.
   - section_id: stable snake_case identifier
   - title: concise section heading
   - description: short scope description
""",
}


def build_lifecycle_operation_instructions(allowed_operations):
    allowed = [op for op in allowed_operations if op in LIFECYCLE_OPERATION_INSTRUCTIONS]
    if not allowed:
        allowed = ["ADD"]
    return """
**Enabled Operations (strict allow-list):**
Propose only operations justified by the current Playbook and supplied
reflection. Every referenced ID must exist. Never delete an entry merely
because it is irrelevant to the current query.

""" + "\n".join(LIFECYCLE_OPERATION_INSTRUCTIONS[op] for op in allowed) + """

Prefer UPDATE over DELETE when the core rule remains valid. Prefer MERGE over
DELETE when several entries express the same valid rule. Return an empty list
when no localized change is justified.
"""
