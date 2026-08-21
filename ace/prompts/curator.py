"""
Curator prompts for ACE system.
"""

# Curator prompt for intelligent playbook management
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
