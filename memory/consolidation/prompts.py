"""Prompts for LLM-driven conflict adjudication (§17)."""

from __future__ import annotations

CONFLICT_SYSTEM = """\
You compare one NEW candidate memory against a small set of EXISTING memories.

For each existing memory, classify the relationship as exactly one of:
- DUPLICATE: they express the same fact in different words
- SUPERSEDES: both describe the same subject, but the new one carries an updated or more recent state (e.g. "user prefers X" -> "user now prefers Y")
- CONTRADICTS: they cannot both be true and neither clearly supersedes the other
- INDEPENDENT: they can coexist without conflict

Respond with strict JSON of the shape:
{"decisions": [{"existing_id": "<id>", "decision": "DUPLICATE|SUPERSEDES|CONTRADICTS|INDEPENDENT", "reason": "<one short sentence>"}]}

Return one decision per existing memory listed. Do not include prose outside the JSON.
"""


CONFLICT_USER_TEMPLATE = """\
NEW MEMORY:
{new_content}

EXISTING MEMORIES:
{existing_block}
"""
