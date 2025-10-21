from typing import List, Dict, Any

SYSTEM_PROMPT = """\
You are a precise information extraction assistant.
Your task is to extract all factual subject–relation–object triples from the given passage.
Return ONLY valid JSON with this exact schema:
{
  "triples": [
    {"head": "string", "relation": "string", "tail": "string"},
    ...
  ]
}

Rules:
- Be exhaustive but avoid hallucinations. If uncertain, omit.
- Use entities exactly as they appear in the text (surface form).
- Prefer canonical relation labels from the provided RELATION_VOCAB if semantically appropriate.
- If none can be extracted, return {"triples": []}.
- Do not include explanations or any extra keys.
"""

USER_PROMPT_TEMPLATE = """\
Extract triples from the passage.
RELATION_VOCAB (use EXACT spelling if applicable): {relation_vocab}

Passage:
\"\"\"
{text}
\"\"\"
Respond with JSON only.
"""

def build_user_prompt(text: str, relation_vocab: List[str]) -> str:
    # Keep vocab short but helpful
    vocab_str = ", ".join(sorted(relation_vocab)) if relation_vocab else "N/A"
    return USER_PROMPT_TEMPLATE.format(text=text, relation_vocab=vocab_str)

def build_messages(text: str, relation_vocab: List[str]) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(text, relation_vocab)},
    ]
