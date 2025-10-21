# prompting.py
from typing import List, Dict, Any
import json  # 新增

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
\"\"\"{text}\"\"\"
Respond with JSON only.
"""

def build_user_prompt(text: str, relation_vocab: List[str]) -> str:
    vocab_str = ", ".join(sorted(relation_vocab)) if relation_vocab else "N/A"
    return USER_PROMPT_TEMPLATE.format(text=text, relation_vocab=vocab_str)

# ====== 你要塞进 prompt 的 few-shot 示例（直接用）======
FEWSHOTS = [
    {
        "text": """There was no mention of the ` ` iron triangle ' ' of members of Congress , the news media and special interest groups who , in a speech to political appointees in Washington on Dec. 13 , Reagan claimed had prevented his administration from balancing the federal budget .""",
        "triple_list": [
            ["Congress", "Organization based in", "Washington"]
        ]
    },
    {
        "text": """Nor did he argue , as he did in a speech at the University of Virginia in Charlottesville Dec. 16 , that Congress had perpetuated a dangerous situation in Central America by its ` ` on-again , off-again indecisiveness ' ' on his program of aid to the anti-communist Contra rebels .""",
        "triple_list": [
            ["University of Virginia", "Organization based in", "Charlottesville"]
        ]
    },
    {
        "text": """Recognition of proper nouns in Japanese text has been studied as a part of the more general problem of morphological analysis in Japanese text processing ( [ 1 ] [ 2 ] ) .""",
        "triple_list": [
            ["Recognition of proper nouns", "Part-of", "morphological analysis"],
            ["proper nouns", "Part-of", "Japanese text"],
            ["morphological analysis", "Used-for", "Japanese text processing"]
        ]
    },
    {
        "text": """It has also been studied in the framework of Japanese information extraction ( [ 3 ] ) in recent years .""",
        "triple_list": [
            ["Japanese information extraction", "Used-for", "It"]
        ]
    },
]
USE_FEWSHOTS = True   # 需要时改 False 立刻关闭
FEWSHOT_K = 4         # 用前 K 条；想少占上下文就改成 2

def build_messages(text: str, relation_vocab: List[str]) -> List[Dict[str, Any]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if USE_FEWSHOTS:
        for ex in FEWSHOTS[:FEWSHOT_K]:
            # few-shot 的 user：照常喂同一套 RELATION_VOCAB
            messages.append({"role": "user", "content": build_user_prompt(ex["text"], relation_vocab)})
            # few-shot 的 assistant：直接给标准 JSON 答案
            messages.append({
                "role": "assistant",
                "content": json.dumps({
                    "triples": [{"head": h, "relation": r, "tail": t} for (h, r, t) in ex["triple_list"]]
                }, ensure_ascii=False)
            })
    # 目标样本
    messages.append({"role": "user", "content": build_user_prompt(text, relation_vocab)})
    return messages

