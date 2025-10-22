import json, re, ast
from typing import List, Tuple, Dict, Any, Optional
import json5

def robust_json_parse(text: str) -> Dict[str, Any]:
    """
    Try to parse model output into {"triples":[{"head":"","relation":"","tail":""}, ...]}
    Accepts:
    - proper JSON
    - JSON5 (trailing commas, single quotes)
    - python dict literal
    - line-based: HEAD|||REL|||TAIL (one per line)
    """
    text = text.strip()

    # 1) Try JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
        # Some models output array directly
        if isinstance(obj, list):
            return {"triples": obj}
    except Exception:
        pass

    # 2) Try JSON5
    try:
        obj = json5.loads(text)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list):
            return {"triples": obj}
    except Exception:
        pass

    # 3) Try python literal_eval
    try:
        obj = ast.literal_eval(text)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list):
            return {"triples": obj}
    except Exception:
        pass

    # 4) Try line-based fallback
    triples = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "|||" in line:
            parts = [p.strip() for p in line.split("|||")]
            if len(parts)==3:
                triples.append({"head":parts[0], "relation":parts[1], "tail":parts[2]})
    if triples:
        return {"triples": triples}

    # 5) Try to locate JSON substring
    m = re.search(r'\{.*\}', text, flags=re.S)
    if m:
        try:
            obj = json5.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return {"triples": []}

def to_tuple_list(obj: Dict[str, Any]) -> List[Tuple[str,str,str]]:
    triples = obj.get("triples", obj.get("extractions", []))
    out = []
    for it in triples:
        if isinstance(it, dict):
            h = it.get("head") or it.get("subject") or it.get("sub") or ""
            r = it.get("relation") or it.get("rel") or ""
            t = it.get("tail") or it.get("object") or it.get("obj") or ""
        elif isinstance(it, (list, tuple)) and len(it)==3:
            h, r, t = it
        else:
            continue
        h, r, t = str(h).strip(), str(r).strip(), str(t).strip()
        if h and r and t:
            out.append((h, r, t))
    return out

def read_dataset(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Normalize structure: list of dicts with text and triple_list
    ds = []
    for item in data:
        ds.append({"text": item["text"], "triple_list": [tuple(tri) for tri in item.get("triple_list", [])]})
    return ds

def relation_vocab_of_dataset(ds) -> List[str]:
    rels = set()
    for item in ds:
        for _, r, _ in item["triple_list"]:
            rels.add(str(r))
    return sorted(rels)

def parse_semantic_judgment(text: str) -> int:
    """
    Parse the model's semantic evaluation output and map it to {0,1}.
    Falls back to 0 if the intent cannot be determined.
    """
    text = (text or "").strip()

    def _coerce(value: Any) -> Optional[int]:
        if isinstance(value, bool):
            return 1 if value else 0
        if isinstance(value, (int, float)):
            return 1 if int(value) == 1 else 0 if int(value) == 0 else None
        if isinstance(value, str):
            stripped = value.strip()
            if stripped in {"1", "0"}:
                return int(stripped)
            if stripped.lower() in {"yes", "true"}:
                return 1
            if stripped.lower() in {"no", "false"}:
                return 0
        return None

    def _extract(obj: Any) -> Optional[int]:
        if isinstance(obj, dict):
            for key in ["semantic_match", "match", "verdict", "result", "score"]:
                if key in obj:
                    coerced = _coerce(obj[key])
                    if coerced is not None:
                        return coerced
        coerced = _coerce(obj)
        if coerced is not None:
            return coerced
        if isinstance(obj, list) and len(obj) == 1:
            return _extract(obj[0])
        return None

    parsers = [json.loads, json5.loads]
    for parser in parsers:
        try:
            obj = parser(text)
            val = _extract(obj)
            if val is not None:
                return val
        except Exception:
            pass

    try:
        obj = ast.literal_eval(text)
        val = _extract(obj)
        if val is not None:
            return val
    except Exception:
        pass

    m = re.search(r"\b([01])\b", text)
    if m:
        return int(m.group(1))

    return 0
