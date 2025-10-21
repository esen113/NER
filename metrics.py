from typing import List, Tuple, Dict, Any
from collections import Counter, defaultdict
import math

def norm(s: str) -> str:
    return " ".join(str(s).strip().split()).lower()

def normalize_triple(tri: Tuple[str, str, str]) -> Tuple[str, str, str]:
    h, r, t = tri
    return (norm(h), norm(r), norm(t))

def as_set_of_triples(triples: List[Tuple[str, str, str]]):
    return set(normalize_triple(t) for t in triples)

def prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1}

def micro_prf(pred: List[List[Tuple[str,str,str]]], gold: List[List[Tuple[str,str,str]]]) -> Dict[str, float]:
    tp=fp=fn=0
    for ps, gs in zip(pred, gold):
        P = as_set_of_triples(ps)
        G = as_set_of_triples(gs)
        tp += len(P & G)
        fp += len(P - G)
        fn += len(G - P)
    return prf(tp, fp, fn)

def headtail_micro_prf(pred: List[List[Tuple[str,str,str]]], gold: List[List[Tuple[str,str,str]]]) -> Dict[str, float]:
    def to_pairs(tri): return set((norm(h), norm(t)) for h,_,t in tri)
    tp=fp=fn=0
    for ps, gs in zip(pred, gold):
        P = to_pairs(ps); G = to_pairs(gs)
        tp += len(P & G); fp += len(P - G); fn += len(G - P)
    return prf(tp, fp, fn)

def entity_micro_prf(pred: List[List[Tuple[str,str,str]]], gold: List[List[Tuple[str,str,str]]]) -> Dict[str,float]:
    def ents(tri): return set([norm(h) for h,_,_ in tri] + [norm(t) for _,_,t in tri])
    tp=fp=fn=0
    for ps, gs in zip(pred, gold):
        P = ents(ps); G = ents(gs)
        tp += len(P & G); fp += len(P - G); fn += len(G - P)
    return prf(tp, fp, fn)

def relation_bag_micro_prf(pred: List[List[Tuple[str,str,str]]], gold: List[List[Tuple[str,str,str]]]) -> Dict[str,float]:
    # Bag (multiset) micro P/R/F1 on relation labels only
    tp=fp=fn=0
    for ps, gs in zip(pred, gold):
        Pc = Counter(norm(r) for _, r, _ in ps)
        Gc = Counter(norm(r) for _, r, _ in gs)
        # true positives: sum of min counts per label
        labels = set(Pc) | set(Gc)
        tp += sum(min(Pc[l], Gc[l]) for l in labels)
        fp += sum(max(Pc[l]-Gc[l],0) for l in labels)
        fn += sum(max(Gc[l]-Pc[l],0) for l in labels)
    return prf(tp, fp, fn)

def per_relation_f1(pred: List[List[Tuple[str,str,str]]], gold: List[List[Tuple[str,str,str]]]) -> Dict[str, Dict[str,float]]:
    labels = set(norm(r) for s in gold for _,r,_ in s) | set(norm(r) for s in pred for _,r,_ in s)
    out = {}
    for l in labels:
        tp=fp=fn=0
        for ps, gs in zip(pred, gold):
            P = set((norm(h), norm(t)) for h,r,t in ps if norm(r)==l)
            G = set((norm(h), norm(t)) for h,r,t in gs if norm(r)==l)
            tp += len(P & G); fp += len(P - G); fn += len(G - P)
        out[l] = prf(tp, fp, fn)
    return out

def macro_from_per_class(per_class: Dict[str, Dict[str,float]]) -> Dict[str,float]:
    # simple arithmetic mean over classes
    ks = list(per_class.keys())
    if not ks: return {"precision":0.0,"recall":0.0,"f1":0.0}
    agg = {"precision":0.0,"recall":0.0,"f1":0.0}
    for k in ks:
        for m in ["precision","recall","f1"]:
            agg[m] += per_class[k][m]
    for m in agg:
        agg[m] /= len(ks)
    return agg
