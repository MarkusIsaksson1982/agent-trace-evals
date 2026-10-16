"""Reference (not given to models): stdlib RAG evaluation helpers."""
import math, re
TOK = re.compile(r"\w+")
def _t(s): return TOK.findall(s.lower())
def chunk(text, size=400, overlap=80):
    if overlap >= size or size <= 0 or overlap < 0: raise ValueError("need 0 <= overlap < size")
    w = text.split()
    if not w: return []
    out, i, step = [], 0, size - overlap
    while True:
        out.append(" ".join(w[i:i + size]))
        if i + size >= len(w): return out
        i += step
def retrieve(query, docs, k=3):
    q = set(_t(query)); res = []
    for i, d in docs.items():
        dt = _t(d); s = sum(1 for x in dt if x in q)
        if s: res.append((i, s / math.sqrt(len(dt))))
    res.sort(key=lambda x: (-x[1], x[0])); return res[:k]
def recall_at_k(retrieved, relevant, k):
    rel = set(relevant); return len(rel & set(retrieved[:k])) / len(rel) if rel else 0.0
def mrr(retrieved, relevant):
    for n, i in enumerate(retrieved, 1):
        if i in relevant: return 1 / n
    return 0.0
def pack_context(hits, budget_chars):
    out, used = [], 0
    for i, t in hits:
        if used + len(t) <= budget_chars: out.append(i); used += len(t)
    return out
def grounding(answer, retrieved_ids):
    c = list(dict.fromkeys(re.findall(r"\[([^\[\]\s]+)\]", answer))); unk = [x for x in c if x not in retrieved_ids]
    return {"citations": c, "grounded": bool(c) and not unk, "unknown": unk}
def evaluate(cases, docs, k=3):
    R = M = 0.0
    for c in cases:
        ids = [i for i, _ in retrieve(c["query"], docs, k)]; R += recall_at_k(ids, c["relevant"], k); M += mrr(ids, c["relevant"])
    n = len(cases) or 1
    return {"cases": len(cases), "recall_at_k": R / n, "mrr": M / n}
