import math
def recall_at_k(ret, gold, k): return len(set(ret[:k]) & set(gold)) / len(gold) if gold else 0.0
def mrr(ret, gold):
    for r, i in enumerate(ret, 1):
        if i in gold: return 1.0 / r
    return 0.0
def ndcg_at_k(ret, gold, k):
    dcg = sum(1 / math.log2(r + 2) for r, i in enumerate(ret[:k]) if i in gold)
    idcg = sum(1 / math.log2(r + 2) for r in range(min(len(gold), k)))
    return dcg / idcg if idcg else 0.0
def evaluate_retriever(r, queries, ks=(1, 3, 5)):
    qs = [q for q in queries if q["gold"]]; out = {f"recall@{k}": 0.0 for k in ks}; out.update(mrr=0.0, **{"ndcg@5": 0.0})
    for q in qs:
        ids = [i for i, _ in r.search(q["query"], 10)]
        for k in ks: out[f"recall@{k}"] += recall_at_k(ids, q["gold"], k)
        out["mrr"] += mrr(ids, q["gold"]); out["ndcg@5"] += ndcg_at_k(ids, q["gold"], 5)
    return {m: round(v / len(qs), 3) for m, v in out.items()}
