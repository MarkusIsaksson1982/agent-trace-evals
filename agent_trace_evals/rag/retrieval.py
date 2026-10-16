"""Retrievers. Dense vectors are precomputed (rag_data/embeddings.json) so tests and the TS port need no model."""
from __future__ import annotations
import math, re
from collections import Counter
import numpy as np

TOK = re.compile(r"[0-9a-zåäöéü]+")
STOP = set("the a an and or to of in on is i my me how do can what for with from at be by are you your this that".split()) | \
    set("och att det jag är på en ett av för med som har inte till om vad hur min mitt".split())

def tokens(s: str) -> list[str]:
    return [t for t in TOK.findall(s.lower()) if t not in STOP]

def doc_text(d: dict) -> str:
    return f"{d['title']}. {d['text']}"

class WordOverlap:
    name = "overlap"
    def __init__(self, docs): self.docs = docs; self.toks = [set(tokens(doc_text(d))) for d in docs]
    def search(self, q, k=10):
        qt = set(tokens(q)); s = [(len(qt & t), d["id"]) for t, d in zip(self.toks, self.docs)]
        return [(i, float(v)) for v, i in sorted(s, key=lambda x: (-x[0], x[1]))[:k] if v > 0]

class BM25:
    name = "bm25"
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.tf = [Counter(tokens(doc_text(d))) for d in docs]
        self.len = [sum(c.values()) for c in self.tf]; self.avg = sum(self.len) / len(docs)
        df = Counter(t for c in self.tf for t in c); n = len(docs)
        self.idf = {t: math.log(1 + (n - v + 0.5) / (v + 0.5)) for t, v in df.items()}
    def score(self, q):
        out = []
        for c, L, d in zip(self.tf, self.len, self.docs):
            s = 0.0
            for t in tokens(q):
                f = c.get(t, 0)
                if f: s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * L / self.avg))
            out.append((d["id"], s))
        return out
    def search(self, q, k=10):
        return [x for x in sorted(self.score(q), key=lambda x: (-x[1], x[0])) if x[1] > 0][:k]

class Dense:
    """Cosine over a numpy matrix of L2-normalised vectors (the 'vector store')."""
    name = "dense"
    def __init__(self, docs, doc_vecs: dict, embed_query):
        self.ids = [d["id"] for d in docs]
        m = np.array([doc_vecs[i] for i in self.ids], dtype=np.float32)
        self.m = m / np.linalg.norm(m, axis=1, keepdims=True); self.embed_query = embed_query
    def search(self, q, k=10):
        v = np.asarray(self.embed_query(q), dtype=np.float32); v /= np.linalg.norm(v)
        s = self.m @ v  # ties broken by doc id, same as BM25/RRF here and the TS port
        return sorted(((self.ids[i], float(s[i])) for i in range(len(self.ids))), key=lambda x: (-x[1], x[0]))[:k]

def rrf(rankings: list[list[tuple[str, float]]], k=60, top=10):
    acc: dict[str, float] = {}
    for r in rankings:
        for rank, (i, _) in enumerate(r):
            acc[i] = acc.get(i, 0.0) + 1.0 / (k + rank + 1)
    return sorted(acc.items(), key=lambda x: (-x[1], x[0]))[:top]

class Hybrid:
    name = "hybrid"
    def __init__(self, sparse, dense, depth=20): self.sparse, self.dense, self.depth = sparse, dense, depth
    def search(self, q, k=10): return rrf([self.sparse.search(q, self.depth), self.dense.search(q, self.depth)], top=k)

class Rerank:
    """Re-scores the first stage's top-N with a scorer(query, texts)->scores (cross-encoder or fallback)."""
    name = "hybrid+rerank"
    def __init__(self, first, docs, scorer, depth=15):
        self.first, self.scorer, self.depth = first, scorer, depth; self.by = {d["id"]: d for d in docs}
    def search(self, q, k=10):
        cand = [i for i, _ in self.first.search(q, self.depth)]
        if not cand: return []
        s = [float(x) for x in self.scorer(q, [doc_text(self.by[i]) for i in cand])]
        if len(s) != len(cand): raise ValueError(f"scorer returned {len(s)} scores for {len(cand)} candidates")
        return sorted(zip(cand, s), key=lambda x: (-x[1], x[0]))[:k]
