"""Precompute doc+query embeddings with a local CPU model (fastembed/ONNX). Run: python -m agent_trace_evals.rag.build_embeddings"""
import json
from .data import DATA
from .retrieval import doc_text
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
def main():
    from fastembed import TextEmbedding
    docs = json.loads((DATA / "corpus.json").read_text()); qs = json.loads((DATA / "queries.json").read_text())
    m = TextEmbedding(MODEL); r6 = lambda v: [round(float(x), 5) for x in v]
    out = {"model": MODEL, "dim": 384, "docs": {d["id"]: r6(v) for d, v in zip(docs, m.embed([doc_text(d) for d in docs]))},
           "queries": {q["qid"]: r6(v) for q, v in zip(qs, m.embed([q["query"] for q in qs]))}}
    (DATA / "embeddings.json").write_text(json.dumps(out)); print("wrote", len(out["docs"]), len(out["queries"]))
if __name__ == "__main__": main()
