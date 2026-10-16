"""Run: python -m agent_trace_evals.rag.bench [--llm mock|opencode] [--model M] [--reranker NAME] [--out DIR]
Traces go to out/ (gitignored); the committed rag_data/traces_*.jsonl are the published run and are never overwritten."""
import argparse, json, time
from pathlib import Path
from .data import load
from .retrieval import WordOverlap, BM25, Dense, Hybrid, Rerank
from .metrics import evaluate_retriever
from .pipeline import RAGPipeline, score_e2e
from .llm import MockLLM, OpenCodeLLM

# rerank-logit gate per reranker, picked on these 30 queries (not held out): bge gap is (-6.80, -5.74], jina (-2.02, -1.94]
DEFAULT_GATE = {"BAAI/bge-reranker-base": -6.5, "jinaai/jina-reranker-v2-base-multilingual": -2.5}

def make_embed_query(emb, queries):
    table = {q["query"]: emb["queries"][q["qid"]] for q in queries}; model = {}
    def f(text):
        if text in table: return table[text]
        if "m" not in model:
            from fastembed import TextEmbedding; model["m"] = TextEmbedding(emb["model"])
        return list(model["m"].embed([text]))[0]
    return f

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--llm", default="mock"); ap.add_argument("--model", default="opencode/deepseek-v4.1-flash-free")
    ap.add_argument("--reranker", default="BAAI/bge-reranker-base"); ap.add_argument("--retriever", default="hybrid+rerank")
    ap.add_argument("--min-score", type=float, default=None, help="rerank-logit gate (default per reranker, see DEFAULT_GATE; off for other retrievers)")
    ap.add_argument("--no-gate", action="store_true"); ap.add_argument("--out", default=None); a = ap.parse_args()
    if a.no_gate and a.min_score is not None: ap.error("--no-gate and --min-score are mutually exclusive")
    if a.no_gate: a.min_score = None
    elif a.min_score is None and a.retriever == "hybrid+rerank":
        if a.reranker not in DEFAULT_GATE: ap.error(f"no calibrated gate for {a.reranker}: pass --min-score or --no-gate")
        a.min_score = DEFAULT_GATE[a.reranker]
    elif a.min_score is not None and a.retriever != "hybrid+rerank": ap.error("--min-score is a cross-encoder logit; only valid with --retriever hybrid+rerank")
    docs, qs, emb = load(); bm = BM25(docs); dense = Dense(docs, emb["docs"], make_embed_query(emb, qs)); hyb = Hybrid(bm, dense)
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    ce = TextCrossEncoder(a.reranker); memo = {}  # one cross-encoder pass per query, reused by metrics, gate printout and pipeline
    rr = Rerank(hyb, docs, lambda q, t: memo.setdefault((q, tuple(t)), list(ce.rerank(q, t))))
    rows = {}
    for r in [WordOverlap(docs), bm, dense, hyb, rr]:
        t0 = time.time(); m = evaluate_retriever(r, qs); m["ms/query"] = round((time.time() - t0) * 1000 / sum(1 for q in qs if q["gold"]), 1); rows[r.name] = m
    cols = list(next(iter(rows.values())))
    print("| retriever | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols))
    for n, m in rows.items(): print(f"| {n} | " + " | ".join(str(m[c]) for c in cols) + " |")
    # unanswerable gate calibration info: top rerank score answerable vs not
    for q in qs: q["_top"] = (rr.search(q["query"], 1) or [("", -99)])[0][1]
    lo_a = min(q["_top"] for q in qs if q["gold"]); hi_u = max(q["_top"] for q in qs if not q["gold"])
    g = a.min_score if a.min_score is not None else float("-inf")
    print(f"top rerank score: answerable min {lo_a:.3f}, unanswerable max {hi_u:.3f}; gate {a.min_score} "
          f"refuses {sum(q['_top'] < g for q in qs if not q['gold'])}/{sum(not q['gold'] for q in qs)} unanswerable, "
          f"{sum(q['_top'] < g for q in qs if q['gold'])} answerable (threshold chosen on this same set, not held out)")
    llm = MockLLM() if a.llm == "mock" else OpenCodeLLM(a.model)
    ret = {"hybrid+rerank": rr, "hybrid": hyb, "bm25": bm}[a.retriever]
    out = Path(a.out) if a.out else Path(__file__).resolve().parents[2] / "out"; out.mkdir(parents=True, exist_ok=True)
    slug = "" if a.reranker == "BAAI/bge-reranker-base" else (a.reranker.split("/")[0].replace("jinaai", "jina") + "_")
    trace = out / f"traces_{a.llm}_{slug}{a.retriever}_{'nogate' if a.min_score is None else f'gate{a.min_score}'}.jsonl"; trace.unlink(missing_ok=True)
    p = RAGPipeline(ret, docs, llm, k=3, min_score=a.min_score, trace_path=trace)
    recs = []
    for q in qs:
        recs.append(p.answer(q["query"], q["qid"])); print(q["qid"], recs[-1]["refused"], recs[-1]["answer"][:100].replace("\n", " "), flush=True)
    print(json.dumps(score_e2e(recs, qs), indent=1))
if __name__ == "__main__": main()
