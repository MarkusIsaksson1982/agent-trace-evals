import json
from agent_trace_evals.rag.data import load, DATA
from agent_trace_evals.rag.retrieval import BM25, WordOverlap, Dense, Hybrid, Rerank, rrf, tokens
from agent_trace_evals.rag.metrics import recall_at_k, mrr, ndcg_at_k, evaluate_retriever
from agent_trace_evals.rag.pipeline import RAGPipeline, faithfulness, score_e2e, REFUSAL, REFUSAL_SV, is_refusal, sentences
from agent_trace_evals.rag.llm import MockLLM, est_tokens

DOCS, QS, EMB = load()
QV = {q["query"]: EMB["queries"][q["qid"]] for q in QS}
dense = Dense(DOCS, EMB["docs"], QV.__getitem__)

def test_corpus_shape():
    assert 60 <= len(DOCS) <= 100 and len({d["id"] for d in DOCS}) == len(DOCS)
    assert len(QS) == 30 and sum(not q["gold"] for q in QS) >= 3
    ids = {d["id"] for d in DOCS}; assert all(g in ids for q in QS for g in q["gold"])

def test_tokens_stopwords_and_swedish():
    assert tokens("How do I reset my lösenord?") == ["reset", "lösenord"]

def test_metrics():
    assert recall_at_k(["a", "b"], ["b"], 1) == 0 and recall_at_k(["a", "b"], ["b"], 2) == 1
    assert mrr(["a", "b"], ["b"]) == 0.5 and mrr(["a"], ["z"]) == 0
    assert ndcg_at_k(["g"], ["g"], 5) == 1.0 and 0 < ndcg_at_k(["x", "g"], ["g"], 5) < 1

def test_bm25_beats_overlap_and_finds_exact():
    assert BM25(DOCS).search("Vasterby-Gast guest network")[0][0] == "wifi-01"
    assert evaluate_retriever(BM25(DOCS), QS)["mrr"] >= evaluate_retriever(WordOverlap(DOCS), QS)["mrr"]

def test_dense_cosine_and_quality():
    r = dense.search(QS[0]["query"], 3); assert r[0][1] >= r[1][1] and -1 <= r[0][1] <= 1
    assert evaluate_retriever(dense, QS)["recall@5"] >= 0.9

def test_rrf():
    out = rrf([[("a", 9), ("b", 1)], [("b", 5), ("c", 1)]])
    assert out[0][0] == "b" and {i for i, _ in out} == {"a", "b", "c"}

def test_hybrid_and_rerank_with_fake_scorer():
    h = Hybrid(BM25(DOCS), dense)
    rr = Rerank(h, DOCS, lambda q, texts: [1.0 if "Vasterby-Gast" in t else 0.0 for t in texts])
    assert rr.search("What wireless network should visitors use?", 1)[0][0] == "wifi-01"

def test_faithfulness():
    by = {d["id"]: d for d in DOCS}
    assert faithfulness("Your account unlocks automatically after 30 minutes [pwd-03].", by, ["pwd-03"]) == 1.0
    assert faithfulness("Bananas are delicious tropical fruit grown abroad [pwd-03].", by, ["pwd-03"]) == 0.0
    assert faithfulness("Something claimed here.", by, []) == 0.0
    assert faithfulness(REFUSAL, by, []) == 1.0 and faithfulness(REFUSAL_SV, by, []) == 1.0

def test_pipeline_mock_traces_and_refusal(tmp_path):
    t = tmp_path / "tr.jsonl"
    rr = Rerank(Hybrid(BM25(DOCS), dense), DOCS, lambda q, texts: [len(set(tokens(q)) & set(tokens(x))) - 2 for x in texts])
    p = RAGPipeline(rr, DOCS, MockLLM(), min_score=0, trace_path=t)
    recs = [p.answer(q["query"], q["qid"]) for q in QS]
    lines = [json.loads(l) for l in t.read_text().splitlines()]; assert len(lines) == 30
    assert all(r["citations_valid"] for r in recs)
    assert {"latency_s", "prompt_tokens_est", "cost_usd", "faithfulness"} <= set(lines[0])
    s = score_e2e(recs, QS); assert s["citation_valid"] == 1.0 and 0 <= s["answer_accuracy"] <= 1

def test_gate_refuses_without_calling_llm():
    class Boom:
        def complete(self, p): raise AssertionError("LLM called")
    p = RAGPipeline(BM25(DOCS), DOCS, Boom(), min_score=1e9)
    r = p.answer("parking permit"); assert r["refused"] and r["gated_refusal"]

def test_invalid_citation_flagged():
    class Liar:
        model = "liar"
        def complete(self, p):
            from agent_trace_evals.rag.llm import LLMResult; return LLMResult("Use the VPN [sec-05].", 0, 1, 1)
    r = RAGPipeline(BM25(DOCS), DOCS, Liar(), k=1).answer("guest wifi Vasterby-Gast")
    assert not r["citations_valid"]

def test_est_tokens(): assert est_tokens("abcd" * 10) == 10


def test_python_numbers_locked_for_ts_parity():
    # same dicts are asserted in ts/test/rag.test.ts
    assert evaluate_retriever(BM25(DOCS), QS) == {"recall@1": 0.615, "recall@3": 0.769, "recall@5": 0.846, "mrr": 0.699, "ndcg@5": 0.736}
    assert evaluate_retriever(dense, QS) == {"recall@1": 0.885, "recall@3": 1.0, "recall@5": 1.0, "mrr": 0.936, "ndcg@5": 0.952}
    assert evaluate_retriever(Hybrid(BM25(DOCS), dense), QS) == {"recall@1": 0.769, "recall@3": 0.923, "recall@5": 0.962, "mrr": 0.862, "ndcg@5": 0.883}
    assert evaluate_retriever(WordOverlap(DOCS), QS) == {"recall@1": 0.385, "recall@3": 0.692, "recall@5": 0.769, "mrr": 0.55, "ndcg@5": 0.602}

def test_score_e2e_any_of_key_fact_and_gate_counts():
    q = [{"qid": "a", "gold": ["files-05"], "key_fact": ["Versionshistorik", "version history"]}, {"qid": "u", "gold": [], "key_fact": None}]
    base = dict(citations_valid=True, faithfulness=1.0, latency_s=1.0, prompt_tokens_est=1, completion_tokens_est=1, cost_usd=0.0)
    recs = [dict(base, qid="a", retrieved=["files-05"], refused=False, gated_refusal=False, answer="Use the version history [files-05].", citations=["files-05"]),
            dict(base, qid="u", retrieved=[], refused=True, gated_refusal=True, answer=REFUSAL, citations=[])]
    m = score_e2e(recs, q)
    assert m["answer_accuracy"] == 1.0 and m["gate_refusals"] == 1 and m["llm_refusals"] == 0 and m["latency_max_s"] == 1.0


def test_faithfulness_short_and_partial_refusal():
    by = {d["id"]: d for d in DOCS}
    assert faithfulness("Bananas! [pwd-03]", by, ["pwd-03"]) == 0.0
    assert sentences("Use the [SÄKER] tag. Done now [mail-05].") == ["Use the [SÄKER] tag.", "Done now [mail-05]."]
    assert sentences("First claim here. [pwd-03]") == ["First claim here. [pwd-03]"]
    assert is_refusal(REFUSAL) and is_refusal(f'"{REFUSAL.rstrip(".")}"') and is_refusal(REFUSAL_SV)
    partial = "I don't know the menu, but use GlobalConnect [vpn-01]."
    assert not is_refusal(partial) and faithfulness(partial, by, ["vpn-01"]) < 1.0


def _trace(name): return [json.loads(l) for l in (DATA / f"traces_opencode_{name}.jsonl").read_text().splitlines()]

def _row(m, keys=("answer_accuracy", "refusal_on_unanswerable", "false_refusal", "citation_valid", "faithfulness_mean", "gate_refusals", "llm_refusals")):
    return tuple(m[k] for k in keys)

README_NUMS = {  # README cells: latency p50 / p95 / max (s), estimated tokens
    "hybrid+rerank_gate-6.5": (4.92, 7.9, 17.07, 7843), "hybrid+rerank_nogate": (5.34, 13.38, 18.72, 8982),
    "jina_hybrid+rerank_gate-2.5": (5.22, 7.8, 8.98, 8203), "jina_hybrid+rerank_nogate": (5.07, 6.7, 27.83, 9036)}

def _recheck(recs):
    """Re-derive refused / citations_valid / faithfulness from answer, citations and retrieved with the current code."""
    by = {d["id"]: d for d in DOCS}
    for r in recs:
        cited, ret = r["citations"], r["retrieved"]
        assert r["refused"] == is_refusal(r["answer"])
        assert r["citations_valid"] == (all(c in ret for c in cited) and (bool(cited) or is_refusal(r["answer"])))
        assert r["faithfulness"] == round(faithfulness(r["answer"], by, cited), 3)

def test_published_traces_recomputed_and_latency_tokens():
    for name, nums in README_NUMS.items():
        tr = _trace(name); _recheck(tr); m = score_e2e(tr, QS)
        assert (m["latency_p50_s"], m["latency_p95_s"], m["latency_max_s"], m["tokens_est_total"]) == nums

def test_published_traces_match_readme():
    # headline: bge-reranker-base; comparison: jina-reranker-v2 (README rows)
    assert _row(score_e2e(_trace("hybrid+rerank_gate-6.5"), QS)) == (1.0, 1.0, 0.0, 1.0, 0.885, 4, 0)
    assert _row(score_e2e(_trace("hybrid+rerank_nogate"), QS)) == (1.0, 1.0, 0.0, 1.0, 0.846, 0, 4)
    assert _row(score_e2e(_trace("jina_hybrid+rerank_gate-2.5"), QS)) == (1.0, 1.0, 0.0, 1.0, 0.865, 3, 1)
    assert _row(score_e2e(_trace("jina_hybrid+rerank_nogate"), QS)) == (1.0, 1.0, 0.0, 1.0, 0.827, 0, 4)
    gold = {q["qid"]: q["gold"] for q in QS}
    for name, gate in [("hybrid+rerank_gate-6.5", -6.5), ("jina_hybrid+rerank_gate-2.5", -2.5)]:
        tr = _trace(name); lo = min(r["top_score"] for r in tr if gold[r["qid"]])
        assert lo > max(r["top_score"] for r in tr if not gold[r["qid"]]) and lo > gate  # gate never refuses an answerable query here

def test_key_fact_whole_phrase_and_cli_header_strip():
    from agent_trace_evals.rag.pipeline import has_fact
    from agent_trace_evals.rag.llm import HEADER
    assert has_fact("Enable online archiving [mail-03].", "online archiving") and not has_fact("Use the archive.", "archiv")
    assert has_fact("Mark it [SÄKER] please", "[SÄKER]")
    assert HEADER.match("> build · deepseek-v4.1-flash-free") and not HEADER.match("> Quoted policy line from the doc")

class _Recorded:
    """Replays recorded rerank rankings (rag_data/rerank_rankings.json) so the rerank rows are tested without model downloads."""
    def __init__(self, ranks): self.by_q = {q["query"]: ranks[q["qid"]] for q in QS}
    def search(self, q, k=10): return [(i, s) for i, s in self.by_q[q]][:k]

def test_rerank_rows_match_readme():
    R = json.loads((DATA / "rerank_rankings.json").read_text())["rankings"]
    assert evaluate_retriever(_Recorded(R["BAAI/bge-reranker-base"]), QS) == {"recall@1": 0.846, "recall@3": 1.0, "recall@5": 1.0, "mrr": 0.917, "ndcg@5": 0.938}
    assert evaluate_retriever(_Recorded(R["jinaai/jina-reranker-v2-base-multilingual"]), QS) == {"recall@1": 0.962, "recall@3": 1.0, "recall@5": 1.0, "mrr": 0.981, "ndcg@5": 0.986}

def test_mock_row_replay():
    R = json.loads((DATA / "rerank_rankings.json").read_text())["rankings"]["BAAI/bge-reranker-base"]
    p = RAGPipeline(_Recorded(R), DOCS, MockLLM(), k=3, min_score=-6.5)
    m = score_e2e([p.answer(q["query"], q["qid"]) for q in QS], QS)
    assert (m["answer_accuracy"], m["refusal_on_unanswerable"], m["gate_refusals"], m["faithfulness_mean"]) == (0.577, 1.0, 4, 1.0)
    assert (m["tokens_est_total"], m["citation_valid"], m["false_refusal"]) == (6732, 1.0, 0.0)

def test_gate_trace_matches_recorded_rerank_rankings():
    R = json.loads((DATA / "rerank_rankings.json").read_text())["rankings"]["BAAI/bge-reranker-base"]
    for r in _trace("hybrid+rerank_gate-6.5"):
        assert r["gated_refusal"] == (r["top_score"] < -6.5)
        doc, sc = R[r["qid"]][0]; assert r["retrieved"][0] == doc and r["top_score"] == round(sc, 4)
