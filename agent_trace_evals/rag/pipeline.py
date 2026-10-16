"""retrieve -> gate -> pack -> LLM -> citation check + faithfulness + refusal; JSONL traces."""
from __future__ import annotations
import json, math, re, time
from .retrieval import tokens, doc_text

REFUSAL = "I don't know based on the available documents."
REFUSAL_SV = "Jag vet inte utifrån de tillgängliga dokumenten."  # accepted if the model answers a Swedish question in Swedish
PROMPT = """You are an IT service desk assistant. Answer ONLY from the CONTEXT. Answer in the language of the question, 1-3 sentences.
Every sentence must end with a citation like [doc-id] naming the context chunk that supports it.
If the CONTEXT does not answer the question, reply exactly: {refusal}

CONTEXT:
{ctx}

QUESTION: {q}
ANSWER:"""
CITE = re.compile(r"\[([a-z]+-\d+)\]")

def pack(hits, by_id, budget_chars=1200):
    out, used = [], 0
    for i, _ in hits:
        t = f"[{i}] {doc_text(by_id[i])}"
        if used + len(t) > budget_chars: break
        out.append(t); used += len(t)
    return "\n".join(out)

def sentences(s):
    """Split on .!? only; a citation-only fragment ("[pwd-03]") is re-attached to the previous sentence, never scored alone."""
    out = []
    for x in (x.strip() for x in re.split(r"(?<=[.!?])\s+", s.strip())):
        if not x: continue
        if out and not tokens(CITE.sub("", x)): out[-1] += " " + x
        else: out.append(x)
    return out

def _canon(s): return re.sub(r"[\s.!?'\"“”‘’]+$", "", re.sub(r"^[\s'\"“”‘’]+", "", s)).lower()
def is_refusal(a):
    """Refusal = the whole answer is the refusal sentence (EN or SV), ignoring quotes/trailing punctuation. Not a substring test."""
    return bool(a and a.strip()) and _canon(a) in (_canon(REFUSAL), _canon(REFUSAL_SV))

def faithfulness(answer, by_id, cited, sim=None, thr=0.5):
    """Fraction of answer sentences supported by a cited chunk. Support = token recall vs chunk >= thr,
    or (if sim given) embedding cosine >= 0.75. Returns 1.0 for refusals/empty."""
    if is_refusal(answer) or not answer.strip(): return 1.0
    ss = sentences(answer)
    if not cited: return 0.0
    src = [doc_text(by_id[c]) for c in cited if c in by_id]; ok = 0
    for s in ss:
        st = set(tokens(CITE.sub("", s)))
        lex = max((len(st & set(tokens(c))) / len(st) for c in src), default=0) if st else 0
        if lex >= thr or (sim and max(sim(s, c) for c in src) >= 0.75): ok += 1
    return ok / len(ss)


class RAGPipeline:
    def __init__(self, retriever, docs, llm, k=3, min_score=None, sim=None, trace_path=None):
        self.r, self.by, self.llm, self.k, self.min_score, self.sim, self.trace = retriever, {d["id"]: d for d in docs}, llm, k, min_score, sim, trace_path
    def answer(self, q, qid=None):
        t0 = time.time(); hits = self.r.search(q, self.k); tr = time.time() - t0
        gated = self.min_score is not None and (not hits or hits[0][1] < self.min_score)
        ctx = pack(hits, self.by) if hits and not gated else ""
        if gated or not ctx:  # gate fired, no hits, or nothing fit the context budget: refuse without calling the LLM
            res = None; text = REFUSAL
        else:
            res = self.llm.complete(PROMPT.format(refusal=REFUSAL, ctx=ctx, q=q)); text = res.text
        cited = CITE.findall(text); ret_ids = [i for i, _ in hits]
        rec = {"qid": qid, "query": q, "retrieved": ret_ids, "top_score": round(hits[0][1], 4) if hits else None,
               "gated_refusal": gated, "answer": text, "citations": cited,
               "citations_valid": all(c in ret_ids for c in cited) and (bool(cited) or is_refusal(text)),
               "refused": is_refusal(text), "faithfulness": round(faithfulness(text, self.by, cited, self.sim), 3),
               "retrieval_ms": round(tr * 1000, 1), "llm_s": round(res.latency_s, 2) if res else 0.0,
               "prompt_tokens_est": res.prompt_tokens if res else 0, "completion_tokens_est": res.completion_tokens if res else 0,
               "cost_usd": res.cost_usd if res else 0.0, "model": res.model if res else ("gate" if gated else "no-context"), "latency_s": round(time.time() - t0, 2)}
        if self.trace:
            with open(self.trace, "a") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

def pct(vals, p):
    """Nearest-rank percentile of an ascending list: element ceil(p*n)-1."""
    return vals[max(0, min(len(vals) - 1, math.ceil(p * len(vals)) - 1))]

def has_fact(answer, fact):
    """Case-insensitive whole-phrase match (no partial words: 'archiv' does not match 'archiving')."""
    return re.search(rf"(?<!\w){re.escape(fact.lower())}(?!\w)", answer.lower()) is not None

def score_e2e(recs, queries):
    by = {q["qid"]: q for q in queries}; n = len(recs); ans = [r for r in recs if by[r["qid"]]["gold"]]; un = [r for r in recs if not by[r["qid"]]["gold"]]
    correct = sum(1 for r in ans if not r["refused"] and any(has_fact(r["answer"], f) for f in by[r["qid"]]["key_fact"])
                  and set(r["citations"]) & set(by[r["qid"]]["gold"]) & set(r["retrieved"]))
    lat = sorted(r["latency_s"] for r in recs)
    return {"n": n, "answer_accuracy": round(correct / len(ans), 3), "refusal_on_unanswerable": round(sum(r["refused"] for r in un) / len(un), 3),
            "false_refusal": round(sum(r["refused"] for r in ans) / len(ans), 3),
            "citation_valid": round(sum(r["citations_valid"] for r in recs) / n, 3),
            "faithfulness_mean": round(sum(r["faithfulness"] for r in ans if not r["refused"]) / max(1, sum(not r["refused"] for r in ans)), 3),
            "latency_p50_s": pct(lat, 0.5), "latency_p95_s": pct(lat, 0.95), "latency_max_s": lat[-1],
             "gate_refusals": sum(r["gated_refusal"] for r in recs), "llm_refusals": sum(r["refused"] and not r["gated_refusal"] for r in recs),
            "tokens_est_total": sum(r["prompt_tokens_est"] + r["completion_tokens_est"] for r in recs), "cost_usd_total": sum(r["cost_usd"] for r in recs)}
