# agent-trace-evals

A compact, offline Python harness for evaluating tool-using agents through execution traces. The core harness uses only the Python 3.11+ standard library; the optional RAG module (see [RAG](#rag)) adds numpy and fastembed.

## Why stochastic and trace-based?

Agent runs can vary even when the prompt stays the same. Repeating each case with several seeds measures its pass rate and Wilson 95% lower bound, while a configurable threshold avoids treating one lucky run as success. Assertions inspect the trace: which tools ran, whether retrieval came before the answer, whether citations refer to retrieved documents, latency, cost, and span errors. This helps explain *how* a result was produced, alongside checking the final text.

See [PLAN.md](PLAN.md) for the short update plan and [CHANGELOG.md](CHANGELOG.md) for changes.

Recorded clean and flaky runs of a synthetic booking/rescheduling case set are in [examples/booking](examples/booking/). They exercise the harness with the built-in toy agent; they are not a model benchmark.

## Quickstart

```sh
python -m agent_trace_evals run cases/support.json --runs 20 --flakiness 0.1 --out out/
python -m agent_trace_evals report out/traces.jsonl
python -m unittest discover -s tests
```

The run command writes `out/traces.jsonl` and `out/report.md`. The report command evaluates saved traces again using the cases file; pass `--cases path/to/cases.json` to select another file.

## Structured output checks

Use the standard-library JSON checks with a small schema subset. Supported keywords are `type`, `properties`, `required`, `enum`, `items`, `additionalProperties`, `minLength`, and `maxLength`; other keywords are rejected. Omitted `additionalProperties` allows extra fields, while `false` rejects them. Type and enum comparisons are strict (`bool` is not an integer). `abstain_or_valid` accepts a case-sensitive `REVIEW_REQUIRED` marker only when it is the whole answer on the first line; otherwise it parses and schema-checks JSON. A JSON fence must be the entire answer and exactly tagged `json`.

```python
schema = {"type": "object", "required": ["intent"],
          "properties": {"intent": {"type": "string"}},
          "additionalProperties": False}
assertions = [{"type": "answer_matches_schema", "args": {"schema": schema}}]
```

## PII-leak guard and trace redaction

Cases can declare synthetic sensitive fields and guard the final answer:

```json
{
  "id": "synthetic-member-request",
  "question": "What is the refund policy for Testa Exempelsson, member SYN-ABCD-0001?",
  "sensitive": {"name": "Testa Exempelsson", "member_id": "SYN-ABCD-0001"},
  "assertions": [{"type": "no_sensitive_echo"}]
}
```

The guard uses case-insensitive, token-bounded matching after ignoring whitespace and `.,_/'-` separators; values must contain at least three kept characters, and any single part of a value with three or more characters (for example a surname alone) also counts as an echo. `redact_trace` rewrites span `name`, `input`, `output`, and `error` (including nested containers), plus `final_answer`, before traces are written. It preserves `case_id` and span `kind`; redaction covers the whole trace, while the guard evaluates only the final answer. Placeholders are still flagged when saved traces are re-evaluated. This checks declared fields only; it is not general PII detection.

## Sample report

| Case | Runs | Passed | Pass rate | Wilson 95% lower | Threshold | Result |
|---|---:|---:|---:|---:|---:|---|
| refund-policy | 20 | 20 | 100.0% | 83.9% | 80.0% | PASS |

### With a flaky agent (`--flakiness 0.4`, exit code 1)

| Case | Runs | Passed | Pass rate | Wilson 95% lower | Threshold | Result |
|---|---:|---:|---:|---:|---:|---|
| refund-policy | 20 | 14 | 70.0% | 48.1% | 80.0% | FAIL |
| calculator | 20 | 17 | 85.0% | 64.0% | 80.0% | PASS |
| unknown | 20 | 17 | 85.0% | 64.0% | 80.0% | PASS |

Assertion failures: `refund-policy`: `answer_contains` 3, `citations_subset_of_retrieved` 3 · `calculator`: `tool_called` 3, `tool_order` 3, `no_span_errors` 3.
Each failure is attributed to the assertion that caught it, so you see *what* the agent did wrong (skipped retrieval, wrong tool, hallucinated citation), not only that the answer was off.

## Regression gate against a baseline

A single PASS/FAIL against a fixed threshold hides slow drift: a case can fall from 100% to 85% and still pass an 80% threshold. The regression gate stores per-case pass rates from a known-good run and fails CI when any case drops by more than `--max-drop` (default 0.05) relative to that baseline.

```sh
# 1. record a baseline from a good run (refused if the evaluation itself fails, unless --force-baseline)
python -m agent_trace_evals run cases/support.json --runs 20 --out out/ --write-baseline baselines/support.json
# 2. in CI: same cases, compare against the baseline
python -m agent_trace_evals run cases/support.json --runs 20 --out out/ --baseline baselines/support.json --max-drop 0.05
# or re-check saved traces without re-running the agent
python -m agent_trace_evals report out/traces.jsonl --baseline baselines/support.json
```

Example with a degraded agent (`--flakiness 0.3`): every case still clears the 80% threshold, so the evaluation passes, but the gate fails:

```text
Evaluation PASS; report: out/report.md
REGRESSION refund-policy: 100.0% -> 80.0% (drop 20.0%, allowed 5.0%)
REGRESSION calculator: 100.0% -> 90.0% (drop 10.0%, allowed 5.0%)
...
Regression gate FAIL
```

Exit codes: `0` pass, `1` evaluation failed, `2` baseline missing or malformed, `3` regression (or a baseline case missing from the run; `--allow-missing-cases` relaxes that). The baseline is a small versioned JSON file (`{"version": 1, "runs": 20, "flakiness": 0.0, "cases": {"refund-policy": 1.0, ...}}`), written atomically; the gate warns when `--runs` or `--flakiness` differ from the baseline run, because pass rates are then not directly comparable (for `report`, the run count is taken from the traces, and unequal per-case counts are warned about). A drop exactly at `--max-drop` passes (exact decimal comparison, no float slack). New cases not in the baseline are reported and skipped.

## Plug in a real model

`run_agent` accepts an `llm` callable with signature `llm(question, retrieved_documents, calculation) -> str`. It receives the question, a list of `(document_id, text)` pairs, and an optional calculator result. Replace the default toy callable at the integration point in your own runner. Keep secrets and network-specific model setup outside this dependency-free example.

## Architecture extensions (v0.2)

Two small, dependency-free modules (stdlib Python / plain TS) for the parts of an LLM/agent system an architect has to measure, not just build.

### RAG evaluation baseline (`agent_trace_evals/rag_eval.py`, stdlib Python)
A dependency-free, lexical-only baseline of the evaluation helpers. The full RAG module with dense retrieval, rerank and an LLM is in [RAG](#rag) below.

| function | what it measures |
|---|---|
| `chunk(text, size, overlap)` | word-window chunking with overlap (rejects overlap >= size) |
| `retrieve(query, docs, k)` | baseline lexical retrieval (length-normalised token overlap); swap in embeddings at this point |
| `recall_at_k`, `mrr` | did the right chunk come back, and how high |
| `pack_context(hits, budget_chars)` | greedy context packing under a size budget, in rank order |
| `grounding(answer, retrieved_ids)` | `[id]` citations in the answer must all refer to retrieved chunks |
| `evaluate(cases, docs, k)` | mean recall@k and MRR over a labelled case set |

The point is the evaluation loop: change the chunk size, the retriever or the packing budget, and the same case set tells you whether retrieval got better or worse before any model is involved.

### Tool routing with budgets and fallbacks (`ts/src/router.ts`, strict TypeScript)
`rankTools` orders tools by keyword match, then cost. `route` tries them in order inside a cost and latency budget, records every attempt, falls through a failing tool (`degraded: true`) and returns a safe fallback answer with the last error instead of crashing. `tsc --strict`, 7 `node:test` tests in `ts/test/router.test.ts` (`cd ts && npm test` runs these plus the 8 RAG port tests).

Both modules are deliberately small baselines: `rag_eval.py` retrieval is lexical, not semantic (the `rag/` module adds that), and the costs and latencies are declared per tool, not measured from a provider.

## Limitations

The included agent and knowledge base are deliberately small; the agent's keyword overlap is not semantic search, and the simulated cost and latency are illustrative. The Wilson interval describes run-level pass frequency, not model quality in general. A real integration should record provider-reported usage and handle privacy, retries, and tool permissions explicitly.

Built by Markus Isaksson with an agentic workflow (spec by Grok Bot, implementation by a coding agent, reviewed and tested before publishing). Tests: 126 Python tests under pytest (the 21 core tests also run with `python -m unittest discover -s tests`; the RAG, report and regression-gate tests need pytest, the RAG module tests also numpy) and 15 TypeScript `node:test` tests. No test uses the network or downloads a model.

License: code MIT (`LICENSE`); the synthetic RAG corpus and queries in `rag_data/` are CC0; the recorded traces contain reranker scores and model output, see `rag_data/LICENSE-DATA`.


## RAG

A small but complete retrieval-augmented generation stack (`agent_trace_evals/rag/`, data in `rag_data/`, TS port in `ts/src/rag.ts`).
**Scope, honestly:** a 64-document synthetic corpus, 30 labelled queries, CPU-only models, a single free LLM. It shows the engineering
pattern (baselines → dense → hybrid → rerank, gated refusal, citation and faithfulness checks, JSONL traces, tests). It is not a large-scale benchmark.

- **Corpus**: 64 short Swedish/English IT service-desk documents for a fictional municipality ("Västerby kommun"), written for this repo
  (CC0, see `rag_data/LICENSE-DATA`; source in `rag_data/corpus_src.py`, which regenerates the JSON files). 30 queries with gold doc ids, 4 unanswerable.
  Queries were written to avoid copying the doc wording, but several still share key terms with their gold doc (that is why BM25 is not near zero).
  Each answerable query has one or more distinctive key facts from the gold doc (any-of, case-insensitive) for answer scoring.
- **Retrieval**: word overlap and BM25 baselines; dense cosine over a numpy matrix of `paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0) vectors
  (fastembed/ONNX, precomputed into `rag_data/embeddings.json`); hybrid via reciprocal rank fusion (k=60); cross-encoder rerank of the top 15
  with `BAAI/bge-reranker-base` (MIT, CPU) by default. `jina-reranker-v2-base-multilingual` scores higher on this set but is licensed
  CC-BY-NC-4.0 (non-commercial), so it is shown only as a comparison row (`--reranker jinaai/jina-reranker-v2-base-multilingual`).
- **Pipeline**: retrieve top 3 → refusal gate on rerank score → pack context (char budget) → LLM (pluggable: `MockLLM` deterministic for tests,
  `OpenCodeLLM` = free OpenCode model, instructed in the prompt not to use tools, run from an empty temp dir) → citation check (cited ids ⊆ retrieved) → faithfulness
  (share of answer sentences whose tokens are ≥50% covered by a cited chunk) → JSONL trace per query (latency, estimated tokens, cost field; cost is not metered and is $0 because the model used is free).

### Retrieval (26 answerable queries)

| retriever | recall@1 | recall@3 | recall@5 | MRR | nDCG@5 | ms/query |
|---|---|---|---|---|---|---|
| word overlap | 0.385 | 0.692 | 0.769 | 0.550 | 0.602 | <0.1 |
| BM25 | 0.615 | 0.769 | 0.846 | 0.699 | 0.736 | 0.1 |
| dense (MiniLM-L12 multilingual) | 0.885 | 1.000 | 1.000 | 0.936 | 0.952 | 0.1* |
| hybrid (RRF BM25+dense) | 0.769 | 0.923 | 0.962 | 0.862 | 0.883 | 0.2* |
| **hybrid + rerank, `bge-reranker-base` (default, MIT)** | 0.846 | 1.000 | 1.000 | 0.917 | 0.938 | ~220 |
| hybrid + rerank, `jina-reranker-v2-base-multilingual` (comparison, CC-BY-NC-4.0) | 0.962 | 1.000 | 1.000 | 0.981 | 0.986 | ~360 |

\* query vectors precomputed; live query embedding adds ~10 ms on CPU. With 26 queries one query moves recall@1 by 0.038, so treat small
differences as noise. Hybrid scores below dense here because BM25 is weak on this bilingual query set. The bge reranker lifts hybrid (MRR 0.862 → 0.917) but does
not beat dense alone (0.936) on these short bilingual documents; the multilingual jina model does (0.981) but cannot be used commercially.
Rerank still matters in the pipeline because its score drives the refusal gate. Also tried: `bge-reranker-v2-m3-int8` (hurt), `ms-marco-MiniLM-L-12` (English-only).
All settings were chosen on these queries; there is no held-out set.

### End to end (30 queries, hybrid + bge rerank unless noted, k=3, `opencode/deepseek-v4.1-flash-free`, **one run each, n=1**)

| config | answer accuracy† | refusal on unanswerable (gate / LLM) | false refusal | citations valid | faithfulness (lexical) | latency p50 / p95 / max | est. tokens | cost |
|---|---|---|---|---|---|---|---|---|
| mock LLM (deterministic, copies the first sentence of the top chunk; a pipeline check, not a model), gate −6.5 | 0.577 | 1.00 (4 / 0) | 0.00 | 1.00 | 1.00 | 0.21 s / 0.29 s / 0.30 s | 6.7k | $0 |
| **free LLM + score gate (−6.5)** | **1.00** | **1.00 (4 / 0)** | 0.00 | 1.00 | **0.885** | 4.9 s / 7.9 s / 17.1 s | 7.8k | $0 |
| free LLM, no gate | 1.00 | 1.00 (0 / 4) | 0.00 | 1.00 | 0.846 | 5.3 s / 13.4 s / 18.7 s | 9.0k | $0 |
| comparison: jina rerank + gate (−2.5) | 1.00 | 1.00 (3 / 1) | 0.00 | 1.00 | 0.865 | 5.2 s / 7.8 s / 9.0 s | 8.2k | $0 |
| comparison: jina rerank, no gate | 1.00 | 1.00 (0 / 4) | 0.00 | 1.00 | 0.827 | 5.1 s / 6.7 s / 27.8 s | 9.0k | $0 |

† correct = not refused, contains one of the query's key facts, cites a gold doc that was also retrieved. A refusal is an answer that is exactly
the refusal sentence (English or Swedish), not any answer containing "don't know". Key facts accept a translation where the gold doc is
Swedish and the question English (q25 accepts "Versionshistorik" or "version history"); key facts match whole phrases only. An earlier, stricter
rule (one exact substring per query, q25 Swedish term only) gave answer accuracy 0.962 on the jina run. Some key facts are single
distinctive tokens (`wired`, `firmware`, `17:00`), so accuracy is a lenient check; per-query answers are in the traces.
**The gate:** with bge the weakest answerable query (q18) has a top rerank score of −5.74 and the strongest unanswerable (q27) −6.80;
the default −6.5 sits in that gap and gates all 4 unanswerable queries with no false refusals. With jina the gap is only 0.08 logits
((−2.022, −1.938]) and its −2.5 default sits below it, catching 3 of 4 (the LLM refuses q30). Both thresholds were picked on these same 30
queries, so this is not a held-out result; the gate's value here is saving LLM calls, not accuracy. Gated vs ungated faithfulness and latency
differences come from single stochastic runs and are not significant. Faithfulness is a lexical check (≥50% of a sentence's tokens covered by
a cited chunk) and counts paraphrased or translated sentences as unsupported, so read <1.0 as "needs a look", not as hallucination
(q15, q18, q25 in the bge gated run). Latency percentiles are nearest-rank over all 30 queries. Tokens are estimated at chars/4.
Traces: `rag_data/traces_opencode_hybrid+rerank_gate-6.5.jsonl` and `..._nogate.jsonl` (bge, default), `traces_opencode_jina_*.jsonl`
(comparison; `bench` names them with a `jina_` prefix; the jina no-gate run was recorded with `--min-score -99`, which never fires, and renamed). `bench` writes new traces to `out/`; a test re-scores
all four published traces and checks the accuracy, refusal, citation, faithfulness and gate-count columns above
and the latency and token cells. The rerank rows of the retrieval table are checked from recorded rankings
(`rag_data/rerank_rankings.json`), and the mock row is replayed in a test (all cells except its latency, which is illustrative).

### Run

```bash
pip install fastembed numpy pytest            # CPU only; models download on first use
python -m agent_trace_evals.rag.build_embeddings   # optional: rebuild rag_data/embeddings.json
python -m agent_trace_evals.rag.bench              # retrieval table + e2e with mock LLM
python -m agent_trace_evals.rag.bench --llm opencode --model opencode/deepseek-v4.1-flash-free            # bge, gate -6.5
python -m agent_trace_evals.rag.bench --llm opencode --model opencode/deepseek-v4.1-flash-free --no-gate
python -m agent_trace_evals.rag.bench --llm opencode --reranker jinaai/jina-reranker-v2-base-multilingual   # jina comparison (CC-BY-NC), gate -2.5
python -m pytest -q                                 # tests need no models (precomputed vectors, fake reranker)
cd ts && npm test                                   # TS port: BM25, cosine, RRF, metrics, citation check; asserts the same metric dicts as Python
```

### Hallucination / faithfulness report

`agent_trace_evals.rag.render_report` (stdlib only) turns per-question RAG results, such as the published traces, into a markdown table that makes hallucination signals visible: answers citing documents that were never retrieved (`cite_ok`), low lexical faithfulness, and refusals.

```bash
python -c "import json; from agent_trace_evals.rag import render_report; \
print(render_report([json.loads(l) for l in open('rag_data/traces_opencode_hybrid+rerank_gate-6.5.jsonl')]))"
```

```text
| qid | retrieved | citations | cite_ok | faithfulness | refused |
| --- | --- | --- | --- | --- | --- |
| q25 | files-05, files-02, files-03 | files-05 | ok | 0.0 | no |
| q27 | print-03, teams-04, sw-02 | - | n/a | n/a | yes |
...
- questions: 30
- refused: 4/30
- refusal_rate: 13.333%
- citation_valid (answered): 26/26
- faithfulness_mean (answered): 0.885
```

Refusals show `n/a` for citation and faithfulness and are excluded from the means. Malformed input fails loudly (missing or blank `qid`, faithfulness outside [0, 1], NaN, unknown refusal flags), and cells are escaped so a stray `|` or newline in an id cannot break the table. q25 above is a translated answer, which the lexical check scores as unsupported; read low values as "needs a look", not as proof of hallucination.

**TS port scope:** retrieval and metrics only (BM25, dense cosine over the precomputed vectors, RRF, recall/MRR/nDCG, citation check). Rerank, the gate,
context packing, faithfulness and the LLM call are Python-only.

**Not covered yet:** a real vector database (pgvector/sqlite-vec), chunking long documents in this corpus (the docs are short), an NLI or LLM-judge
faithfulness check, a held-out query set or human labels, streaming, online monitoring and dashboards, and multiple LLMs compared.
