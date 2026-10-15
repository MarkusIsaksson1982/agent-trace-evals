# agent-trace-evals

A compact, offline Python harness for evaluating tool-using agents through execution traces. It uses only the Python 3.11+ standard library.

## Why stochastic and trace-based?

Agent runs can vary even when the prompt stays the same. Repeating each case with several seeds measures its pass rate and Wilson 95% lower bound, while a configurable threshold avoids treating one lucky run as success. Assertions inspect the trace: which tools ran, whether retrieval came before the answer, whether citations refer to retrieved documents, latency, cost, and span errors. This helps explain *how* a result was produced, alongside checking the final text.

## Quickstart

```sh
python -m agent_trace_evals run cases/support.json --runs 20 --flakiness 0.1 --out out/
python -m agent_trace_evals report out/traces.jsonl
python -m unittest discover -s tests
```

The run command writes `out/traces.jsonl` and `out/report.md`. The report command evaluates saved traces again using the cases file; pass `--cases path/to/cases.json` to select another file.

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

## Plug in a real model

`run_agent` accepts an `llm` callable with signature `llm(question, retrieved_documents, calculation) -> str`. It receives the question, a list of `(document_id, text)` pairs, and an optional calculator result. Replace the default toy callable at the integration point in your own runner. Keep secrets and network-specific model setup outside this dependency-free example.

## Limitations

The included agent and knowledge base are deliberately small; keyword overlap is not semantic search, and the simulated cost and latency are illustrative. The Wilson interval describes run-level pass frequency, not model quality in general. A real integration should record provider-reported usage and handle privacy, retries, and tool permissions explicitly.

Built by Markus Isaksson with an agentic workflow (spec by Grok Bot, implementation by a coding agent, reviewed and tested before publishing). 15 unittest tests, no network.

License: MIT
