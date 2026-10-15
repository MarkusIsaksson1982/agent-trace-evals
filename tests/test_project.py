import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from agent_trace_evals.agent import run_agent
from agent_trace_evals.assertions import (answer_contains, citations_subset_of_retrieved,
    max_cost_usd, max_latency_ms, no_span_errors, retrieval_before_answer, tool_called,
    tool_not_called, tool_order)
from agent_trace_evals.runner import evaluate_traces, wilson_lower_bound
from agent_trace_evals.trace import Span, Trace, read_jsonl, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
KB = json.loads((ROOT / "kb/support_kb.json").read_text())


class AssertionTests(unittest.TestCase):
    def setUp(self):
        self.trace = Trace("x", 0, 0, [Span("search", "tool", start_ms=0, end_ms=1),
             Span("retriever", "retrieval", output=["refunds"], start_ms=1, end_ms=2),
             Span("answer", "llm", output="Refunds available [doc:refunds]", start_ms=2, end_ms=4)],
             "Refunds available [doc:refunds]")

    def test_tool_called(self): self.assertTrue(tool_called("search")(self.trace)[0])
    def test_tool_missing(self): self.assertFalse(tool_called("calculator")(self.trace)[0])
    def test_tool_not_called(self): self.assertTrue(tool_not_called("calculator")(self.trace)[0])
    def test_tool_order(self): self.assertTrue(tool_order(["search"])(self.trace)[0])
    def test_retrieval_order(self): self.assertTrue(retrieval_before_answer()(self.trace)[0])
    def test_citations(self): self.assertTrue(citations_subset_of_retrieved()(self.trace)[0])
    def test_answer_contains(self): self.assertTrue(answer_contains("refunds")(self.trace)[0])
    def test_latency(self): self.assertTrue(max_latency_ms(4)(self.trace)[0])
    def test_cost(self): self.assertTrue(max_cost_usd(0)(self.trace)[0])
    def test_errors(self): self.assertTrue(no_span_errors()(self.trace)[0])


class RunnerTests(unittest.TestCase):
    def test_wilson_bounds(self):
        self.assertAlmostEqual(wilson_lower_bound(10, 10), 0.72246, places=4)
        self.assertEqual(wilson_lower_bound(0, 0), 0)

    def test_pass_fail_threshold(self):
        cases = [{"id":"x", "question":"x", "assertions":[{"type":"answer_contains","args":["yes"]}], "min_pass_rate":0.5}]
        traces = [Trace("x", i, i, final_answer="yes" if i < 3 else "no") for i in range(4)]
        summary = evaluate_traces(cases, traces)["cases"][0]
        self.assertEqual(summary["pass_rate"], 0.75)
        self.assertTrue(summary["passed"])
        self.assertEqual(summary["assertion_failures"], {"answer_contains": 1})

    def test_flakiness_zero_and_high(self):
        clean = run_agent("Can I get a refund?", KB, 0, 0)
        self.assertTrue(citations_subset_of_retrieved()(clean)[0])
        case = {"id":"c", "question":"What is 17 + 25?", "assertions":[{"type":"tool_called","args":["calculator"]}]}
        traces = [run_agent(case["question"], KB, i, 1, case_id="c") for i in range(30)]
        summary = evaluate_traces([case], traces)["cases"][0]
        self.assertGreater(summary["assertion_failures"].get("tool_called", 0), 0)

    def test_jsonl_round_trip(self):
        trace = run_agent("What is 2 + 3?", KB, 7)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "traces.jsonl"
            write_jsonl([trace], path)
            self.assertEqual(read_jsonl(path), [trace])

    def test_cli_exit_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = subprocess.run([sys.executable, "-m", "agent_trace_evals", "run", str(ROOT / "cases/support.json"), "--runs", "2", "--out", tmp], cwd=ROOT)
            self.assertEqual(good.returncode, 0)
            cases = Path(tmp) / "fail.json"
            cases.write_text(json.dumps([{"id":"bad", "question":"What is 2 + 2?", "assertions":[{"type":"answer_contains","args":["never"]}]}]))
            bad = subprocess.run([sys.executable, "-m", "agent_trace_evals", "run", str(cases), "--runs", "1", "--out", str(Path(tmp)/"badout")], cwd=ROOT)
            self.assertEqual(bad.returncode, 1)


if __name__ == "__main__": unittest.main()
