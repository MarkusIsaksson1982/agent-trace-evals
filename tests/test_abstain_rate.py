import json
import math
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_trace_evals.assertions import is_abstention
from agent_trace_evals.runner import (Baseline, BaselineError, compare_to_baseline,
                                      evaluate_traces, format_report, load_baseline,
                                      load_cases, write_baseline)
from agent_trace_evals.trace import Trace


class AbstainRateTests(unittest.TestCase):
    def test_marker_rule(self):
        self.assertTrue(is_abstention("REVIEW_REQUIRED: please review"))
        self.assertTrue(is_abstention("  REVIEW_REQUIRED"))
        self.assertFalse(is_abstention("REVIEW_REQUIRED: {\"ok\": true}"))
        self.assertFalse(is_abstention("please REVIEW_REQUIRED"))
        self.assertTrue(is_abstention("HUMAN: please review", "HUMAN"))
        self.assertFalse(is_abstention("REVIEW_REQUIRED: [not an abstention]"))
        self.assertFalse(is_abstention("REVIEW_REQUIRED: ```json\n{}\n```"))
        self.assertFalse(is_abstention(None))

    def test_custom_marker_and_abstention_is_independent_of_pass(self):
        cases = [{"id": "c", "question": "q", "abstain_marker": "ESCALATE",
                  "assertions": [{"type": "answer_contains", "args": ["ok"]}]}]
        traces = [Trace("c", 0, 0, final_answer="ESCALATE: help"),
                  Trace("c", 1, 1, final_answer="ok")]
        row = evaluate_traces(cases, traces)["cases"][0]
        self.assertEqual((row["abstained_runs"], row["passed_runs"]), (1, 1))

    def test_zero_run_case(self):
        row = evaluate_traces([{"id": "c", "question": "q"}], [])["cases"][0]
        self.assertEqual(row["runs"], 0)
        self.assertIsNone(row["abstain_rate"])
        self.assertFalse(row["passed"])

    def test_summary_and_report(self):
        cases = [{"id": "c", "question": "q", "assertions": [{"type": "answer_contains", "args": ["ok"]}]}]
        traces = [Trace("c", 0, 0, final_answer="REVIEW_REQUIRED: help"), Trace("c", 1, 1, final_answer="no")]
        summary = evaluate_traces(cases, traces)
        row = summary["cases"][0]
        self.assertEqual((row["abstained_runs"], row["abstain_rate"]), (1, 0.5))
        self.assertEqual(summary["abstain_rate"], 0.5)
        self.assertIn("Review rate", format_report(summary))
        self.assertIn("Overall human-review rate: 50.0%", format_report(summary))

    def test_baseline_and_gate(self):
        summary = {"cases": [{"id": "c", "pass_rate": 1.0, "abstain_rate": 0.4}]}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.json"
            write_baseline(summary, path)
            baseline = load_baseline(path)
            self.assertEqual(baseline.abstain_rates, {"c": 0.4})
        current = {"cases": [{"id": "c", "pass_rate": 1.0, "abstain_rate": 0.5}]}
        self.assertFalse(compare_to_baseline(current, Baseline({"c": 1.0}, abstain_rates={"c": 0.4}),
                                             max_abstain_drift=0.05)["passed"])
        self.assertTrue(compare_to_baseline(current, Baseline({"c": 1.0}, abstain_rates={"c": 0.4}),
                                            max_abstain_drift=0.1)["passed"])

    def test_old_baseline_and_missing_rate_gate(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.json"
            path.write_text(json.dumps({"cases": {"c": 1.0}}), encoding="utf-8")
            baseline = load_baseline(path)
        self.assertIsNone(baseline.abstain_rates)
        summary = {"cases": [{"id": "c", "pass_rate": 1.0, "abstain_rate": 0.2}]}
        with self.assertRaises(BaselineError):
            compare_to_baseline(summary, baseline, max_abstain_drift=0.1)
        gate = compare_to_baseline(summary, baseline, max_abstain_drift=0.1,
                                   allow_missing_abstain_rates=True)
        self.assertTrue(gate["passed"])
        self.assertTrue(gate["abstain_gate_skipped"])

    def test_downward_drift_and_exact_limit(self):
        baseline = Baseline({"c": 1.0}, abstain_rates={"c": 0.8})
        lower = {"cases": [{"id": "c", "pass_rate": 1.0, "abstain_rate": 0.5}]}
        self.assertFalse(compare_to_baseline(lower, baseline, max_abstain_drift=0.2)["passed"])
        exact = {"cases": [{"id": "c", "pass_rate": 1.0, "abstain_rate": 0.6}]}
        self.assertTrue(compare_to_baseline(exact, baseline, max_abstain_drift=0.2)["passed"])

    def test_invalid_baseline_abstain_rates(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.json"
            for value in ("bad", -0.1, 1.1, True, float("nan")):
                path.write_text(json.dumps({"cases": {"c": 1.0},
                                            "abstain_rates": {"c": value}}), encoding="utf-8")
                with self.assertRaises(BaselineError):
                    load_baseline(path)

    def test_invalid_max_abstain_drift(self):
        baseline = Baseline({"c": 1.0}, abstain_rates={"c": 0.0})
        summary = {"cases": [{"id": "c", "pass_rate": 1.0, "abstain_rate": 0.0}]}
        for value in (-0.1, 1.1, float("nan"), True):
            with self.assertRaises(ValueError):
                compare_to_baseline(summary, baseline, max_abstain_drift=value)

    def test_load_cases_rejects_invalid_marker(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            for marker in ("", 3, None):
                path.write_text(json.dumps([{"id": "c", "question": "q",
                                             "abstain_marker": marker}]), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_cases(path)

    def test_cli_abstain_drift_exit_code_and_warning(self):
        root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            cases = directory / "cases.json"
            traces = directory / "traces.jsonl"
            baseline = directory / "baseline.json"
            cases.write_text(json.dumps([{"id": "c", "question": "q"}]), encoding="utf-8")
            traces.write_text(json.dumps({"case_id": "c", "run_idx": 0, "seed": 0,
                                          "spans": [], "final_answer": "REVIEW_REQUIRED: help"}) + "\n",
                                encoding="utf-8")
            baseline.write_text(json.dumps({"cases": {"c": 1.0},
                                            "abstain_rates": {"c": 0.0}}), encoding="utf-8")
            result = subprocess.run([sys.executable, "-m", "agent_trace_evals", "report",
                                     str(traces), "--cases", str(cases), "--baseline", str(baseline),
                                     "--max-abstain-drift", "0.1", "--out", str(directory / "report.md")],
                                    cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 3)
            warning = subprocess.run([sys.executable, "-m", "agent_trace_evals", "report",
                                      str(traces), "--cases", str(cases), "--max-abstain-drift", "0.1",
                                      "--out", str(directory / "report-no-baseline.md")],
                                     cwd=root, capture_output=True, text=True)
            self.assertIn("warning: --max-abstain-drift has no effect without --baseline", warning.stderr)


if __name__ == "__main__":
    unittest.main()
