import unittest

from agent_trace_evals.agent import run_agent
from agent_trace_evals.runner import evaluate_traces, format_report, load_cases

CASES = load_cases("cases/booking_requests.json")


def _summary(flakiness: float, runs: int = 20):
    traces = [run_agent(c["question"], {}, seed, flakiness, case_id=c["id"], run_idx=seed)
              for c in CASES for seed in range(runs)]
    return evaluate_traces(CASES, traces)


class BookingCaseTests(unittest.TestCase):
    def test_clean_runs_pass_all_cases(self):
        summary = _summary(0.0)
        self.assertTrue(summary["passed"])
        self.assertTrue(all(row["pass_rate"] == 1.0 for row in summary["cases"]))

    def test_flakiness_breaks_booking_cases(self):
        summary = _summary(0.3)
        failing = {row["id"] for row in summary["cases"] if row["assertion_failures"]}
        self.assertIn("book-date-slot", failing)
        self.assertFalse(summary["passed"])

    def test_same_seed_same_result(self):
        self.assertEqual(_summary(0.3), _summary(0.3))

    def test_failures_attributed_in_report(self):
        report = format_report(_summary(0.3))
        self.assertIn("## Failures by assertion", report)
        self.assertIn("**book-date-slot**", report)


if __name__ == "__main__":
    unittest.main()
