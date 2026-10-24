import json
import tempfile
import unittest
from pathlib import Path

from agent_trace_evals.assertions import no_sensitive_echo
from agent_trace_evals.runner import evaluate_traces, load_cases, report_from_traces, run_cases
from agent_trace_evals.trace import Span, Trace, redact_text, redact_trace, read_jsonl, write_jsonl


class SensitiveGuardTests(unittest.TestCase):
    fields = {"name": "Testa Exempelsson", "member": "SYN-ABCD-0001"}

    def test_matching_boundaries_and_normalized_separators(self):
        self.assertEqual(redact_text("Testa, Exempelsson SYN_ABCD_0001", self.fields, "[{field}]"), "[name] [member]")
        self.assertEqual(redact_text("Anniversary valid ab", {"a": "Ann"}, "[{field}]"), "Anniversary valid ab")
        with self.assertRaises(ValueError): redact_text("ab", {"x": "id"}, "[{field}]")
        with self.assertRaises(ValueError): redact_text(".", {"x": "."}, "[{field}]")

    def test_assertion_checks_answers_only(self):
        check = no_sensitive_echo(self.fields)
        self.assertFalse(check(Trace("x", 0, 0, final_answer="Exempelsson"))[0])
        self.assertTrue(check(Trace("x", 0, 0, [Span("x", "tool", input="Testa Exempelsson")]))[0])
        self.assertFalse(check(Trace("x", 0, 0, final_answer="[REDACTED:name]"))[0])
        self.assertTrue(check(Trace("x", 0, 0, final_answer="The request was handled safely."))[0])
        ok, message = check(Trace("x", 0, 0, final_answer="Testa"))
        self.assertFalse(ok)
        self.assertNotIn("Testa Exempelsson", message)
        self.assertFalse(check(Trace("x", 0, 0, final_answer="syn abcd 0001"))[0])

    def test_redaction_is_deep_and_non_mutating(self):
        original = Trace("Testa Exempelsson", 0, 0, [Span("Testa Exempelsson", "tool", {"Testa": "SYN-ABCD-0001"})], "Exempelsson")
        redacted = redact_trace(original, self.fields)
        self.assertEqual(redacted.case_id, original.case_id)
        self.assertEqual(redacted.spans[0].kind, "tool")
        self.assertEqual(original.final_answer, "Exempelsson")
        self.assertEqual(redacted.spans[0].input["Testa"], "[REDACTED:member]")
        self.assertEqual(redacted.spans[0].name, "[REDACTED:name]")

    def test_fixture_loads_and_jsonl_is_redacted(self):
        root = Path(__file__).resolve().parent.parent
        cases = load_cases(root / "cases/pii_guard.json")
        self.assertEqual(len(cases), 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "traces.jsonl"
            trace = Trace(cases[0]["id"], 0, 0, final_answer="Testa Exempelsson")
            write_jsonl([redact_trace(trace, cases[0]["sensitive"])], path)
            self.assertNotIn("Testa Exempelsson", path.read_text())
            self.assertFalse(evaluate_traces(cases[:1], [trace])["passed"])

    def test_malformed_sensitive_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.json"
            path.write_text(json.dumps({"cases": [{"id": "x", "question": "q", "sensitive": {"x": ""}}]}))
            with self.assertRaises(ValueError): load_cases(path)
            path.write_text(json.dumps({"cases": [{"id": "x", "question": "q", "sensitive": "bad"}]}))
            with self.assertRaises(ValueError): load_cases(path)

    def test_end_to_end_redacted_traces_still_evaluate(self):
        root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            summary = run_cases(root / "cases/pii_guard.json", runs=2, out=out)
            self.assertTrue(summary["passed"])
            saved = (out / "traces.jsonl").read_text().casefold()
            self.assertNotIn("testa exempelsson", saved)
            self.assertNotIn("syn-abcd-0001", saved)
            self.assertTrue(report_from_traces(out / "traces.jsonl", root / "cases/pii_guard.json")["passed"])
            traces = read_jsonl(out / "traces.jsonl")
            traces[0].final_answer = "[REDACTED:name]"
            bad = out / "bad.jsonl"
            write_jsonl([traces[0]], bad)
            self.assertFalse(report_from_traces(bad, root / "cases/pii_guard.json")["passed"])


if __name__ == "__main__": unittest.main()
