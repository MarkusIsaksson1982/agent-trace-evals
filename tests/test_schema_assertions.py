import unittest
import json
from pathlib import Path

from agent_trace_evals.assertions import (abstain_or_valid, answer_is_json,
    answer_matches_schema, _json_answer)
from agent_trace_evals.agent import _toy_llm
from agent_trace_evals.schema import validate
from agent_trace_evals.trace import Trace


def trace(answer): return Trace("test", 0, 0, final_answer=answer)


class SchemaTests(unittest.TestCase):
    def test_valid_and_missing_required(self):
        schema = {"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}}}
        self.assertEqual(validate({"name": "Ada"}, schema), [])
        self.assertIn("$.name", validate({}, schema)[0])

    def test_wrong_type_and_bool_as_int(self):
        self.assertTrue(validate("x", {"type": "integer"}))
        self.assertTrue(validate(True, {"type": "number"}))
        self.assertTrue(validate(True, {"type": "integer"}))
        self.assertTrue(validate(1.0, {"type": "integer"}))
        self.assertFalse(validate(1, {"type": "number"}))

    def test_enum_and_additional_properties(self):
        schema = {"type": "object", "properties": {"kind": {"type": "string", "enum": ["a"]}}, "additionalProperties": False}
        self.assertTrue(validate({"kind": "b"}, schema))
        self.assertTrue(validate({"extra": 1}, schema))

    def test_nested_array_items(self):
        schema = {"type": "array", "items": {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}}
        self.assertTrue(any("$[0].n" in error for error in validate([{"n": "x"}], schema)))

    def test_unknown_keyword(self):
        with self.assertRaises(ValueError): validate({}, {"type": "object", "wat": True})

    def test_enum_must_be_a_list(self):
        with self.assertRaises(ValueError): validate("a", {"enum": "a"})

    def test_strict_enum_and_lengths(self):
        self.assertTrue(validate(True, {"enum": [1]}))
        self.assertTrue(validate("x", {"minLength": 2}))
        self.assertTrue(validate("xxx", {"maxLength": 2}))
        self.assertEqual(validate({"x": 1}, {"type": "object"}), [])
        with self.assertRaises(ValueError): validate({}, {"required": ["x"], "properties": {}, "additionalProperties": False})


class StructuredAssertionTests(unittest.TestCase):
    schema = {"type": "object", "required": ["ok"], "properties": {"ok": {"type": "boolean"}}, "additionalProperties": False}

    def test_fenced_json(self):
        self.assertTrue(answer_is_json()(trace("```json\n{\"ok\": true}\n```"))[0])
        self.assertTrue(answer_matches_schema(self.schema)(trace("```json\n{\"ok\": true}\n```"))[0])

    def test_non_json(self): self.assertFalse(answer_is_json()(trace("not json"))[0])

    def test_abstain_marker_and_invalid_without_it(self):
        check = abstain_or_valid(self.schema)
        self.assertTrue(check(trace("REVIEW_REQUIRED: missing details"))[0])
        self.assertFalse(check(trace("This mentions REVIEW_REQUIRED in passing"))[0])
        self.assertFalse(check(trace('{"ok":'))[0])
        self.assertFalse(check(trace('{"other": 1}'))[0])
        self.assertFalse(check(trace("answer\nREVIEW_REQUIRED: later"))[0])
        self.assertFalse(check(trace("REVIEW_REQUIRED: {\"ok\": true}"))[0])
        self.assertFalse(check(trace(None))[0])

    def test_json_edge_cases_and_schema_errors(self):
        self.assertTrue(answer_is_json()(trace("null"))[0])
        self.assertFalse(answer_is_json()(trace('{"ok": true, "ok": false}'))[0])
        self.assertFalse(answer_is_json()(trace("```\n{}\n```"))[0])
        self.assertFalse(answer_matches_schema({"description": "no"})(trace("{}"))[0])

    def test_toy_llm(self):
        self.assertIn('"date":"2027-01-02"', _toy_llm("book an appointment 2027-01-02 morning", []))
        self.assertTrue(_toy_llm("change my meeting 2027-01-02 morning", []).startswith("REVIEW_REQUIRED"))
        self.assertNotIn('"intent"', _toy_llm("meeting room cost", []))

    def test_structured_case_loads(self):
        data = json.loads(Path("cases/structured_output.json").read_text())
        self.assertEqual(len(data["cases"]), 3)


if __name__ == "__main__": unittest.main()
