"""Composable checks over execution traces."""
from __future__ import annotations

import re
import json
from typing import Callable

from .trace import Trace, redact_text
from .schema import validate

Assertion = Callable[[Trace], tuple[bool, str]]


def tool_called(name: str) -> Assertion:
    return lambda trace: ((name in [s.name for s in trace.spans if s.kind == "tool"]),
                          f"tool '{name}' {'was' if name in [s.name for s in trace.spans if s.kind == 'tool'] else 'was not'} called")


def tool_not_called(name: str) -> Assertion:
    check = tool_called(name)
    return lambda trace: (not check(trace)[0], f"tool '{name}' was not called" if not check(trace)[0] else f"tool '{name}' was called")


def tool_order(names: list[str]) -> Assertion:
    def check(trace: Trace) -> tuple[bool, str]:
        actual = [s.name for s in trace.spans if s.kind == "tool"]
        positions, offset = [], 0
        for name in names:
            try: offset = actual.index(name, offset) + 1; positions.append(name)
            except ValueError: return False, f"tool order {names} not found in {actual}"
        return True, f"tool order {names} found"
    return check


def retrieval_before_answer() -> Assertion:
    def check(trace: Trace) -> tuple[bool, str]:
        retrieval = next((i for i, s in enumerate(trace.spans) if s.kind == "retrieval"), None)
        answer = next((i for i, s in enumerate(trace.spans) if s.kind == "llm"), None)
        ok = retrieval is not None and answer is not None and retrieval < answer
        return ok, "retrieval precedes answer" if ok else "retrieval does not precede answer"
    return check


def citations_subset_of_retrieved() -> Assertion:
    def check(trace: Trace) -> tuple[bool, str]:
        retrieved = {str(item) for span in trace.spans if span.kind == "retrieval"
                     for item in (span.output if isinstance(span.output, list) else [])}
        citations = set(re.findall(r"\[doc:([\w-]+)\]", trace.final_answer))
        extra = sorted(citations - retrieved)
        return not extra, "citations are retrieved" if not extra else f"uncited documents: {', '.join(extra)}"
    return check


def answer_contains(text: str) -> Assertion:
    def check(trace: Trace) -> tuple[bool, str]:
        ok = text.casefold() in trace.final_answer.casefold()
        return ok, f"answer {'contains' if ok else 'does not contain'} {text!r}"
    return check


def no_sensitive_echo(fields: dict[str, str], placeholder: str = "[REDACTED:{field}]") -> Assertion:
    if not isinstance(fields, dict) or not fields or any(
            not isinstance(field, str) or not isinstance(value, str) or
            not value or not any(char not in " \t\r\n.,_/'-" for char in value)
            for field, value in fields.items()):
        raise ValueError("no_sensitive_echo fields must be a non-empty map of strings")
    def check(trace: Trace) -> tuple[bool, str]:
        leaked = []
        for field, value in fields.items():
            tokens = [token for token in re.split(r"[\s.,_/'-]+", value) if len(token) >= 3]
            candidates = [value] + tokens
            answer = trace.final_answer
            if any(redact_text(answer, {field: candidate}, placeholder) != answer
                   for candidate in candidates) or placeholder.format(field=field).casefold() in answer.casefold():
                leaked.append(field)
        return not leaked, "no sensitive fields echoed" if not leaked else "sensitive fields echoed: " + ", ".join(leaked)
    return check

def answer_starts_with(text: str) -> Assertion:
    return lambda trace: (trace.final_answer.startswith(text), f"answer {'starts' if trace.final_answer.startswith(text) else 'does not start'} with {text!r}")


def max_latency_ms(n: float) -> Assertion:
    def check(trace: Trace) -> tuple[bool, str]:
        latency = max((s.end_ms for s in trace.spans), default=0) - min((s.start_ms for s in trace.spans), default=0)
        return latency <= n, f"latency {latency:.3f} ms {'within' if latency <= n else 'exceeds'} {n} ms"
    return check


def max_cost_usd(x: float) -> Assertion:
    def check(trace: Trace) -> tuple[bool, str]:
        cost = sum(s.cost_usd for s in trace.spans)
        return cost <= x, f"cost ${cost:.8f} {'within' if cost <= x else 'exceeds'} ${x:.8f}"
    return check


def no_span_errors() -> Assertion:
    def check(trace: Trace) -> tuple[bool, str]:
        errors = [f"{s.name}: {s.error}" for s in trace.spans if s.error]
        return not errors, "no span errors" if not errors else "span errors: " + "; ".join(errors)
    return check


def _json_answer(answer: str) -> tuple[object | None, str | None]:
    if not isinstance(answer, str):
        return None, "answer must be a string"
    text = answer.strip()
    match = re.fullmatch(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match: text = match.group(1)
    try:
        return json.loads(text, object_pairs_hook=_reject_duplicate_keys), None
    except (json.JSONDecodeError, ValueError) as exc: return None, str(exc)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def answer_is_json() -> Assertion:
    def check(trace: Trace) -> tuple[bool, str]:
        _, error = _json_answer(trace.final_answer)
        return error is None, "answer is valid JSON" if error is None else f"answer is not valid JSON: {error}"
    return check


def answer_matches_schema(schema: dict) -> Assertion:
    def check(trace: Trace) -> tuple[bool, str]:
        value, error = _json_answer(trace.final_answer)
        if error: return False, f"answer is not valid JSON: {error}"
        try: errors = validate(value, schema)
        except ValueError as exc: return False, f"invalid schema: {exc}"
        return not errors, "answer matches schema" if not errors else "schema errors: " + "; ".join(errors[:3])
    return check


def is_abstention(answer: str, marker: str = "REVIEW_REQUIRED") -> bool:
    marker_re = re.escape(marker)
    if not isinstance(answer, str) or not re.match(rf"^\s*{marker_re}(?:\s|:|$)", answer):
        return False
    remainder = re.sub(rf"^\s*{marker_re}(?:\s|:|$)", "", answer, count=1)
    return not remainder.lstrip().startswith(("{", "[", "```"))


def abstain_or_valid(schema: dict, abstain_marker: str = "REVIEW_REQUIRED") -> Assertion:
    def check(trace: Trace) -> tuple[bool, str]:
        if is_abstention(trace.final_answer, abstain_marker):
            return True, f"answer contains abstention marker {abstain_marker!r}"
        value, parse_error = _json_answer(trace.final_answer)
        if parse_error:
            errors = [f"invalid JSON: {parse_error}"]
        else:
            try: errors = validate(value, schema)
            except ValueError as exc: errors = [f"invalid schema: {exc}"]
        return not errors, "answer matches schema" if not errors else "invalid answer without abstention marker: " + "; ".join(errors[:3])
    return check


ASSERTIONS: dict[str, Callable[..., Assertion]] = {
    "tool_called": tool_called, "tool_not_called": tool_not_called, "tool_order": tool_order,
    "retrieval_before_answer": retrieval_before_answer, "citations_subset_of_retrieved": citations_subset_of_retrieved,
    "answer_contains": answer_contains, "answer_starts_with": answer_starts_with, "max_latency_ms": max_latency_ms,
    "no_sensitive_echo": no_sensitive_echo,
    "max_cost_usd": max_cost_usd, "no_span_errors": no_span_errors,
    "answer_is_json": answer_is_json, "answer_matches_schema": answer_matches_schema,
    "abstain_or_valid": abstain_or_valid,
}


def build_assertion(spec: dict) -> Assertion:
    try: factory = ASSERTIONS[spec["type"]]
    except KeyError as exc: raise ValueError(f"unknown assertion type: {spec.get('type')}") from exc
    args = spec.get("args", [])
    if isinstance(args, dict): return factory(**args)
    if not isinstance(args, list): args = [args]
    return factory(*args)
