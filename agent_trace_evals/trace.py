"""Trace data structures and JSON Lines persistence."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import copy
import json
from pathlib import Path
from typing import Any, Iterable
import re


@dataclass
class Span:
    name: str
    kind: str
    input: Any = None
    output: Any = None
    start_ms: float = 0.0
    end_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    error: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"llm", "tool", "retrieval"}:
            raise ValueError(f"invalid span kind: {self.kind}")


@dataclass
class Trace:
    case_id: str
    run_idx: int
    seed: int
    spans: list[Span] = field(default_factory=list)
    final_answer: str = ""


def _sensitive_pattern(value: str) -> re.Pattern[str]:
    parts = [re.escape(char) for char in value if char not in " \t\r\n.,_/'-"]
    if len(parts) < 3:
        raise ValueError("sensitive values must contain at least three kept characters")
    return re.compile(r"(?<![A-Za-z0-9])" + r"[\s.,_/'-]*".join(parts) + r"(?![A-Za-z0-9])", re.IGNORECASE)


def redact_text(text: str, fields: dict[str, str], placeholder: str) -> str:
    patterns = [(field, _sensitive_pattern(value)) for field, value in
                sorted(fields.items(), key=lambda item: len(item[1]), reverse=True)]
    return re.sub(r"|".join(f"(?P<field_{i}>{pattern.pattern})" for i, (_, pattern) in enumerate(patterns)),
                  lambda match: placeholder.format(field=patterns[int(match.lastgroup.split('_')[1])][0]), text,
                  flags=re.IGNORECASE) if patterns else text


def redact_trace(trace: Trace, fields: dict[str, str],
                 placeholder: str = "[REDACTED:{field}]") -> Trace:
    """Return a deep redacted copy of a trace without changing the original."""
    def redact(value: Any) -> Any:
        if isinstance(value, str): return redact_text(value, fields, placeholder)
        if isinstance(value, dict): return {redact(key): redact(item) for key, item in value.items()}
        if isinstance(value, list): return [redact(item) for item in value]
        if isinstance(value, tuple): return tuple(redact(item) for item in value)
        return copy.deepcopy(value)

    return Trace(trace.case_id, trace.run_idx, trace.seed,
                 [Span(redact(s.name), s.kind, redact(s.input), redact(s.output), s.start_ms, s.end_ms,
                       s.tokens_in, s.tokens_out, s.cost_usd, redact(s.error)) for s in trace.spans],
                 redact(trace.final_answer))


def trace_from_dict(data: dict[str, Any]) -> Trace:
    return Trace(
        case_id=str(data["case_id"]), run_idx=int(data["run_idx"]), seed=int(data["seed"]),
        spans=[Span(**span) for span in data.get("spans", [])],
        final_answer=str(data.get("final_answer", "")),
    )


def write_jsonl(traces: Iterable[Trace], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for trace in traces:
            stream.write(json.dumps(asdict(trace), ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[Trace]:
    traces = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    traces.append(trace_from_dict(json.loads(line)))
                except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
                    raise ValueError(f"invalid trace JSON on line {line_number}: {exc}") from exc
    return traces
