"""Trace data structures and JSON Lines persistence."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable


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
