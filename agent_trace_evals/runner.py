"""Case loading, repeated execution, and compact Markdown summaries."""
from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from fractions import Fraction
from typing import Any

from .agent import run_agent
from .assertions import build_assertion
from .trace import Trace, read_jsonl, write_jsonl

BASELINE_VERSION = 1


def wilson_lower_bound(successes: int, trials: int, z: float = 1.96) -> float:
    if trials <= 0: return 0.0
    p = successes / trials
    denom = 1 + z * z / trials
    center = p + z * z / (2 * trials)
    margin = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
    return (center - margin) / denom


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream: data = json.load(stream)
    cases = data.get("cases", data) if isinstance(data, dict) else data
    if not isinstance(cases, list): raise ValueError("cases file must contain a list or a 'cases' list")
    for case in cases:
        if not all(key in case for key in ("id", "question")): raise ValueError("each case requires id and question")
        case.setdefault("assertions", [])
        case.setdefault("min_pass_rate", 0.8)
    return cases


def evaluate_traces(cases: list[dict[str, Any]], traces: list[Trace]) -> dict[str, Any]:
    summaries = []
    by_id: dict[str, list[Trace]] = {case["id"]: [] for case in cases}
    for trace in traces:
        if trace.case_id in by_id: by_id[trace.case_id].append(trace)
    for case in cases:
        case_traces = by_id[case["id"]]
        assertions = [(spec, build_assertion(spec)) for spec in case.get("assertions", [])]
        failed = Counter()
        passing_runs = 0
        for trace in case_traces:
            run_pass = True
            for spec, assertion in assertions:
                ok, _ = assertion(trace)
                if not ok:
                    run_pass = False
                    failed[spec["type"]] += 1
            passing_runs += int(run_pass)
        total = len(case_traces)
        rate = passing_runs / total if total else 0.0
        threshold = float(case.get("min_pass_rate", 0.8))
        summaries.append({"id": case["id"], "question": case["question"], "runs": total,
                          "passed_runs": passing_runs, "pass_rate": rate,
                          "wilson_lower_bound": wilson_lower_bound(passing_runs, total),
                          "min_pass_rate": threshold, "passed": total > 0 and rate >= threshold,
                          "assertion_failures": dict(failed)})
    return {"cases": summaries, "passed": bool(summaries) and all(row["passed"] for row in summaries)}


def format_report(summary: dict[str, Any]) -> str:
    lines = ["# Agent trace evaluation report", "", "| Case | Runs | Passed | Pass rate | Wilson 95% lower | Threshold | Result |",
             "|---|---:|---:|---:|---:|---:|---|"]
    fails: list[str] = []
    for row in summary["cases"]:
        lines.append(f"| {row['id']} | {row['runs']} | {row['passed_runs']} | {row['pass_rate']:.1%} | "
                     f"{row['wilson_lower_bound']:.1%} | {row['min_pass_rate']:.1%} | {'PASS' if row['passed'] else 'FAIL'} |")
        if row["assertion_failures"]:
            fails.append(f"- **{row['id']}**: " + ", ".join(f"`{name}` {count}" for name, count in sorted(row["assertion_failures"].items())))
    if fails:
        lines.extend(["", "## Failures by assertion", *fails])
    lines.extend(["", f"**Overall: {'PASS' if summary['passed'] else 'FAIL'}**", ""])
    return "\n".join(lines)


def write_report(summary: dict[str, Any], path: str | Path) -> None:
    destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(format_report(summary), encoding="utf-8")


class BaselineError(ValueError):
    """Raised when a baseline file is missing, unreadable, or malformed."""


@dataclass
class Baseline:
    cases: dict[str, float]
    runs: int | None = None
    flakiness: float | None = None
    version: int = BASELINE_VERSION


def baseline_from_summary(summary: dict[str, Any]) -> dict[str, float]:
    return {str(row["id"]): float(row["pass_rate"]) for row in summary["cases"]}


def write_baseline(summary: dict[str, Any], path: str | Path,
                   runs: int | None = None, flakiness: float | None = None) -> None:
    runs = _validate_runs(runs)
    flakiness = _validate_flakiness(flakiness)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": BASELINE_VERSION, "runs": runs, "flakiness": flakiness,
               "cases": baseline_from_summary(summary)}
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    fd, tmp_name = tempfile.mkstemp(dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp")
    try:
        old_umask = os.umask(0)
        os.umask(old_umask)
        os.chmod(tmp_name, 0o666 & ~old_umask)
        try:
            stream = os.fdopen(fd, "w", encoding="utf-8")
        except BaseException:
            os.close(fd)
            raise
        with stream:
            stream.write(text)
        os.replace(tmp_name, destination)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _validate_rate(case_id: str, rate: Any) -> float:
    if isinstance(rate, bool) or not isinstance(rate, (int, float)):
        raise BaselineError(f"baseline pass rate for case {case_id!r} is not a number")
    value = float(rate)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise BaselineError(f"baseline pass rate for case {case_id!r} must be between 0 and 1")
    return value


def _validate_runs(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BaselineError("baseline 'runs' must be a positive integer")
    return value


def _validate_flakiness(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BaselineError("baseline 'flakiness' must be a number")
    flakiness = float(value)
    if not math.isfinite(flakiness) or not 0.0 <= flakiness <= 1.0:
        raise BaselineError("baseline 'flakiness' must be between 0 and 1")
    return flakiness


def load_baseline(path: str | Path) -> Baseline:
    baseline_path = Path(path)
    try:
        with baseline_path.open(encoding="utf-8") as stream:
            data = json.load(stream)
    except FileNotFoundError:
        raise BaselineError(f"baseline file not found: {baseline_path}") from None
    except UnicodeDecodeError:
        raise BaselineError(f"baseline file is not valid UTF-8: {baseline_path}") from None
    except json.JSONDecodeError as error:
        raise BaselineError(f"baseline file is not valid JSON: {baseline_path} ({error})") from None
    except OSError as error:
        raise BaselineError(f"could not read baseline file {baseline_path}: {error}") from None
    if not isinstance(data, dict):
        raise BaselineError("baseline must be a JSON object with a 'cases' mapping")
    version = data.get("version", BASELINE_VERSION)
    if isinstance(version, bool) or not isinstance(version, int) or version != BASELINE_VERSION:
        raise BaselineError(f"unsupported baseline version {version!r} (expected {BASELINE_VERSION})")
    raw_cases = data.get("cases")
    if not isinstance(raw_cases, dict):
        raise BaselineError("baseline must contain a 'cases' object mapping case ids to pass rates")
    rates = {str(case_id): _validate_rate(str(case_id), rate) for case_id, rate in raw_cases.items()}
    return Baseline(cases=rates, runs=_validate_runs(data.get("runs")),
                    flakiness=_validate_flakiness(data.get("flakiness")), version=version)


def _exact_rate(value: Any) -> Fraction:
    """Exact decimal value of a printed float, for ULP-free comparisons."""
    return Fraction(repr(float(value)))


def compare_to_baseline(summary: dict[str, Any], baseline: Baseline, max_drop: float = 0.05,
                        allow_missing: bool = False) -> dict[str, Any]:
    """Gate that fails iff ``baseline - current > max_drop`` for any shared case.

    Uses an exact decimal comparison (``Fraction(repr(...))``), so rates are
    compared as printed: a drop exactly at the threshold passes and any larger
    drop fails, with no floating-point slack.
    """
    if isinstance(max_drop, bool) or not isinstance(max_drop, (int, float)) or \
            not math.isfinite(max_drop) or not 0.0 <= max_drop <= 1.0:
        raise ValueError("max_drop must be a finite number between 0 and 1")
    rates = baseline.cases
    current_rates = {str(row["id"]): float(row["pass_rate"]) for row in summary["cases"]}
    limit = _exact_rate(max_drop)
    regressions: list[dict[str, Any]] = []
    new_cases: list[str] = []
    for row in summary["cases"]:
        case_id = str(row["id"])
        if case_id not in rates:
            new_cases.append(case_id)
            continue
        drop = _exact_rate(rates[case_id]) - _exact_rate(current_rates[case_id])
        if drop > limit:
            regressions.append({"id": case_id, "baseline_pass_rate": rates[case_id],
                                "pass_rate": row["pass_rate"], "drop": float(drop)})
    missing_cases = sorted(case_id for case_id in rates if case_id not in current_rates)
    failed = bool(regressions) or (bool(missing_cases) and not allow_missing)
    return {"passed": not failed, "max_drop": max_drop, "regressions": regressions,
            "missing_cases": missing_cases, "new_cases": sorted(new_cases)}


def run_cases(cases_path: str | Path, runs: int = 20, flakiness: float = 0.0,
              out: str | Path = "out", kb_path: str | Path | None = None) -> dict[str, Any]:
    if runs < 1: raise ValueError("runs must be at least 1")
    cases = load_cases(cases_path)
    if kb_path is None: kb_path = Path(__file__).resolve().parent.parent / "kb" / "support_kb.json"
    with Path(kb_path).open(encoding="utf-8") as stream: kb = json.load(stream)
    traces = [run_agent(case["question"], kb, seed, flakiness, case_id=case["id"], run_idx=seed)
              for case in cases for seed in range(runs)]
    out_dir = Path(out); out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(traces, out_dir / "traces.jsonl")
    summary = evaluate_traces(cases, traces)
    write_report(summary, out_dir / "report.md")
    return summary


def report_from_traces(traces_path: str | Path, cases_path: str | Path,
                       output_path: str | Path | None = None) -> dict[str, Any]:
    summary = evaluate_traces(load_cases(cases_path), read_jsonl(traces_path))
    if output_path is not None: write_report(summary, output_path)
    return summary
