"""Command line interface for agent trace evaluations."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .runner import (BaselineError, compare_to_baseline, load_baseline,
                     report_from_traces, run_cases, write_baseline)

DEFAULT_MAX_DROP = 0.05
EXIT_BASELINE_ERROR = 2
EXIT_REGRESSION = 3


def _add_baseline_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--write-baseline", default=None, metavar="PATH",
                        help="write per-case pass rates to PATH for future regression checks")
    parser.add_argument("--baseline", default=None, metavar="PATH",
                        help="exit 3 if any case drops more than --max-drop below this baseline")
    parser.add_argument("--max-drop", type=float, default=None, metavar="RATE",
                        help=f"allowed pass-rate drop per case before the gate fails (default: {DEFAULT_MAX_DROP})")
    parser.add_argument("--force-baseline", action="store_true",
                        help="allow --write-baseline when the evaluation itself failed")
    parser.add_argument("--allow-missing-cases", action="store_true",
                        help="do not fail the gate when baseline cases are absent from this run")


def _validate_baseline_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> float:
    if args.max_drop is not None and not 0.0 <= args.max_drop <= 1.0:
        parser.error("--max-drop must be between 0 and 1")
    if args.write_baseline and args.baseline:
        parser.error("--write-baseline and --baseline cannot be used together")
    if args.max_drop is not None and not args.baseline:
        print("warning: --max-drop has no effect without --baseline", file=sys.stderr)
    return DEFAULT_MAX_DROP if args.max_drop is None else args.max_drop


def _runs_from_summary(summary: dict[str, Any]) -> int | None:
    values = {row["runs"] for row in summary["cases"]
              if isinstance(row.get("runs"), int) and not isinstance(row["runs"], bool)
              and row["runs"] > 0}
    if len(values) == 1:
        return values.pop()
    if len(values) > 1:
        print("warning: traces have unequal run counts per case "
              f"({sorted(values)}); pass rates may not be comparable", file=sys.stderr)
    return None


def _apply_baseline(summary: dict[str, Any], args: argparse.Namespace, max_drop: float,
                    runs: int | None = None) -> int:
    if args.write_baseline:
        if not summary.get("passed") and not args.force_baseline:
            print("error: refusing to write a baseline from a failing evaluation; "
                  "pass --force-baseline to override", file=sys.stderr)
            return EXIT_BASELINE_ERROR
        try:
            write_baseline(summary, args.write_baseline, runs=runs,
                           flakiness=getattr(args, "flakiness", None))
        except (OSError, BaselineError) as error:
            print(f"error: could not write baseline: {error}", file=sys.stderr)
            return EXIT_BASELINE_ERROR
        print(f"Baseline written: {args.write_baseline}", file=sys.stderr)
    if not args.baseline:
        return 0
    try:
        baseline = load_baseline(args.baseline)
    except BaselineError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_BASELINE_ERROR
    flakiness = getattr(args, "flakiness", None)
    if baseline.runs is not None and runs is not None and baseline.runs != runs:
        print(f"warning: baseline recorded with --runs {baseline.runs}, current run uses --runs {runs}; "
              "pass rates may not be comparable", file=sys.stderr)
    if baseline.flakiness is not None and flakiness is not None and baseline.flakiness != flakiness:
        print(f"warning: baseline recorded with --flakiness {baseline.flakiness}, "
              f"current run uses --flakiness {flakiness}", file=sys.stderr)
    gate = compare_to_baseline(summary, baseline, max_drop, allow_missing=args.allow_missing_cases)
    if gate["new_cases"]:
        print(f"note: {len(gate['new_cases'])} case(s) not in baseline, skipped: "
              + ", ".join(gate["new_cases"]), file=sys.stderr)
    for regression in gate["regressions"]:
        print(f"REGRESSION {regression['id']}: {regression['baseline_pass_rate']:.1%} -> "
              f"{regression['pass_rate']:.1%} (drop {regression['drop']:.1%}, allowed {max_drop:.1%})")
    for case_id in gate["missing_cases"]:
        if args.allow_missing_cases:
            print(f"note: baseline case {case_id!r} absent from this run, ignored", file=sys.stderr)
        else:
            print(f"MISSING {case_id}: in baseline but not in this run")
    print(f"Regression gate {'PASS' if gate['passed'] else 'FAIL'}")
    return 0 if gate["passed"] else EXIT_REGRESSION


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent_trace_evals")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run evaluation cases")
    run.add_argument("cases")
    run.add_argument("--runs", type=int, default=20)
    run.add_argument("--flakiness", type=float, default=0.0)
    run.add_argument("--out", default="out/")
    _add_baseline_args(run)
    report = commands.add_parser("report", help="recompute a report from saved traces")
    report.add_argument("traces")
    report.add_argument("--cases", default=str(Path(__file__).resolve().parent.parent / "cases" / "support.json"))
    report.add_argument("--out", default=None)
    _add_baseline_args(report)
    args = parser.parse_args(argv)
    max_drop = _validate_baseline_args(parser, args)
    if args.command == "run":
        summary = run_cases(args.cases, args.runs, args.flakiness, args.out)
        print(f"Evaluation {'PASS' if summary['passed'] else 'FAIL'}; report: {Path(args.out) / 'report.md'}")
        gate_code = _apply_baseline(summary, args, max_drop, runs=args.runs)
    else:
        output = args.out or str(Path(args.traces).with_name("report.md"))
        summary = report_from_traces(args.traces, args.cases, output)
        print(f"Report {'PASS' if summary['passed'] else 'FAIL'}: {output}")
        gate_code = _apply_baseline(summary, args, max_drop, runs=_runs_from_summary(summary))
    if gate_code:
        return gate_code
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
