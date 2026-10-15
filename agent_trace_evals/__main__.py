"""Command line interface for agent trace evaluations."""
from __future__ import annotations

import argparse
from pathlib import Path

from .runner import report_from_traces, run_cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent_trace_evals")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run evaluation cases")
    run.add_argument("cases")
    run.add_argument("--runs", type=int, default=20)
    run.add_argument("--flakiness", type=float, default=0.0)
    run.add_argument("--out", default="out/")
    report = commands.add_parser("report", help="recompute a report from saved traces")
    report.add_argument("traces")
    report.add_argument("--cases", default=str(Path(__file__).resolve().parent.parent / "cases" / "support.json"))
    report.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    if args.command == "run":
        summary = run_cases(args.cases, args.runs, args.flakiness, args.out)
        print(f"Evaluation {'PASS' if summary['passed'] else 'FAIL'}; report: {Path(args.out) / 'report.md'}")
    else:
        output = args.out or str(Path(args.traces).with_name("report.md"))
        summary = report_from_traces(args.traces, args.cases, output)
        print(f"Report {'PASS' if summary['passed'] else 'FAIL'}: {output}")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
