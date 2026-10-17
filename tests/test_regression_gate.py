import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agent_trace_evals.__main__ import _apply_baseline, _runs_from_summary, main
from agent_trace_evals.runner import (BASELINE_VERSION, Baseline, BaselineError,
    baseline_from_summary, compare_to_baseline, load_baseline, write_baseline)

ROOT = Path(__file__).resolve().parents[1]


def _summary(*rates):
    cases = [{"id": f"c{index}", "pass_rate": rate} for index, rate in enumerate(rates)]
    return {"cases": cases, "passed": True}


def test_baseline_from_summary():
    assert baseline_from_summary(_summary(1.0, 0.5)) == {"c0": 1.0, "c1": 0.5}


def test_numeric_ids_are_compared_consistently():
    summary = {"cases": [{"id": 12, "pass_rate": 1.0}], "passed": True}
    baseline = Baseline(cases=baseline_from_summary(summary))
    gate = compare_to_baseline(summary, baseline, max_drop=0.1)
    assert gate["passed"] and gate["new_cases"] == [] and gate["missing_cases"] == []


def test_write_read_round_trip(tmp_path):
    path = tmp_path / "nested" / "baseline.json"
    write_baseline(_summary(0.75), path, runs=20, flakiness=0.0)
    baseline = load_baseline(path)
    assert baseline.cases == {"c0": 0.75}
    assert baseline.runs == 20
    assert baseline.flakiness == 0.0
    assert baseline.version == BASELINE_VERSION


def test_load_rejects_bad_shapes(tmp_path):
    for payload in ([1, 2], {"runs": 20}, {"cases": [1]}, {"cases": {"c0": "high"}},
                    {"cases": {"c0": float("nan")}}, {"cases": {"c0": 1.5}},
                    {"cases": {"c0": True}}, {"version": 99, "cases": {"c0": 0.5}}):
        path = tmp_path / "baseline.json"
        path.write_text(json.dumps(payload))
        with pytest.raises(BaselineError):
            load_baseline(path)


def test_load_rejects_bad_metadata(tmp_path):
    bad_metadata = [{"runs": 0}, {"runs": -1}, {"runs": 1.5}, {"runs": True}, {"runs": "20"},
                    {"flakiness": -0.1}, {"flakiness": 1.5}, {"flakiness": "0"},
                    {"flakiness": True}, {"flakiness": float("nan")}]
    for metadata in bad_metadata:
        path = tmp_path / "baseline.json"
        path.write_text(json.dumps({"cases": {"c0": 0.5}, **metadata}))
        with pytest.raises(BaselineError):
            load_baseline(path)


def test_load_rejects_non_int_version(tmp_path):
    for version in (True, 1.0):
        path = tmp_path / "baseline.json"
        path.write_text(json.dumps({"version": version, "cases": {"c0": 0.5}}))
        with pytest.raises(BaselineError):
            load_baseline(path)


def test_load_missing_file_is_clean(tmp_path):
    with pytest.raises(BaselineError):
        load_baseline(tmp_path / "absent.json")


def test_drop_beyond_threshold_fails():
    gate = compare_to_baseline(_summary(0.5), Baseline(cases={"c0": 0.9}), max_drop=0.1)
    assert not gate["passed"]
    assert gate["regressions"][0]["drop"] == pytest.approx(0.4)


def test_drop_exactly_at_threshold_passes():
    gate = compare_to_baseline(_summary(0.85), Baseline(cases={"c0": 0.90}), max_drop=0.05)
    assert gate["passed"]


def test_drop_infinitesimally_above_threshold_fails():
    gate = compare_to_baseline(_summary(0.9 - 0.05 - 1e-9), Baseline(cases={"c0": 0.9}), max_drop=0.05)
    assert not gate["passed"]


def test_drop_one_ulp_above_threshold_fails():
    current = math.nextafter(0.85, 0.0)
    gate = compare_to_baseline(_summary(current), Baseline(cases={"c0": 0.90}), max_drop=0.05)
    assert not gate["passed"]


def test_drop_just_over_threshold_fails():
    gate = compare_to_baseline(_summary(0.90 - (0.05 + 1e-12)), Baseline(cases={"c0": 0.90}), max_drop=0.05)
    assert not gate["passed"]


def test_drop_within_threshold_passes():
    gate = compare_to_baseline(_summary(0.85), Baseline(cases={"c0": 0.9}), max_drop=0.1)
    assert gate["passed"]


def test_compare_rejects_invalid_max_drop():
    baseline = Baseline(cases={"c0": 0.5})
    for value in (-0.1, 1.5, float("nan"), True):
        with pytest.raises(ValueError):
            compare_to_baseline(_summary(0.5), baseline, max_drop=value)


def test_missing_case_fails_by_default():
    gate = compare_to_baseline(_summary(0.5), Baseline(cases={"c0": 0.5, "gone": 1.0}), max_drop=0.1)
    assert not gate["passed"]
    assert gate["missing_cases"] == ["gone"]


def test_missing_case_can_be_allowed():
    gate = compare_to_baseline(_summary(0.5), Baseline(cases={"c0": 0.5, "gone": 1.0}),
                               max_drop=0.1, allow_missing=True)
    assert gate["passed"]


def test_new_cases_are_reported_not_failed():
    gate = compare_to_baseline(_summary(0.5, 1.0), Baseline(cases={"c0": 0.5}), max_drop=0.1)
    assert gate["passed"]
    assert gate["new_cases"] == ["c1"]


def test_write_baseline_is_atomic_and_leaves_no_temp_files(tmp_path):
    path = tmp_path / "baseline.json"
    write_baseline(_summary(0.75), path, runs=20, flakiness=0.0)
    assert json.loads(path.read_text()) == {
        "version": BASELINE_VERSION, "runs": 20, "flakiness": 0.0, "cases": {"c0": 0.75}}
    assert load_baseline(path).cases == {"c0": 0.75}
    assert [p.name for p in tmp_path.iterdir()] == ["baseline.json"]


def test_write_baseline_honours_umask(tmp_path):
    old = os.umask(0o022)
    try:
        path = tmp_path / "baseline.json"
        write_baseline(_summary(0.75), path)
        assert (path.stat().st_mode & 0o777) == 0o644
    finally:
        os.umask(old)


def test_write_baseline_failure_cleans_temp_and_keeps_existing(tmp_path, monkeypatch):
    path = tmp_path / "baseline.json"
    path.write_text('{"existing": true}')

    def boom(*args, **kwargs):
        raise OSError("replace failed")

    monkeypatch.setattr("agent_trace_evals.runner.os.replace", boom)
    with pytest.raises(OSError):
        write_baseline(_summary(0.75), path)
    assert path.read_text() == '{"existing": true}'
    assert [p.name for p in tmp_path.iterdir()] == ["baseline.json"]


def test_apply_baseline_write_error_returns_exit_code(monkeypatch):
    def boom(*args, **kwargs):
        raise BaselineError("bad")

    monkeypatch.setattr("agent_trace_evals.__main__.write_baseline", boom)
    args = argparse.Namespace(write_baseline="baseline.json", force_baseline=False)
    assert _apply_baseline(_summary(0.5), args, max_drop=0.05) == 2


def test_write_baseline_rejects_invalid_metadata_without_writing(tmp_path):
    path = tmp_path / "baseline.json"
    for metadata in ({"runs": 0}, {"runs": True}, {"flakiness": 1.5}, {"flakiness": float("nan")}):
        with pytest.raises(BaselineError):
            write_baseline(_summary(0.5), path, **metadata)
        assert not path.exists()
        assert list(tmp_path.iterdir()) == []


def _write_cases(path, min_pass_rate=0.0):
    path.write_text(json.dumps([{"id": "c", "question": "What is 2 + 2?",
        "min_pass_rate": min_pass_rate,
        "assertions": [{"type": "answer_contains", "args": ["zzzznope"]}]}]))


def _run_cli(cases, out, *extra):
    return subprocess.run([sys.executable, "-m", "agent_trace_evals", "run", str(cases),
        "--runs", "1", "--out", str(out), *extra], cwd=ROOT, capture_output=True, text=True)


def test_write_then_compare_passes(tmp_path):
    cases, baseline = tmp_path / "cases.json", tmp_path / "baseline.json"
    _write_cases(cases)
    assert _run_cli(cases, tmp_path / "o1", "--write-baseline", str(baseline)).returncode == 0
    assert baseline.exists()
    second = _run_cli(cases, tmp_path / "o2", "--baseline", str(baseline))
    assert second.returncode == 0
    assert "Regression gate PASS" in second.stdout


def test_regression_against_stricter_baseline_fails(tmp_path):
    cases, baseline = tmp_path / "cases.json", tmp_path / "baseline.json"
    _write_cases(cases)
    baseline.write_text(json.dumps({"cases": {"c": 1.0}}))
    result = _run_cli(cases, tmp_path / "o", "--baseline", str(baseline))
    assert result.returncode == 3
    assert "Regression gate FAIL" in result.stdout
    assert "REGRESSION c" in result.stdout


def test_missing_baseline_is_clean_error(tmp_path):
    cases = tmp_path / "cases.json"
    _write_cases(cases)
    result = _run_cli(cases, tmp_path / "o", "--baseline", str(tmp_path / "absent.json"))
    assert result.returncode == 2
    assert "baseline file not found" in result.stderr
    assert "Traceback" not in result.stderr


def test_write_and_compare_combination_rejected(tmp_path):
    cases = tmp_path / "cases.json"
    _write_cases(cases)
    result = _run_cli(cases, tmp_path / "o", "--write-baseline", str(tmp_path / "b.json"),
                      "--baseline", str(tmp_path / "b.json"))
    assert result.returncode == 2
    assert "cannot be used together" in result.stderr


def test_invalid_max_drop_rejected(tmp_path):
    cases = tmp_path / "cases.json"
    _write_cases(cases)
    for value in ("-0.5", "1.5"):
        result = _run_cli(cases, tmp_path / "o", f"--max-drop={value}")
        assert result.returncode == 2
        assert "--max-drop must be between 0 and 1" in result.stderr


def test_max_drop_without_baseline_warns(tmp_path):
    cases = tmp_path / "cases.json"
    _write_cases(cases)
    result = _run_cli(cases, tmp_path / "o", "--max-drop", "0.2")
    assert result.returncode == 0
    assert "--max-drop has no effect without --baseline" in result.stderr


def test_baseline_runs_mismatch_warns(tmp_path):
    cases, baseline = tmp_path / "cases.json", tmp_path / "baseline.json"
    _write_cases(cases)
    baseline.write_text(json.dumps({"cases": {"c": 0.0}, "runs": 99}))
    result = _run_cli(cases, tmp_path / "o", "--baseline", str(baseline))
    assert "may not be comparable" in result.stderr


def test_refuses_baseline_from_failing_run(tmp_path):
    cases, baseline = tmp_path / "cases.json", tmp_path / "baseline.json"
    _write_cases(cases, min_pass_rate=0.5)
    result = _run_cli(cases, tmp_path / "o", "--write-baseline", str(baseline))
    assert result.returncode == 2
    assert not baseline.exists()
    assert "refusing to write" in result.stderr
    forced = _run_cli(cases, tmp_path / "o2", "--write-baseline", str(baseline), "--force-baseline")
    assert forced.returncode == 1
    assert baseline.exists()


def test_missing_case_fails_gate(tmp_path):
    cases, baseline = tmp_path / "cases.json", tmp_path / "baseline.json"
    _write_cases(cases)
    baseline.write_text(json.dumps({"cases": {"c": 0.0, "gone": 1.0}}))
    result = _run_cli(cases, tmp_path / "o", "--baseline", str(baseline))
    assert result.returncode == 3
    assert "MISSING gone" in result.stdout
    allowed = _run_cli(cases, tmp_path / "o2", "--baseline", str(baseline), "--allow-missing-cases")
    assert allowed.returncode == 0


def test_exit_code_precedence(tmp_path):
    cases, baseline = tmp_path / "cases.json", tmp_path / "baseline.json"
    _write_cases(cases, min_pass_rate=0.5)
    baseline.write_text(json.dumps({"cases": {"c": 0.0}}))
    assert _run_cli(cases, tmp_path / "o1", "--baseline", str(baseline)).returncode == 1
    _write_cases(cases)
    baseline.write_text(json.dumps({"cases": {"c": 1.0}}))
    assert _run_cli(cases, tmp_path / "o2", "--baseline", str(baseline)).returncode == 3
    _write_cases(cases, min_pass_rate=0.5)
    assert _run_cli(cases, tmp_path / "o3", "--baseline", str(baseline)).returncode == 3


def test_report_command_supports_baseline(tmp_path):
    cases = tmp_path / "cases.json"
    _write_cases(cases)
    run_out = tmp_path / "run"
    _run_cli(cases, run_out)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"cases": {"c": 0.0}}))
    result = subprocess.run([sys.executable, "-m", "agent_trace_evals", "report",
        str(run_out / "traces.jsonl"), "--cases", str(cases), "--baseline", str(baseline)],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0
    assert "Regression gate PASS" in result.stdout


def test_default_run_without_baseline_is_unchanged(tmp_path):
    cases = tmp_path / "cases.json"
    _write_cases(cases)
    result = _run_cli(cases, tmp_path / "o")
    assert result.returncode == 0
    assert "Regression gate" not in result.stdout


def test_usage_error_exit_code_in_process(tmp_path):
    cases = tmp_path / "cases.json"
    _write_cases(cases)
    with pytest.raises(SystemExit) as excinfo:
        main(["run", str(cases), "--write-baseline", str(tmp_path / "b.json"),
              "--baseline", str(tmp_path / "b.json")])
    assert excinfo.value.code == 2


def test_runs_from_summary_single_value():
    summary = {"cases": [{"runs": 5}, {"runs": 5}], "passed": True}
    assert _runs_from_summary(summary) == 5


def test_runs_from_summary_ignores_missing_and_bools():
    summary = {"cases": [{"runs": 5}, {}, {"runs": True}], "passed": True}
    assert _runs_from_summary(summary) == 5


def test_runs_from_summary_mixed_values_warns(capsys):
    summary = {"cases": [{"runs": 3}, {"runs": 5}], "passed": True}
    assert _runs_from_summary(summary) is None
    assert "unequal run counts per case ([3, 5])" in capsys.readouterr().err


def test_runs_from_summary_empty_is_none(capsys):
    assert _runs_from_summary({"cases": [], "passed": True}) is None
    assert capsys.readouterr().err == ""


def test_report_baseline_runs_mismatch_warns(tmp_path, capsys):
    cases = tmp_path / "cases.json"
    _write_cases(cases)
    traces_out = tmp_path / "run5"
    assert main(["run", str(cases), "--runs", "5", "--out", str(traces_out)]) == 0
    baseline = tmp_path / "baseline.json"
    main(["run", str(cases), "--runs", "3", "--out", str(tmp_path / "run3"),
          "--write-baseline", str(baseline), "--force-baseline"])
    capsys.readouterr()
    main(["report", str(traces_out / "traces.jsonl"), "--cases", str(cases),
          "--baseline", str(baseline)])
    assert "baseline recorded with --runs 3" in capsys.readouterr().err
