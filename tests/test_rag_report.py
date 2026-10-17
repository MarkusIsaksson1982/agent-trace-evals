import pytest

from agent_trace_evals.rag.report import HEADER, SEPARATOR, render_report

ENTITY = "&#124;"

RECS = [
    {"qid": "q1", "retrieved": ["pwd-01", "pwd-02"], "citations": ["pwd-01", "pwd-02"],
     "citations_valid": True, "refused": False, "faithfulness": 1.0},
    {"qid": "q2", "retrieved": ["refund-01"], "citations": ["refund-01", "made-up"],
     "citations_valid": False, "refused": False, "faithfulness": 0.5},
    {"qid": "q3", "retrieved": ["ship-01"], "citations": [],
     "citations_valid": True, "refused": True, "faithfulness": 1.0},
]

FIELDS = ["qid", "retrieved", "citations", "cite_ok", "faithfulness", "refused"]


def _cells(row):
    assert row.startswith("| ") and row.endswith(" |"), row
    return [c.strip() for c in row.strip("|").split("|")]


def _data_rows(md):
    return [ln for ln in md.splitlines() if ln.startswith("|") and ln not in (HEADER, SEPARATOR)]


def test_header_and_separator_contract():
    lines = render_report([]).splitlines()
    assert lines[0] == HEADER
    assert lines[1] == SEPARATOR
    assert _cells(HEADER) == FIELDS
    assert len(_cells(SEPARATOR)) == len(FIELDS)


def test_one_row_per_question_with_exactly_six_cells():
    rows = _data_rows(render_report(RECS))
    assert len(rows) == 3
    assert all(len(_cells(r)) == len(FIELDS) for r in rows)
    assert _cells(rows[0]) == ["q1", "pwd-01, pwd-02", "pwd-01, pwd-02", "ok", "1.0", "no"]
    assert _cells(rows[1]) == ["q2", "refund-01", "refund-01, made-up", "no: made-up", "0.5", "no"]
    assert _cells(rows[2]) == ["q3", "ship-01", "-", "n/a", "n/a", "yes"]


def test_totals():
    md = render_report(RECS)
    assert "- questions: 3" in md
    assert "- refused: 1/3" in md
    assert "- refusal_rate: 33.333%" in md
    assert "- citation_valid (answered): 1/2" in md
    assert "- faithfulness_mean (answered): 0.75" in md


def test_empty_records_render_header_and_n_a_totals():
    md = render_report([])
    assert HEADER in md and SEPARATOR in md
    assert "- questions: 0" in md
    assert "- refused: n/a" in md
    assert "- refusal_rate: n/a" in md
    assert "- citation_valid (answered): n/a" in md
    assert "- faithfulness_mean (answered): n/a" in md


def test_all_refused_does_not_fabricate_totals():
    md = render_report([RECS[2]])
    assert "- refused: 1/1" in md
    assert "- refusal_rate: 100.0%" in md
    assert "- citation_valid (answered): n/a" in md
    assert "- faithfulness_mean (answered): n/a" in md


def test_citations_valid_flag_is_authoritative():
    rec = {"qid": "x", "retrieved": ["a"], "citations": ["a", "bogus"],
           "citations_valid": True, "refused": False, "faithfulness": 1.0}
    md = render_report([rec])
    assert _cells(_data_rows(md)[0]) == ["x", "a", "a, bogus", "ok", "1.0", "no"]
    assert "- citation_valid (answered): 1/1" in md


def test_cite_ok_recomputed_only_when_flag_absent():
    rec = {"qid": "x", "retrieved": ["a"], "citations": ["a", "bogus"],
           "refused": False, "faithfulness": 1.0}
    md = render_report([rec])
    assert _cells(_data_rows(md)[0])[3] == "no: bogus"
    assert "- citation_valid (answered): 0/1" in md


def test_answered_record_without_citations_is_invalid():
    rec = {"qid": "x", "retrieved": ["a"], "citations": [], "refused": False, "faithfulness": 1.0}
    md = render_report([rec])
    assert _cells(_data_rows(md)[0])[3] == "no"
    assert "- citation_valid (answered): 0/1" in md


def test_pipes_are_escaped_and_rows_keep_six_columns():
    rec = {"qid": "a|b", "retrieved": ["x|y"], "citations": ["x|y"],
           "citations_valid": True, "refused": False, "faithfulness": 1.0}
    row = _data_rows(render_report([rec]))[0]
    assert ENTITY in row
    assert row.count("|") == 7
    assert len(_cells(row)) == 6
    assert _cells(row)[0] == "a" + ENTITY + "b"
    assert _cells(row)[1] == "x" + ENTITY + "y"


def test_newlines_are_flattened_into_one_row():
    rec = {"qid": "a\nb", "retrieved": [], "citations": "",
           "refused": False, "faithfulness": 0.0}
    row = _data_rows(render_report([rec]))[0]
    assert "\n" not in row
    assert len(_cells(row)) == 6
    assert _cells(row)[0] == "a b"


def test_refused_string_values_are_counted():
    yes = {"qid": "y", "retrieved": [], "citations": [], "refused": "yes", "faithfulness": None}
    no = {"qid": "n", "retrieved": ["a"], "citations": ["a"], "citations_valid": True,
          "refused": "no", "faithfulness": 0.5}
    md = render_report([yes, no])
    assert "- refused: 1/2" in md
    assert "- citation_valid (answered): 1/1" in md
    assert "- faithfulness_mean (answered): 0.5" in md


def test_unknown_refused_value_raises_value_error():
    with pytest.raises(ValueError):
        render_report([{"qid": "x", "refused": "maybe"}])


def test_non_dict_record_raises_type_error_naming_index():
    with pytest.raises(TypeError, match="index 1"):
        render_report([{"qid": "ok"}, "not-a-dict"])


def test_answered_none_faithfulness_is_n_a_and_skipped_in_mean():
    a = {"qid": "a", "retrieved": ["x"], "citations": ["x"], "citations_valid": True,
         "refused": False, "faithfulness": None}
    b = {"qid": "b", "retrieved": ["y"], "citations": ["y"], "citations_valid": True,
         "refused": False, "faithfulness": 0.5}
    rows = _data_rows(render_report([a, b]))
    assert _cells(rows[0])[4] == "n/a"
    assert _cells(rows[1])[4] == "0.5"
    md = render_report([a, b])
    assert "- faithfulness_mean (answered): 0.5" in md
    assert "- citation_valid (answered): 2/2" in md


def test_mean_rounds_non_binary_values():
    recs = [{"qid": str(i), "retrieved": ["x"], "citations": ["x"], "citations_valid": True,
             "refused": False, "faithfulness": f} for i, f in enumerate((1.0, 1.0, 0.5))]
    assert "- faithfulness_mean (answered): 0.833" in render_report(recs)


def test_citations_str_is_a_single_id():
    rec = {"qid": "x", "retrieved": ["a-b"], "citations": "a-b",
           "citations_valid": True, "refused": False, "faithfulness": 1.0}
    assert _cells(_data_rows(render_report([rec]))[0])[2] == "a-b"


def test_missing_optional_keys_use_defaults():
    md = render_report([{"qid": "only"}])
    assert _cells(_data_rows(md)[0]) == ["only", "-", "-", "no", "n/a", "no"]
