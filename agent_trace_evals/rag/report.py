"""Stdlib-only markdown report that surfaces RAG hallucinations from eval results.

Input is the list of per-question records produced by ``RAGPipeline.answer``
(or anything with the same shape)::

    {
        "qid": "q1",               # required; non-empty str or int
        "retrieved": ["pwd-01"],   # optional list/tuple of ids; a str is ONE id
        "citations": ["pwd-01"],   # optional list/tuple of ids; a str is ONE id
        "citations_valid": True,   # optional; authoritative when present
        "refused": False,          # optional; bool or yes/no/true/false/1/0
        "faithfulness": 1.0,       # optional; float in [0, 1] or None
    }

The result is a per-question table plus aggregate totals.  Column ``cite_ok`` is
``n/a`` for a refusal (a refusal needs no citation); otherwise it uses the
record's ``citations_valid`` verbatim when present and only recomputes
``citations`` against ``retrieved`` when that field is absent.  Column
``faithfulness`` is likewise ``n/a`` for refusals and for missing values, so
exactly the rows feeding the mean show a number.  Every cell is escaped: a
literal pipe becomes the HTML entity ``&#124;`` and newlines become spaces, so a
data row always parses as the six GFM columns declared in ``HEADER``.

``faithfulness`` must be ``None`` or a finite real number in ``[0, 1]`` (a
numeric string is accepted, ``bool``/NaN/+-inf are rejected); anything else
raises ``ValueError`` naming the offending record's ``qid``.

Usage::

    from agent_trace_evals.rag import render_report

    print(render_report([
        {"qid": "q1", "retrieved": ["pwd-01"], "citations": ["pwd-01"],
         "refused": False, "faithfulness": 1.0},
    ]))
"""

from __future__ import annotations

import math

HEADER = "| qid | retrieved | citations | cite_ok | faithfulness | refused |"
SEPARATOR = "| --- | --- | --- | --- | --- | --- |"


def _escape_cell(value):
    """Render any value as one GFM cell: no raw pipe and no newline."""
    text = "" if value is None else str(value)
    return text.replace("|", "&#124;").replace("\r\n", " ").replace("\n", " ").replace("\r", " ")


def _valid_qid(qid) -> bool:
    """A qid must be a non-blank str or an int (bool is not an int here)."""
    if isinstance(qid, bool):
        return False
    if isinstance(qid, int):
        return True
    return isinstance(qid, str) and bool(qid.strip())


def _as_id_list(value):
    """Normalize an id field: None -> [], str -> one id, list/tuple -> ids."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    raise TypeError(f"expected str/list/tuple of ids, got {type(value).__name__}")


def _as_bool(value):
    """Strict yes/no coercion for ``refused``; unknown values fail loudly."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("yes", "true", "1"):
            return True
        if text in ("", "no", "false", "0"):
            return False
    raise ValueError(f"cannot interpret {value!r} as a refusal flag")


def _faith_error(qid, value):
    return ValueError(
        f"faithfulness for qid {qid!r} must be None or a finite number in [0, 1], got {value!r}")


def _as_float(value, qid):
    """Faithfulness: None stays None; else a finite real in [0, 1] (no bool)."""
    if value is None:
        return None
    original = value
    if isinstance(value, bool):
        raise _faith_error(qid, original)
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            raise _faith_error(qid, original) from None
    if not isinstance(value, (int, float)):
        raise _faith_error(qid, original)
    try:
        number = float(value)
    except OverflowError:
        raise _faith_error(qid, original) from None
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise _faith_error(qid, original)
    return number


def _num(value):
    """Readable numeric format: rounded to three decimals, e.g. 0.833."""
    return str(round(float(value), 3))


def _analyze(rec):
    qid = rec.get("qid", "")
    retrieved = _as_id_list(rec.get("retrieved"))
    citations = _as_id_list(rec.get("citations"))
    refused = _as_bool(rec.get("refused"))
    faith = _as_float(rec.get("faithfulness"), qid)
    unknown = [c for c in citations if c not in retrieved]

    flag = rec.get("citations_valid")
    if refused:
        valid = None
    elif flag is None:
        valid = not unknown and bool(citations)
    else:
        valid = _as_bool(flag)

    return {"qid": qid, "retrieved": retrieved, "citations": citations,
            "refused": refused, "faith": faith, "valid": valid, "unknown": unknown}


def _cite_ok(a):
    if a["refused"]:
        return "n/a"
    if a["valid"]:
        return "ok"
    return "no: " + ", ".join(a["unknown"]) if a["unknown"] else "no"


def _faith_cell(a):
    if a["refused"] or a["faith"] is None:
        return "n/a"
    return _num(a["faith"])


def _row(a):
    cells = [
        _escape_cell(a["qid"]),
        _escape_cell(", ".join(a["retrieved"]) or "-"),
        _escape_cell(", ".join(a["citations"]) or "-"),
        _escape_cell(_cite_ok(a)),
        _escape_cell(_faith_cell(a)),
        "yes" if a["refused"] else "no",
    ]
    return "| " + " | ".join(cells) + " |"


def _ratio(num, den):
    return f"{num}/{den}" if den else "n/a"


def _mean(values):
    return _num(sum(values) / len(values)) if values else "n/a"


def render_report(records):
    """Render ``records`` as markdown (table + totals).

    Raises ``TypeError`` for a non-dict record (naming its index),
    ``ValueError`` for a missing/empty ``qid`` or a refused/faithfulness value
    that cannot be parsed.
    """
    lines = [HEADER, SEPARATOR]
    analyses = []
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise TypeError(f"record at index {i} is not a dict (got {type(rec).__name__})")
        qid = rec.get("qid")
        if not _valid_qid(qid):
            raise ValueError(f"record at index {i} has no valid qid (got {qid!r})")
        analyses.append(_analyze(rec))
        lines.append(_row(analyses[-1]))

    n = len(analyses)
    answered = [a for a in analyses if not a["refused"]]
    refused_n = n - len(answered)
    cite = [a["valid"] for a in answered]
    faith = [a["faith"] for a in answered if a["faith"] is not None]
    refusal_rate = _num(100.0 * refused_n / n) + "%" if n else "n/a"
    lines += [
        "",
        f"- questions: {n}",
        f"- refused: {_ratio(refused_n, n)}",
        f"- refusal_rate: {refusal_rate}",
        f"- citation_valid (answered): {_ratio(sum(cite), len(cite))}",
        f"- faithfulness_mean (answered): {_mean(faith)}",
    ]
    return "\n".join(lines)
