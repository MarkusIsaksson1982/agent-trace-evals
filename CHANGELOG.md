# Changelog

## 2026-10-24
- Added a PII-leak guard: the `no_sensitive_echo` assertion fails when the final answer echoes synthetic case fields declared under `sensitive` (or their redaction placeholders), and `run_cases` now redacts those fields from traces before writing `traces.jsonl`; with a synthetic case set (`cases/pii_guard.json`), tests and README docs.

## 2026-10-23
- Added a synthetic booking/rescheduling case set (`cases/booking_requests.json`), made `--flakiness` inject structured-output failures into booking answers, renamed the report section to "Failures by assertion", and recorded clean and flaky reports under `examples/booking/`.

## 2026-10-22
- Added structured-output reliability checks: a small stdlib JSON-Schema-subset validator and the assertions `answer_is_json`, `answer_matches_schema` and `abstain_or_valid`, with tests and README docs.
- Added PLAN.md (short update plan).
