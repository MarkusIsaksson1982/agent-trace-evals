# Changelog

## 2026-10-23
- Added a synthetic booking/rescheduling case set (`cases/booking_requests.json`), made `--flakiness` inject structured-output failures into booking answers, renamed the report section to "Failures by assertion", and recorded clean and flaky reports under `examples/booking/`.

## 2026-10-22
- Added structured-output reliability checks: a small stdlib JSON-Schema-subset validator and the assertions `answer_is_json`, `answer_matches_schema` and `abstain_or_valid`, with tests and README docs.
- Added PLAN.md (short update plan).
