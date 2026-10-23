# Agent trace evaluation report

| Case | Runs | Passed | Pass rate | Wilson 95% lower | Threshold | Result |
|---|---:|---:|---:|---:|---:|---|
| book-date-slot | 20 | 14 | 70.0% | 48.1% | 80.0% | FAIL |
| book-slot-first | 20 | 14 | 70.0% | 48.1% | 80.0% | FAIL |
| reschedule-request | 20 | 19 | 95.0% | 76.4% | 80.0% | PASS |
| cancel-request | 20 | 19 | 95.0% | 76.4% | 80.0% | PASS |
| missing-date | 20 | 19 | 95.0% | 76.4% | 80.0% | PASS |
| ambiguous-slot | 20 | 19 | 95.0% | 76.4% | 80.0% | PASS |

## Failures by assertion
- **book-date-slot**: `answer_is_json` 6, `answer_matches_schema` 6
- **book-slot-first**: `answer_matches_schema` 6
- **reschedule-request**: `answer_starts_with` 1
- **cancel-request**: `answer_starts_with` 1
- **missing-date**: `answer_starts_with` 1
- **ambiguous-slot**: `answer_starts_with` 1

**Overall: FAIL**
