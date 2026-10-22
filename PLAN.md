# Update plan (Oct 22–25, 2026)

Short, public plan for a few small, checkable updates to this repo. Each day adds **one** change that can be checked: code + tests, docs, or recorded eval results. All data is synthetic, and nothing here is a clinical or medical feature.

Theme: **reliability checks for AI features in sensitive domains**. The checks cover structured output, abstaining and handing off to a human, and trace evidence. They are about *how* an AI feature behaves, not domain advice.

| Day | Change | Done when |
|---|---|---|
| Thu Oct 22 | Structured-output checks: stdlib JSON-Schema-subset validator, `answer_is_json`, `answer_matches_schema`, `abstain_or_valid` (invalid output never passes unless the answer explicitly flags review) | New unit tests pass; README section; CHANGELOG entry |
| Fri Oct 23 | Synthetic admin-request case set (e.g. booking/rescheduling extraction) plus a recorded run with and without injected flakiness | `report.md` committed under `examples/`; assertion failures attributed per case |
| Sat Oct 24 | PII-leak guard assertion: the answer must not echo fixture fields marked sensitive (synthetic names/IDs), plus redaction of traces before they are written | Tests for leak / no-leak / redacted trace; docs |
| Sun Oct 25 | Human-review rate metric: share of runs that abstain, shown in the report next to pass rate, plus a regression-gate option on abstain-rate drift | Tests; README example; CHANGELOG |
| Continuation | Optional: run one case set against a real free model through the existing LLM interface, and record the results and limitations honestly | Results file + "what this does not prove" note |

Rules for every update: offline tests by default, no real personal data, no claims the code does not support, and one focused commit with a CHANGELOG line.
