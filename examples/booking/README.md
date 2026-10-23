# Booking/rescheduling example (synthetic)

Recorded with the built-in toy agent:

    python3 -m agent_trace_evals run cases/booking_requests.json --runs 20 --out examples/booking/clean
    python3 -m agent_trace_evals run cases/booking_requests.json --runs 20 --flakiness 0.3 --out examples/booking/flaky

Each run uses its run index (0-19) as the seed, so both reports are reproducible. With `--flakiness`, booking answers can be
truncated (invalid JSON), lose the `needs_human` field, get `needs_human` as a string, or lose the `REVIEW_REQUIRED` marker
on change/cancel/incomplete requests. The "Failures by assertion" section shows which check caught each failure.

What this does not prove: it is a deterministic toy agent on synthetic requests, so it shows that the checks catch these
failure modes, not how any real model behaves. Only `report.md` is kept; traces are written next to it when you re-run.
