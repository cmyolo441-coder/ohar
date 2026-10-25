---
name: test
description: "Design meaningful tests: boundaries, failure paths, regressions, and real commands instead of mocks."
aliases: testing, test-design, coverage
license: MIT
source: oh-my-hermes (MIT) — adapted
---

# Test

Tests are executable claims about behavior. A good suite proves the contract at
its boundaries, fails clearly when the contract breaks, and runs in a time that
developers will tolerate. Coverage percentage is a diagnostic, not a goal.

## Use this skill when

- Adding a feature that needs tests, or fixing a bug that needs a regression test.
- A suite is slow, flaky, over-mocked, or passing while production fails.
- Deciding what belongs in a unit test versus an integration or end-to-end test.
- Reviewing whether a change is actually proven by its tests.

## Do not use this skill when

- You only need to run an existing check and report the result. Run it and
  report; this skill is for designing or improving tests.
- The behavior is visual or accessibility-specific; combine this with the
  accessibility or web skill rather than asserting style in a unit test.
- The code is not yet understood well enough to name the contract. Read it first.

## Workflow

### 1. State the behavior under test

Write one sentence: given this input and state, the system should produce this
observable result. Include what must not change. For a bug fix, reproduce the
reported failure before touching the implementation.

### 2. Choose the right level

- **Unit** - pure logic, parsing, calculations, state machines, validation.
  Fast, deterministic, no network or clock.
- **Integration** - real database, filesystem, HTTP boundary, queue, or CLI.
  Proves wiring and serialization that mocks cannot.
- **End-to-end** - critical user journeys only. They are expensive; do not use
  them to cover every branch.
- **Contract** - API schemas, error shapes, and compatibility with consumers.

Push assertions to the lowest level that can observe them. Use integration tests
for code that mainly composes libraries; mocking everything proves the mock.

### 3. Enumerate cases before writing code

For each behavior, list:

- the happy path with a realistic input;
- boundaries: zero, one, maximum, off-by-one, empty string, empty list;
- invalid input: wrong type, malformed, missing, too large, unsupported version;
- state: first use, repeat use, concurrent use, retry, cancellation;
- failure injection: timeout, connection reset, partial write, disk full,
  permission denied, malformed upstream response, rate limit;
- security negatives: unauthenticated, wrong tenant, unauthorized object,
  injected path, oversized payload;
- regressions: the exact input that failed in the bug report.

Keep a short table of case -> expected result -> test level. Drop duplicates.

### 4. Make tests deterministic

Control every source of nondeterminism:

- freeze or inject the clock; do not assert on `now()` directly;
- seed random generators and report the seed on failure;
- use a temporary directory, not a shared fixture path;
- bind test servers to port 0 and read the assigned port;
- replace network calls at the boundary, not deep inside the code;
- avoid sleeps for synchronization; use events, polling with a deadline, or
  dependency injection;
- clean up resources in a finally block or fixture teardown.

A test that fails once in twenty runs is a defect. Record the seed and run it in
a loop to reproduce before quarantining it.

### 5. Write the tests

Use the repository's existing framework and style. The arrange-act-assert shape
should be obvious. Assert on observable behavior, not private methods. Prefer one
logical assertion per test name; a test name should read as the behavior.

For a bug fix, the sequence is:

1. write the failing regression test;
2. run it and confirm it fails for the reported reason;
3. fix the cause;
4. rerun the test and the nearest broader suite;
5. keep the test even if the fix is later refactored.

### 6. Run real commands

Inspect the manifest or Makefile first, then use the project's own commands:

```sh
python -m pytest -q tests/test_feature.py
python -m pytest -q
npm test -- --run
go test ./...
cargo test
```

Run the smallest relevant test while iterating, then the full suite before
claiming completion. Record the command, exit status, and a short result. If a
test cannot run because a service is missing, report that as a gap, not a pass.
Use coverage only to find untested branches, never to replace behavioral cases.

### 7. Review coverage and gaps

Ask which risks remain unproven: error handling, migrations, concurrency,
timeouts, permissions, serialization, and rollback. Add tests for the highest
risk, not for the easiest line. Note intentionally untested areas and why.

## Completion checklist

- [ ] Behavior and expected result stated before implementation.
- [ ] Test level chosen deliberately; mocks limited to real boundaries.
- [ ] Boundary, invalid, failure, and security-negative cases covered.
- [ ] Regression test fails before the fix and passes after it.
- [ ] Time, randomness, network, and filesystem are controlled.
- [ ] No fixed sleeps or order-dependent tests.
- [ ] Targeted and full suites run with observed results.
- [ ] Remaining coverage gaps named.

## Failure modes

- Asserting that a mock was called instead of that behavior changed.
- Testing implementation details that refactoring will break.
- One giant end-to-end test standing in for all unit coverage.
- A flaky test retried in CI instead of diagnosed.
- A regression test that passes before the fix, proving nothing.
- Chasing 100% coverage while error paths remain untested.
- Treating "tests are green" as true when the command was never run.
- Ignoring time zones, locale, or database state that makes the test environment-specific.
