---
name: debug
description: "Debug failing or flaky tests and wrong behavior: reproduce, isolate, fix, and verify the cause."
aliases: debugging, root-cause
license: MIT
source: oh-my-hermes (MIT) — adapted
---

# Debug

Debugging is a sequence, not a guess: reproduce, observe, isolate, fix, verify.
A change made before the failure is demonstrated is a guess with a diff attached.

## Use this skill when

- Code returns a wrong value, crashes, hangs, leaks, or behaves differently
  across runs or environments.
- A test is flaky, order-dependent, or passes locally but fails in CI.
- The user reports a bug but the cause is unknown.

## Do not use this skill when

- The cause is already demonstrated and the fix needs implementation; go fix it
  and verify with this skill's final step.
- The task is a feature request or a review with no failing behavior.

## Workflow

### 1. Reproduce

State the symptom and the expected behavior separately. Write down:

- the exact command, request, or test;
- the input and environment;
- the observed output or assertion;
- the expected output and where that expectation comes from;
- the failure rate over N runs for intermittent faults.

Start with the smallest command that shows the problem. Capture the full error,
stack trace, exit code, and relevant logs. Confirm that the running code is the
code you are reading: print the module path or version from inside the failing
process, and check for stale builds, cached bytecode, or a different virtual
environment.

For a flaky failure, measure before touching anything:

```sh
for i in $(seq 50); do <command> || echo "FAIL $i"; done
```

A single sighting is a report, not a reproduction. If you cannot reproduce,
make reproduction the first problem to solve: add an assertion or log at the
boundary, identify the missing precondition, or report exactly what is unknown.

### 2. Form competing hypotheses

Write at least three hypotheses on distinct axes before choosing an
observation. Three phrasings of one guess is still one hypothesis.

- Input and state: a value, fixture, cache, or database row differs.
- Ordering: test order, iteration order, or arrival order matters.
- Timing: a deadline, sleep, clock read, or scheduler race.
- Shared state: two workers share a file, port, row, global, or connection.
- Environment: dependency version, OS, build flag, or configuration differs.
- Dependency behavior: a library or remote service does not behave as assumed.

For each hypothesis, state the observation that would refute it and the
cheapest way to obtain that observation. Prefer an observation that separates
several hypotheses at once. Record which hypotheses each result eliminates.

### 3. Observe and isolate

Change one variable at a time. Useful moves:

- Add assertions or structured logs at the boundary where good data becomes
  bad, not everywhere.
- Run the failing case alone, then in the suite, then with shuffled order.
- Pin, stub, or swap the suspected dependency and compare.
- Bisect the input, data set, or commit range.
- Inspect the smallest failing unit and trace the value backward to its source.

Avoid observations that change timing, such as adding a print inside a race.
Prefer counters, recorded traces, or post-mortem assertions. A bug that
disappears when observed is a timing fault, not a fixed bug.

Use `read_file`, `grep`, and `shell` to inspect call sites, configuration, and
logs. Use `task` for independent read-only searches when several subsystems
must be mapped at once; do not delegate the diagnosis or the fix.

### 4. Name the root cause

Write the causal chain: trigger -> state change -> failure. Distinguish:

- the symptom (what the user sees);
- the trigger (the input or event that exposes it);
- the mechanism (why the code produces the wrong result);
- the root cause (the assumption or defect that made the mechanism possible).

If the explanation does not account for the failure rate or for every part of
the symptom, it is incomplete. If a rival hypothesis is still standing, design
one more discriminating observation rather than promoting the survivor.

### 5. Fix

Make the smallest change that addresses the demonstrated root cause. Do not
clean up unrelated code, reformat files, or refactor inside the fix.

Add a regression test that fails before the change and passes after it. For an
intermittent bug, the test must reproduce the schedule or the state that
caused it; a test that passes only because it sleeps is a mask.

If the fix changes behavior beyond the bug, stop and confirm the intended
behavior. If three fix attempts fail against the same reproduction, stop
patching: the shared assumption is now the suspect. Re-examine ownership,
ordering, and boundaries, and present the next step as a decision.

### 6. Verify

Run the original reproduction, not a substitute. For a flaky bug, run the same
loop with the same number of iterations and compare the rate before and after.
Then run the closest test suite and any check the change could affect.

Report:

- the reproduction command and observed before/after output;
- the regression test and its result;
- the broader checks run, with exit status;
- what was not run or remains uncertain.

## Debug log template

```
Symptom: <observed wrong behavior>
Expected: <correct behavior and source>
Reproduction: <command> -> <output>, rate: <n/N>
Environment: <versions, OS, config>
Hypotheses:
  H1 <claim> -> refuted by <observation>
  H2 <claim> -> supported by <observation>
  H3 <claim> -> still open
Root cause: <trigger -> mechanism -> defect>
Fix: <minimal change>
Verification: <command> -> <result>
Not verified: <gaps>
```

## Completion checklist

- [ ] Reproduction observed, with a rate for intermittent faults.
- [ ] Running code confirmed to match the source being read.
- [ ] At least three distinct hypotheses written.
- [ ] Each observation records what it eliminated.
- [ ] Root cause explains the whole symptom.
- [ ] Regression test fails before and passes after.
- [ ] Original reproduction and relevant suites re-run.
- [ ] No symptom masks such as sleeps or retries left as the fix.

## Failure modes

- Patching the first plausible cause without reproducing.
- Adding a sleep, retry, or catch-all to make the symptom disappear.
- Changing several variables at once, so no result is attributable.
- Trusting a log line without checking which process or version wrote it.
- Calling it fixed because one run passed after a flaky failure.
- Fixing the symptom at the call site while the corrupt state remains.
- Continuing to patch after three failed fixes against the same reproduction.
