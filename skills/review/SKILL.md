---
name: review
description: Review a diff for correctness, security, and regressions; report findings with file-and-line evidence.
aliases: code-review, pr-review
license: MIT
source: oh-my-hermes (MIT) — adapted
---

# Review

Review the change, not the author's summary. The unit of review is the diff
plus the code and behavior it touches. Findings come first; a verdict without
evidence is noise.

## Use this skill when

- A pull request, branch, patch, or uncommitted diff needs review.
- The user asks for a bug hunt, risk audit, release check, or second opinion.
- A change touches security, data, public interfaces, or shared infrastructure.

## Do not use this skill when

- The user asks you to implement a feature. Review first if a diff exists,
  then implement separately.
- There is no target and no way to identify one. Ask for the branch, commit,
  files, or expected behavior before reviewing.

## Workflow

### 1. Establish the review target

Inspect the repository state before forming opinions:

```sh
git status --short
git branch --show-current
git log --oneline -5
git diff --stat
git diff
git diff --cached
```

If the change is on a branch, compare against the correct base:

```sh
git merge-base HEAD origin/main
git diff <base>...HEAD
```

Read the issue, plan, or user description. Write down the intended behavior in
one sentence. A diff that is clean but does not do what was asked is still a
finding.

### 2. Read the changed code in context

For every changed hunk, open the full function, class, or module. Check callers
and callees with `grep` and `read_file`. Do not review isolated lines as if the
surrounding invariants did not exist.

Ask:

- What inputs reach this code, including malformed, empty, large, and hostile?
- What does this code assume about callers, ordering, state, and failure?
- What happens on error, timeout, cancellation, retry, and partial failure?
- Does the change alter a contract other code depends on?
- Are generated files, lockfiles, schemas, or migrations consistent?

### 3. Run four lenses

**Correctness** - logic, types, boundary values, null/empty handling, state
transitions, concurrency, resource cleanup, off-by-one, and error propagation.
Look for tests that assert the wrong thing or pass for the wrong reason.

**Security** - trust boundaries, authentication, authorization, injection,
path traversal, unsafe deserialization, command construction, secrets in code
or logs, permissive CORS, weak crypto, and dependency changes. Verify that
deleted validation or refusal checks still have an equivalent guard and a
negative test.

**Regressions** - changed defaults, public API or CLI behavior, data formats,
migrations, configuration, performance, error messages, platform assumptions,
and compatibility with existing callers. Search the repository for usages of
every renamed or removed symbol.

**Evidence** - did the author run the relevant tests? Do the tests cover the
new behavior and its failure modes? Is CI green at this revision? Are claims
backed by commands, or only by prose?

### 4. Validate each candidate finding

Before reporting, re-read the exact lines and try to falsify the finding. Check
whether an earlier guard makes it unreachable, whether the test covers it, and
whether the behavior is intentional. Keep only findings you can defend with a
path, a line range, and a concrete scenario.

Use `task` for read-only parallel lenses on large diffs: one subagent traces
data flow, one checks security, one checks test coverage, one searches for
regressions. Give each a bounded scope and require file-and-line evidence.
Re-verify their claims yourself; subagent output is a lead, not a finding.

### 5. Write the review

Order findings by severity, most serious first. Use a stable format:

```
[P1] Unbounded retry loop on 429 responses
File: src/client.py:142-158
Evidence: retry_count resets on every response; the loop has no cap.
Impact: a throttled service can spin forever and exhaust file descriptors.
Fix: preserve the attempt count across responses and cap it.
Confidence: high
```

Severity guide:

- P0: data loss, security breach, or a broken build in the main path.
- P1: likely user-visible bug, regression, or exploitable weakness.
- P2: edge-case bug, missing validation, or maintainability risk.
- P3: minor cleanup, naming, or documentation.

Then report:

- **Verdict** - ship or no ship, with one sentence of reasoning.
- **Checked and clean** - surfaces actually inspected, so silence is not
  mistaken for coverage.
- **Not assessed** - areas you could not review and why.
- **Test gaps** - behavior not proven by an observed run.
- **Open questions** - only questions whose answers change the verdict.

Do not mix fix implementation into the review. If the user asks for fixes,
finish the review, then switch modes and edit with the findings as input.

## Completion checklist

- [ ] Diff base and intended behavior identified.
- [ ] Changed code read in context, including callers.
- [ ] Correctness, security, regressions, and evidence lenses applied.
- [ ] Every finding cites a path, line range, and scenario.
- [ ] Candidate findings re-checked for false positives.
- [ ] Findings ordered by severity; verdict follows the evidence.
- [ ] Checked-clean and not-assessed surfaces named.
- [ ] No edits made during review.

## Failure modes

- Reviewing the PR description instead of the diff.
- Reporting style preferences as bugs without impact.
- Speculating about a bug without reading the surrounding code.
- Trusting tests that were not run or that assert mocks, not behavior.
- Missing a deleted guard because only added lines were reviewed.
- Claiming "no issues" without naming what was checked.
- Burying a P0 under a summary or a list of nits.
- Approving a diff that does not satisfy the stated requirement.
