---
name: verify
description: "Prove a change is done: define done, run the real tests, and report exact commands and observed results."
aliases: verification, quality-gate
license: MIT
source: oh-my-hermes (MIT) — adapted
---

# Verify

Verification turns "done" from an opinion into evidence. The evidence is the
command, the exit status, and the output that proves the claim. A plan, a
green check from yesterday, or another tool's summary is not verification.

## Use this skill when

- You are about to claim a change is complete, fixed, safe, or ready to merge.
- The user asks whether tests pass, whether a fix works, or whether a release
  is ready.
- A change touches security, data, generated files, or shared contracts and
  needs a broader gate.

## Do not use this skill when

- The user asks for a code review or a bug diagnosis; those produce findings,
  not a completion gate.
- There is no change or claim to verify. Do not run the whole suite at random.

## Workflow

### 1. State the claim and the risk

Write one sentence: "This change does X for scenario Y without changing Z."
Name the risk surface: internal logic, public API, CLI, data/migration,
authentication/authorization, payment, generated artifact, dependency, or
deployment. The risk surface decides how broad the gate must be.

### 2. Define done from the repository

Read the repository's own instructions and tooling before inventing commands:

- `AGENTS.md`, `README`, `CONTRIBUTING`, and `Makefile`.
- Package manifests: `pyproject.toml`, `package.json`, `Cargo.toml`, `go.mod`.
- CI workflows and pre-commit configuration.
- Test layout and existing test names.

Build a small verification matrix. Use only checks that exist:

| Area | Command | Required? |
| --- | --- | --- |
| Format/lint | repository command | if configured |
| Typecheck | repository command | if configured |
| Unit tests | targeted file or package | yes |
| Integration/e2e | affected suite | if behavior crosses boundaries |
| Build | package/build command | yes |
| Security | dependency or secret scan | if configured or risk is high |
| Generated files | regenerate and compare | if generated paths changed |
| Docs | link or docs check | if docs changed |

`task` can map test files and CI commands in parallel, read-only. It cannot
substitute for running the commands in the working tree.

### 3. Establish the baseline when needed

If a check fails, determine whether the change caused it. Run the same command
at the merge base in a separate detached worktree, without touching the user's
working tree:

```sh
git worktree add --detach /tmp/verify-base <merge-base>
# run the same command there
git worktree remove /tmp/verify-base
```

Never stash, reset, or check out the user's dirty tree to obtain a baseline.
If the failure reproduces identically at the base, report it as pre-existing,
with the command and output. If the check fails before it reaches the changed
code (collection error, missing dependency message), the change was not
exercised; treat that check as missing evidence, not as a pass.

### 4. Run the real checks

Run the smallest check that proves the change, then widen for shared surfaces.
Capture the exact command, working directory, exit code, and a short output
summary. Do not use a substituted command, a mock harness, or a hand-written
"PASS" when a real test exists.

Rules:

- Run tests, do not describe them.
- Run the new regression test and the surrounding suite.
- Confirm the test fails without the fix when practical.
- Check that tests assert behavior, not only that a mock was called.
- Never delete, skip, or weaken a failing test to get green.
- Re-run after the last edit. A green run before the last change is stale.
- Keep unrelated user changes; do not clean the tree to make checks pass.

### 5. Check the definition of done

A change is done only when all applicable items hold:

- [ ] The requested behavior exists and is covered by a test or a specific
      observed exercise.
- [ ] The failure or edge case that motivated the change has a negative test.
- [ ] Existing tests for the touched area pass.
- [ ] Format, lint, typecheck, and build pass where configured.
- [ ] No skipped tests, debug prints, temporary files, or commented-out code
      remain in the diff.
- [ ] No secrets, tokens, private URLs, or credentials were added.
- [ ] Generated files were regenerated from their source and match.
- [ ] Public interfaces, migrations, and compatibility are handled.
- [ ] Documentation or changelog updated if the repository expects it.
- [ ] The final diff was read, not just the test output.

### 6. Report evidence, not adjectives

Use this shape:

```
Claim: <what is now true>
Result: PASS | HOLD | BLOCK
Checks:
  - <command> -> exit 0, <summary>
  - <command> -> exit 1, <failure summary> (pre-existing at <base>)
Not run:
  - <check> - <reason>
Tree: <commit or working tree state when checks ran>
Risks: <remaining uncertainty>
```

Verdicts:

- PASS: every required check passed on the current tree, with fresh output.
- HOLD: a required check is missing, stale, blocked by an external dependency,
  or failing for a known pre-existing reason. Name the exact gap and how to
  close it.
- BLOCK: the change fails a required check or has an unaddressed security,
  data, or compatibility defect.

PASS never stands next to a failing required check, regardless of origin.
Also verify that deleted validation, refusal, permission, or sanitization
checks still have an equivalent guard and a negative case proving it refuses
what it should.

## Failure modes

- Saying "tests pass" without running them or naming the command.
- Reporting a subset as if it were the full suite.
- Reusing output from before the last edit.
- Treating a missing check as a pass.
- Blaming a failure on the environment without a baseline run.
- Making the test pass by mocking the behavior under test.
- Broadening the claim from "this unit passes" to "the release is ready".
- Cleaning or resetting the working tree to avoid unrelated failures.
- Claiming CI, review, or merge status that was not observed.
