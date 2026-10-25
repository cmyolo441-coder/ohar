---
name: refactor
description: "Change structure without changing behavior: test first, move in small verified steps, keep rollback easy."
aliases: refactoring, refactor-plan
license: MIT
source: oh-my-hermes (MIT) — adapted
---

# Refactor

A refactor changes structure while preserving observable behavior. If behavior
changes, that is a feature or a bug fix and must be named, tested, and reviewed
as such. A large refactor is a sequence of small, green, reversible steps.

## Use this skill when

- Moving, renaming, splitting, or merging modules.
- Replacing an implementation while keeping the same contract.
- Removing duplication or untangling dependencies with no intended behavior
  change.
- A change has become too large to review safely in one step.

## Do not use this skill when

- The goal is a new feature or a bug fix. Do those first or separately.
- The direction of the architecture is still undecided. Plan the target shape
  before sequencing phases.
- The work is a dependency upgrade with breaking changes; treat it as a
  migration with its own compatibility plan.

## Workflow

### 1. Define the behavior contract

Write down what must remain true:

- public functions, commands, routes, and response shapes;
- input validation and error behavior;
- persisted data formats and migrations;
- side effects, ordering, and concurrency guarantees;
- performance characteristics that users depend on;
- configuration and environment behavior.

If behavior is not covered by tests, write characterization tests first. Those
tests capture current behavior, including quirks you intend to preserve. Do not
"clean up" behavior while capturing it.

Run the baseline:

```sh
<targeted test command>
<typecheck or build command>
```

Record any pre-existing failures before changing code.

### 2. Map the blast radius

Use `grep`, `glob`, and `read_file` to find:

- direct callers of every symbol that moves or changes;
- tests that assert the current structure or behavior;
- configuration, routes, serializers, and documentation that reference it;
- generated files that depend on the source;
- external consumers, if the package is published.

Use `task` for parallel read-only discovery when the surface is large: one
subagent maps call sites, one maps tests, one maps configuration and docs.
Require file:line references and search terms. Do not let subagents edit; verify
their lists with your own searches before acting.

Assess the blast radius: local module, shared library, public API, persisted
data, or deployment. The radius determines the verification gate for each step.

### 3. Choose the target and phase order

Define the end state in one paragraph. Then order phases contracts-first:

1. Types, interfaces, and schemas.
2. Implementations behind the existing contract.
3. Callers, in reviewable groups.
4. Tests and fixtures.
5. Removal of old paths and cleanup.

Each phase must leave the tree working and end at a commit that could ship. If
a phase cannot be made independently green, split it again.

Build a files table before editing:

| Path | Action | Phase | Depends on |
| --- | --- | --- | --- |
| src/foo.py | split | 1 | - |
| src/bar.py | update caller | 3 | src/foo.py |
| tests/test_foo.py | update imports | 4 | src/foo.py |

A file without a phase is unplanned work.

### 4. Refactor in small steps

Rules:

- One mechanical transformation per step: rename, extract, move, or inline.
  Do not mix a move with a logic edit.
- Keep the old behavior available until all callers migrate. A compatibility
  shim is acceptable when it is explicit and temporary.
- Update call sites in the same step as the contract change, or keep both
  paths working until the migration is complete.
- Run the targeted test after every step, not only at the end.
- Keep separate commits for mechanical moves and behavior changes.
- Do not delete the old path until the new path is verified and no callers
  remain, confirmed by grep.

If a step grows beyond the plan, stop and re-plan. A refactor that cannot be
paused safely is too large.

### 5. Preserve behavior

Check the edges, not only the happy path:

- empty, null, and malformed inputs;
- error types, messages, and status codes;
- ordering and idempotency;
- default values and configuration precedence;
- resource cleanup and cancellation;
- serialization formats and compatibility;
- permissions and trust-boundary checks.

Run the characterization tests against the refactored code. If a test must
change, explain why the observable contract changed. "The test was coupled to
the old structure" is valid only if the behavior assertion is preserved.

### 6. Verify each phase

For every phase:

1. Run the targeted tests.
2. Run typecheck, lint, and build for the touched package.
3. Search for remaining references to removed or renamed symbols.
4. Regenerate generated artifacts and confirm no drift.
5. Read the diff as a reviewer would.

At the end, run the broad suite. Compare performance only if the refactor
could affect it, using the same profile and workload as the baseline.

### 7. Clean up last

Deletion and cosmetic cleanup come after the new path is proven:

- remove dead code only when grep shows no callers;
- remove compatibility shims with a named follow-up if they must outlive this
  change;
- update documentation and examples;
- delete obsolete tests only when their behavior is covered elsewhere.

Tag or note a rollback point before the cleanup phase. Nothing is deleted
before the replacement is verified.

## Completion checklist

- [ ] Behavior contract written and covered by tests.
- [ ] Baseline commands and pre-existing failures recorded.
- [ ] Blast radius mapped with file:line evidence.
- [ ] Phase order contracts-first, each phase independently shippable.
- [ ] Files table complete with actions and dependencies.
- [ ] Each step is mechanical or behavioral, not both.
- [ ] Targeted tests run after every step.
- [ ] Old paths removed only after zero callers are confirmed.
- [ ] Full suite, typecheck, lint, and build pass at the end.
- [ ] Rollback point exists for the cleanup phase.
- [ ] No unintended behavior change is hidden in the diff.

## Failure modes

- Mixing a rename, a move, and a logic change in one commit.
- Refactoring without tests, then discovering the behavior changed.
- A big-bang rewrite with no green intermediate state.
- Updating the happy path but missing error, empty, or concurrency behavior.
- Deleting the old path before all callers are migrated.
- Changing a public contract and calling it internal cleanup.
- Letting subagents edit overlapping files.
- Declaring success from a targeted test while the full suite is red.
- Rewriting tests to match the new code without checking the behavior.
