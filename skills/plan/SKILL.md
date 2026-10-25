---
name: plan
description: Turn a request into a small, ordered plan with explicit assumptions and testable acceptance criteria.
aliases: planning, implementation-plan
license: MIT
source: oh-my-hermes (MIT) — adapted
---

# Plan

A plan is a short contract for changing a codebase. It states what will change,
what will not, what is assumed, and how completion will be recognized. It is
not an essay and it is not an implementation.

## Use this skill when

- The task spans several files, steps, or verification commands.
- Requirements are incomplete, ambiguous, or carry hidden constraints.
- The change is risky: public interfaces, data, security, migrations, releases.
- The user asks for a plan, breakdown, estimate, or safe path before coding.

## Do not use this skill when

- The request is a one-line fix with an obvious check. Do the work instead.
- The user wants immediate implementation and the change is small; make a
  three-line plan inline, then proceed.
- The work is a pure question or explanation.

## Workflow

### 1. Frame the request

Restate the desired outcome in one paragraph: who is affected, what behavior
changes, and what must stay the same. Quote the user's acceptance language
rather than replacing it with your own.

Inspect the repository before planning:

- `list_files` at the root, then read `README`, `AGENTS.md`, `CONTRIBUTING`,
  and the package/build manifest.
- `glob` for source, test, and configuration layout.
- `grep` for the names, routes, commands, or errors the request mentions.
- Read the files that will be edited and their closest callers.

Use `task` for parallel, read-only reconnaissance when the repository is
large: one subagent maps architecture, one maps tests and commands, one finds
conventions. Ask for file paths and line numbers. Do not delegate edits, and
verify important claims yourself before relying on them.

### 2. Separate facts from assumptions

Record three lists:

- Facts: observed in the repository or supplied by the user.
- Assumptions: needed to proceed but not confirmed.
- Unknowns: could change the design and cannot be resolved from the repo.

For every assumption, write the impact if it is wrong and the cheapest way to
check it. Ask the user only when an unknown blocks the design or could cause
data loss or public breakage. Otherwise state the assumption and proceed.

### 3. Define acceptance criteria

Each criterion must be observable by a command, test, or specific inspection.
Replace vague goals with evidence:

- Bad: "login works."
- Good: "POST /login with a valid password returns 200 and sets a session
  cookie; an invalid password returns 401 and no cookie; both are covered by
  tests in tests/test_login.py."
- Bad: "performance is fine."
- Good: "the list endpoint returns 1000 rows in under 300 ms locally with the
  existing benchmark command."

Include negative cases, compatibility constraints, migrations, and any
behavior that must not change. Name the command that will prove each item.

### 4. Decompose into ordered steps

Prefer small, vertical steps that leave the tree working:

1. Add or update the failing test or characterization test.
2. Make the smallest implementation change.
3. Update callers or schema.
4. Run the targeted check.
5. Update docs or generated outputs if required.

For each step name:

- the action and the files likely touched;
- the verification command or observation;
- dependencies on earlier steps;
- the rollback or recovery point for risky steps.

If a step cannot be verified on its own, split it. If two steps touch the same
files for different reasons, consider whether they belong in separate changes.

### 5. Check risks and alternatives

Name the blast radius: internal only, public API, persisted data, security
boundary, generated artifact, or deployment. For each risk, state mitigation
and how it will be detected. Record one or two alternatives that were rejected
and why, especially when the user may prefer a different tradeoff.

### 6. Write the plan

Use this compact format:

```
Goal: <one sentence>
Non-goals: <what is explicitly out of scope>
Facts: <observed constraints>
Assumptions: <assumption -> validation -> impact>
Steps:
  1. <action> [files] -> verify: <command or observation>
  2. ...
Acceptance criteria:
  - <observable criterion> -> <proof>
Verification: <commands to run, in order>
Risks: <risk -> mitigation>
Open questions: <only decisions the user must make>
```

Keep implementation detail out of the plan. Signatures, test names, and
commands are useful; full function bodies are not. If the plan is longer than
the code it describes, cut it.

### 7. Sanity-check before handoff

- Every acceptance criterion maps to a step and a proof.
- Every step has a check that can fail.
- No hidden "and then make it robust" step remains.
- Ordering respects tests, migrations, and compatibility.
- The plan states what is not being done.

Present the plan, then wait for approval if the user asked for a plan. If the
user asked to implement, start at step 1 and update the plan when reality
diverges.

## Completion checklist

- [ ] Outcome, scope, and non-goals stated.
- [ ] Assumptions visible with validation and impact.
- [ ] Acceptance criteria observable and testable.
- [ ] Steps ordered, small, and individually verifiable.
- [ ] Verification commands named.
- [ ] Risks and rollback identified.
- [ ] No implementation smuggled into the plan.

## Failure modes

- Planning by guessing: reads no code and invents architecture.
- A plan with no acceptance test, so "done" becomes an opinion.
- Hiding assumptions until implementation fails.
- Copying a generic template instead of the repository's actual commands.
- A single enormous step that cannot be reviewed or rolled back.
- Writing code inside the plan and calling it design.
- Ignoring compatibility, migrations, or generated files until the end.
- Asking the user questions that the repository already answers.
