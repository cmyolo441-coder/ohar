---
name: onboard
description: "Map an unfamiliar repository fast: structure, entry points, tests, and the safest first change."
aliases: onboarding, codebase-tour
license: MIT
source: oh-my-hermes (MIT) — adapted
---

# Onboard

Onboarding produces a usable map, not a tour of every file. Read the smallest
set of files that explains how the system is built, how it runs, and how it is
verified. Separate what you observed from what you inferred.

## Use this skill when

- You are starting in an unfamiliar repository.
- The user asks how the codebase works, where to begin, or what is safe to
  change first.
- You need enough context to plan an implementation without wandering.

## Do not use this skill when

- The user named a concrete bug or feature and the relevant code is already
  known. Go to plan or debug.
- The task is a code review of a specific diff.

## Workflow

### 1. Set the scope

Decide the depth before reading:

- Quick map: 10-15 minutes, enough to run one thing and find one area.
- Working map: entry points, main modules, tests, and conventions for a task.
- Deep onboarding: architecture, data flow, risks, and first-task runway.

Name the output the user wants and stop when it is delivered. Do not read the
entire repository by default.

### 2. Read the repository's own instructions

Start at the root with `list_files`, then read, in order:

1. `README` - the product claim and the first-run path.
2. `AGENTS.md`, `CONTRIBUTING`, or another repository instructions file - how
   work is expected to be done and which commands are mandatory.
3. Build and package manifests - language, version, dependencies, entry
   points, scripts.
4. CI workflows and test configuration - what "green" means.
5. `docs/` index - architecture, decisions, and operational notes.

If those files disagree, report the disagreement rather than choosing silently.

### 3. Map the tree by role

Use `list_files` and `glob` to classify top-level and second-level directories:

- source code and public entry points;
- tests and fixtures;
- configuration and deployment;
- documentation and decisions;
- scripts and developer tooling;
- generated, vendored, or build output;
- data, migrations, and fixtures.

Do not descend into vendored dependencies, build output, or generated trees
unless the question is specifically about them. Note their locations so you
can avoid confusing them with source.

### 4. Find the entry points

Identify how work enters the system:

- CLI command registration and argument parsing;
- server bootstrap, routes, middleware, and handlers;
- library exports and public API surface;
- scheduled jobs, workers, or event consumers;
- configuration loading and environment variables.

Trace one path end to end: from entry point to the module that performs the
work to the place where state or output changes. This one trace teaches more
than a directory listing.

### 5. Map verification

Find:

- the fastest targeted test command;
- the full suite command and its expected duration;
- lint, typecheck, format, and build commands;
- how tests are named and where fixtures live;
- CI jobs and any required generated-file checks;
- known slow, flaky, or environment-dependent tests.

Read a representative test to learn the project's testing style. Run one
targeted test only if dependencies are already available; do not install
packages, mutate configuration, or start services without permission.

### 6. Parallelize exploration

Use `task` for read-only subagents when the repository is large or the
questions are independent:

- Subagent A: architecture and dependency flow. Return module boundaries,
  entry points, and import direction with file:line references.
- Subagent B: verification map. Return commands, test layout, CI jobs, and
  gaps with file:line references.
- Subagent C: configuration, data, and security surfaces. Return where secrets
  are read, where persistence happens, and which paths are trust boundaries.
- Subagent D: recent history. Return the last meaningful changes, ownership
  hints, and areas with active churn.

Give each subagent a bounded question and a read-only instruction. Ask for
uncertainties separately from findings. Reconcile conflicts by reading the
source yourself; a subagent's summary is a lead, not proof.

### 7. Produce the map

Report in this order:

```
Purpose: <what the system does, in one paragraph>
Stack: <language, runtime, framework, package manager>
Run: <command to start it, if known>
Test: <fast command and full command>
Entry points:
  - <path:line> - <what enters here>
Core modules:
  - <path> - <responsibility> - <depends on>
Data and state:
  - <path> - <what is stored or migrated>
Verification:
  - <command> - <what it proves>
Generated or vendored:
  - <path> - <how it is produced>
Risks and unknowns:
  - <risk> - <observed evidence or missing evidence>
Reading path:
  1. <file> - <why>
  2. <file> - <why>
First safe task:
  - <small change> - <files> - <test> - <why it is low risk>
```

Every claim about structure should carry a path. Mark inferences as
inferences. Say "not found" rather than inventing an owner or process.

### 8. Choose a safe first task

A good first task is small, local, covered by a test, and reversible. Examples:
add a missing validation with a test, fix a documentation command that no
longer works, add an assertion to an existing test, or remove a dead branch
confirmed by grep.

Avoid as a first task: dependency upgrades, auth changes, schema migrations,
large refactors, or anything that requires infrastructure you cannot observe.

## Completion checklist

- [ ] Repository instructions and build manifests read.
- [ ] Tree classified by role, including generated and vendored paths.
- [ ] Entry points traced to the code that does the work.
- [ ] Test and verification commands identified.
- [ ] Parallel exploration used where it saved time, with findings verified.
- [ ] Map separates observed facts from inferences and unknowns.
- [ ] A low-risk first task is proposed with its test.

## Failure modes

- Inferring architecture from directory names without reading code.
- Reading every file and producing a summary with no priorities.
- Confusing generated or vendored code with source.
- Claiming a run command works without running or citing its source.
- Installing dependencies or mutating the repo during orientation.
- Trusting a subagent summary that cites no files.
- Recommending a risky migration as a "first task".
- Producing a long tour that never says where to start.
