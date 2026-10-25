---
name: docs
description: "Write accurate README, API, usage, and runbook docs with verified commands and examples."
aliases: documentation, readme, runbook
license: MIT
source: oh-my-hermes (MIT) — adapted
---

# Docs

Documentation is part of the product. A reader should be able to install, run,
configure, and recover the system without reading the source. Every command and
example is a claim that must be executed before it is published.

## Use this skill when

- Creating or revising a README, quickstart, or installation guide.
- Documenting an API, CLI, configuration format, or error code.
- Writing a runbook, operations guide, or changelog.
- A change alters behavior and existing docs may now be wrong.

## Do not use this skill when

- The user wants marketing copy, a landing page, or persuasive content.
- The documentation target is a live third-party product; use web research to
  verify current facts instead of inventing them.
- The code does not yet work. Fix or verify it first; documenting broken behavior
  only spreads the defect.

## Workflow

### 1. Identify the reader and the job

State who will read the page and what they need to accomplish. A new contributor
needs different content from an operator paged at 3 a.m. Choose the document type
before drafting:

- **README** - what it is, why it exists, quickstart, links.
- **Tutorial** - a guided first success from a clean environment.
- **How-to** - a task with prerequisites and exact steps.
- **Reference** - complete options, endpoints, fields, status codes, limits.
- **Runbook** - symptoms, checks, remediation, rollback, escalation.
- **Changelog** - user-visible changes grouped by version and type.

### 2. Verify the source of truth

Read the code, configuration schema, CLI help, tests, and deployment files. Run
the commands in a clean temporary directory or container with the documented
prerequisites. Note the exact versions used. When code and prose disagree, the
tested behavior wins; fix the prose or report the code defect.

Do not describe an environment you have not observed. If a step needs a service,
credential, or platform you cannot access, mark it unverified and say what is
missing.

### 3. Structure the README

A useful default order:

1. name and one-sentence purpose;
2. the problem it solves and who it is for;
3. status and supported versions;
4. install or prerequisites;
5. a minimal quickstart that produces a visible result;
6. common usage with real inputs and outputs;
7. configuration table with defaults and required fields;
8. API or CLI reference link;
9. troubleshooting and error recovery;
10. contributing, security reporting, license, and changelog links.

Put the shortest path to success near the top. Long background sections belong
after the quickstart, not before it.

### 4. Write examples that can be copied

- Use a realistic resource name, not `foo` and `bar` everywhere.
- Show the command, the expected output, and the exit status when it matters.
- State the working directory and shell where it affects the result.
- Use placeholders such as `<project-id>` consistently and define them once.
- Keep line lengths and output snippets short enough to read in a terminal.
- Never include real tokens, passwords, internal hostnames, or customer data.
- Show the failure case for destructive or security-sensitive operations.

For API reference, document method and path, auth and scopes, path/query/body
parameters with types and limits, success response, error envelope, idempotency,
pagination, rate limits, and a complete curl example. Use one consistent example
resource across the page.

### 5. Document operations and change

A runbook should be usable under stress:

```
Symptom: 5xx rate above 2% on checkout
Check: dashboard, recent deploys, dependency status, database health
Act: rollback to the previous release if the error started with a deploy
Verify: error rate returns below baseline for 10 minutes
Escalate: platform on-call if rollback does not recover within 15 minutes
```

A changelog should separate added, changed, deprecated, removed, fixed, and
security entries. Link the version, date, and upgrade notes. Do not rewrite
history; correct it with a new entry.

### 6. Review the docs as a product

- Run every command from a clean starting point.
- Open every relative link and check that it resolves.
- Search for stale version numbers, renamed flags, and removed endpoints.
- Confirm the table of contents and headings match the actual structure.
- Read the quickstart as a newcomer: is any step implicit?
- Check that error messages and screenshots match the current UI or output.
- Update docs in the same change as the behavior, or file a tracked follow-up.

## Completion checklist

- [ ] Reader, task, and document type named.
- [ ] Commands run against the documented version and environment.
- [ ] Quickstart reaches a visible result from a clean state.
- [ ] Examples use safe, realistic values and consistent placeholders.
- [ ] API reference covers auth, errors, limits, and pagination.
- [ ] Runbook has symptom, check, action, verification, and escalation.
- [ ] Changelog groups changes and includes upgrade notes.
- [ ] Links, versions, flags, and output checked for staleness.
- [ ] No secret or private data appears in an example.

## Failure modes

- Publishing commands that were never executed.
- A quickstart that assumes an undocumented prerequisite.
- Copying generated API output without checking the current schema.
- Examples with `YOUR_API_KEY` committed as if it were a real value.
- A runbook that says "investigate" without a concrete check.
- Rewriting a released changelog entry instead of adding a correction.
- Hiding limitations or unsupported platforms to make the product look simple.
- Letting docs drift because the change was "too small to document".
