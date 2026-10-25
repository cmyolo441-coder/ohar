---
name: security
description: "Review code and configuration for exploitable security risks and propose evidence-based fixes."
aliases: security-review, appsec, threat-review
license: MIT
source: oh-my-hermes (MIT) — adapted
---

# Security

Security review is a defensive exercise: find realistic weaknesses, prove why
they matter, and make the smallest safe fix. Report risk, not fear. A finding
needs a location, an attack path, and evidence that the code is reachable.

## Use this skill when

- Reviewing a change, service, CLI, or configuration before release.
- Auditing secrets, authentication, authorization, input handling, or file and
  network access.
- Checking dependency, build, CI, container, or deployment posture.
- A user reports a possible vulnerability and wants triage and remediation.

## Do not use this skill when

- The request is offensive exploitation, credential theft, or bypassing another
  system's controls. Decline and offer defensive verification instead.
- The task is general code quality with no security boundary. Use review.
- A live incident needs containment first. Stabilize, preserve evidence, and
  involve the owner before deep code analysis.

## Workflow

### 1. Define the target and trust boundaries

Name what is being reviewed and where untrusted input enters: HTTP routes, CLI
arguments, files, environment variables, queues, webhooks, model output, and
third-party APIs. Identify identities, privileges, data stores, and secrets.
State what an attacker could gain: data, money, execution, availability, or
reputation. If the boundary is unknown, stop and ask rather than guessing.

### 2. Inventory the attack surface

Use `grep`, `glob`, and `read_file` to find:

- route and handler definitions, middleware, and authorization checks;
- subprocess, shell, SQL, template, and deserialization calls;
- file reads and writes, archive extraction, path joins, and upload handlers;
- outbound HTTP clients, URL fetching, redirects, and webhook receivers;
- credential reads, logging, error handlers, and debug flags;
- dependency manifests, lockfiles, Dockerfiles, CI workflows, and IaC.

Read the full call path, not just the changed line. Trace at least one hostile
input from entry point to sink.

### 3. Review the core risk classes

**Secrets** - hardcoded keys, tokens in fixtures, secrets in logs or errors,
secrets exposed to client bundles, long-lived credentials, missing rotation.
Never print a discovered value; report the path and a redacted fingerprint.

**Authentication and authorization** - every route has an explicit identity
check; object-level checks prevent IDOR; deny by default; sessions expire and
rotate; CSRF protection exists where cookies authorize writes; JWT algorithms,
audience, issuer, and expiry are validated.

**Injection** - SQL uses parameters; shell avoids string concatenation and
`shell=True`; templates autoescape; HTML output is encoded by context; LDAP,
NoSQL, and header injection are considered.

**SSRF and path traversal** - outbound URLs use an allowlist of scheme, host,
port, and resolved IP; redirects are rechecked; private ranges and metadata
endpoints are blocked. File paths are resolved, normalized, and confined to a
root; archives are checked for `../` and absolute paths.

**Deserialization and parsers** - untrusted pickle, unsafe YAML loaders, native
object deserialization, and unbounded XML are avoided or isolated. Parsers have
size, depth, and time limits.

**Dependencies and supply chain** - lockfiles are committed and reviewed;
versions are pinned; install scripts and unfamiliar packages are scrutinized;
known-vulnerability output is checked against the actual version; CI secrets are
scoped and not exposed to pull requests from forks.

**Deployment** - debug mode is off; TLS is enforced; CORS is explicit; security
headers are present; default credentials are changed; storage buckets and admin
endpoints are not public; containers run as non-root with minimal capabilities.

### 4. Validate each candidate finding

Before reporting, try to disprove it. Check whether an upstream guard, framework
default, gateway rule, or test makes the path unreachable. Confirm the vulnerable
version is actually deployed. If exploitation would require an already-compromised
host or a privileged user, say so and lower the severity.

Assign severity by impact and reachability, plus confidence:

- Critical: remote code execution, auth bypass, mass data exposure.
- High: privilege escalation, stored injection, secret leakage with impact.
- Medium: limited disclosure, CSRF on sensitive writes, weak controls.
- Low: hardening, defense in depth, information leakage with low impact.

### 5. Propose safe fixes

Prefer the smallest change that removes the root cause:

- parameterize queries; use argument arrays instead of shell strings;
- enforce authorization at the data-access boundary, not only in the UI;
- allowlist outbound destinations and revalidate redirects;
- replace unsafe deserialization with a data-only format;
- move secrets to a managed store and rotate anything exposed;
- add a negative regression test that fails before the fix and passes after.

Do not print, commit, or transmit a live secret. Do not rotate credentials or
change production configuration without explicit authorization. Hand those steps
to the owner with an ordered plan: issue new, deploy new, verify new, revoke old,
verify revoked.

### 6. Report with evidence

Use one block per finding:

```
[High] Object-level authorization missing on invoice download
Location: app/routes/invoices.py:41-58
Evidence: handler loads by id from the path; no owner or tenant check is made.
Attack path: authenticated user changes the id and receives another tenant's PDF.
Impact: cross-tenant data disclosure.
Fix: scope the query by tenant and add a 404 regression test.
Confidence: high
```

End with what was checked and found clean, what was not assessed, and residual
risk. Separate confirmed findings from hardening suggestions.

## Completion checklist

- [ ] Target, trust boundaries, and attacker capabilities stated.
- [ ] Every finding cites a path, line range, and reachable scenario.
- [ ] Candidate findings checked against upstream guards and deployed versions.
- [ ] Secret values never printed or copied into the report.
- [ ] Severity reflects impact, reachability, and confidence.
- [ ] Fixes address root cause and include a regression test.
- [ ] Credential rotation and production changes left to explicit authorization.
- [ ] Checked-clean and not-assessed surfaces named.

## Failure modes

- Reporting a scanner signature without tracing whether the code is reachable.
- Equating "uses crypto" with "is secure".
- Missing object-level authorization because the route has a login decorator.
- Trusting a client-supplied role, tenant, price, or file path.
- Printing a secret while "redacting" it.
- Calling a dependency vulnerable without checking the resolved version.
- Proposing a large rewrite where a parameterized query or guard is enough.
- Claiming a clean review without naming what was inspected.
