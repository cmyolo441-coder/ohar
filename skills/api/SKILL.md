---
name: api
description: "Design and review APIs for stable contracts, errors, pagination, auth, and compatibility."
aliases: api-design, api-review, contracts
license: MIT
source: oh-my-hermes (MIT) — adapted
---

# API

An API is a contract with callers you cannot see. Design the observable surface
deliberately: resources, methods, status codes, error shapes, limits, and
compatibility. Implementation details may change; the contract must not change
by accident.

## Use this skill when

- Designing a new HTTP, RPC, GraphQL, or CLI interface.
- Reviewing an API change for breaking behavior, missing validation, or unclear
  errors.
- Adding pagination, filtering, idempotency, versioning, or rate limits.
- Writing an OpenAPI/schema contract or consumer-facing documentation.

## Do not use this skill when

- The task is only internal refactoring with no observable interface change.
- The user needs a database schema migration plan; combine this with review and
  treat the storage change as its own risk.
- The interface is a private function inside one module. Test the behavior
  instead of inventing a public contract.

## Workflow

### 1. Name the consumers and the job

List who calls the API, from where, with what trust level, and what they need to
accomplish. Identify the compatibility window and whether old clients will exist
after the change. If consumers are unknown, say so and design conservatively.

### 2. Model resources and operations

Use nouns for resources and explicit actions only when an operation is not a
resource state change. Prefer:

- `GET /invoices/{id}` - read, safe, cacheable;
- `POST /invoices` - create, not idempotent unless keyed;
- `PUT /invoices/{id}` - full replace, idempotent;
- `PATCH /invoices/{id}` - partial update, define merge semantics;
- `DELETE /invoices/{id}` - idempotent in effect, define repeat behavior.

Use plural collection names, stable identifiers, and consistent casing. Keep
tenant, user, and object identity out of client-controlled fields when they can
be derived from auth.

### 3. Choose status codes deliberately

| Situation | Status |
| --- | --- |
| Success with body | 200 |
| Created resource | 201 plus Location |
| Accepted async work | 202 plus status URL |
| No content | 204 |
| Malformed syntax or type | 400 |
| Missing or invalid authentication | 401 |
| Authenticated but not allowed | 403 |
| Unknown resource, or hidden for authorization | 404 |
| Method not allowed | 405 |
| Conflict with current state | 409 |
| Validation failed semantically | 422 |
| Rate limited | 429 plus Retry-After |
| Unexpected server failure | 500 |

Do not return 200 with `{"success": false}` for an HTTP-level failure. Do not use
500 for client input errors. Do not leak whether a resource exists when revealing
that fact is itself a disclosure.

### 4. Define request and response shapes

- Give every field a type, nullability, format, and constraints.
- Validate at the boundary: required fields, lengths, ranges, enums, unknown
  fields, content type, and body size.
- Use one error envelope across the API:

```json
{
  "error": {
    "code": "invoice_not_found",
    "message": "No invoice with that id.",
    "details": [{"field": "id", "issue": "unknown"}],
    "request_id": "req_..."
  }
}
```

- Keep `code` stable and machine-readable; keep `message` safe for humans.
- Never expose stack traces, SQL, internal hostnames, or secrets.
- Return the resource representation after create/update unless there is a
  documented reason not to.

### 5. Pagination, filtering, and limits

Prefer cursor pagination for large or changing collections; offset pagination
is acceptable for small, stable sets. Define:

- default and maximum page size;
- the sort key and tie-breaker, so pages do not duplicate or skip items;
- the cursor's opacity and expiry;
- whether a total count is exact, estimated, or omitted;
- filter and sort field allowlists;
- behavior for invalid cursors and unknown filters.

Cap response size and query complexity. Document limits in the reference.

### 6. Idempotency and concurrency

Retries happen. For unsafe operations:

- accept an `Idempotency-Key` on create or payment-like calls;
- store the key with the result for a defined window;
- return the original result on replay, not a duplicate side effect;
- scope keys to the authenticated principal and operation;
- define what happens when the same key is used with a different body.

Use optimistic concurrency (`ETag`/`If-Match` or a version field) for updates
that can collide. Document the conflict response.

### 7. Authentication, authorization, and abuse

Separate authentication from authorization. Document which credential is
accepted, how it is sent, how it expires, and which scopes or roles are needed.
Enforce object-level authorization at the data access boundary, not only in the
route. Use 401 for missing/invalid credentials and 403 for insufficient
permission. Add rate limits per identity and per IP where appropriate, with a
clear 429 and `Retry-After`.

### 8. Versioning and change review

Prefer additive change. Adding an optional field or endpoint is usually safe;
renaming, removing, retyping, changing defaults, tightening validation, or
changing error codes is breaking. For a breaking change, provide a new version
or a documented migration window and sunset date.

Review every change against this list:

- [ ] contract updated (OpenAPI/schema) and validated;
- [ ] request and response examples are current;
- [ ] error cases and status codes covered by tests;
- [ ] auth and object-level authorization tested negatively;
- [ ] pagination boundaries and empty results tested;
- [ ] retries and idempotency behavior tested;
- [ ] old clients still work, or the break is versioned and announced;
- [ ] rate limits, payload caps, and timeouts documented.

## Completion checklist

- [ ] Consumers, trust levels, and compatibility window named.
- [ ] Resource model and method semantics consistent.
- [ ] Status codes match the actual failure category.
- [ ] One documented error envelope, with safe messages.
- [ ] Validation rules, limits, and unknown-field behavior stated.
- [ ] Pagination has stable ordering and bounded page size.
- [ ] Idempotency and concurrency behavior defined for writes.
- [ ] Authn, authz, scopes, and rate limits documented and tested.
- [ ] Breaking changes versioned with a migration path.

## Failure modes

- Designing endpoints around database tables instead of caller tasks.
- Using 200 for every response and putting failure in the body.
- Returning different error shapes from different handlers.
- Offset pagination on a collection that changes between pages.
- Accepting a client-supplied tenant or owner field.
- Calling a write idempotent because the handler catches duplicate errors.
- Adding a required field to an existing request without a migration plan.
- Documenting an example that was never sent to a running server.
