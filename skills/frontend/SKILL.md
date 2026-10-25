---
name: frontend
description: "Build and polish interfaces with strong hierarchy, accessibility, and browser-verified results."
aliases: frontend-design, ui
license: MIT
source: oh-my-hermes (MIT) — adapted
---

# Frontend

A frontend is finished when it reads clearly, works with a keyboard, holds up
at every target width, and has been seen running. Code that compiles and looks
plausible in a diff is not visual evidence.

## Use this skill when

- Building or redesigning a page, component, dashboard, or terminal UI.
- Polishing layout, typography, color, motion, or responsive behavior.
- Auditing accessibility, interaction states, or visual consistency.
- The user says the interface looks generic, broken, cramped, or unfinished.

## Do not use this skill when

- The task is backend-only or a pure data change.
- The user asks for a brand identity, logo, or illustration. Those are separate
  design deliverables; this skill covers the interface.

## Workflow

### 1. Write the brief

Before touching code, state:

- the surface and route being changed;
- the primary user task and audience;
- the existing design system, tokens, and components;
- the target widths and input modes;
- the states that must exist: empty, loading, error, success, disabled;
- the quality bar and one visual reference, if the user supplied one.

Inspect the current implementation and tokens first. Do not invent a parallel
design system when one exists. If none exists, define a minimal one before
building many components.

### 2. Choose one direction

Pick a single primary direction and make it concrete:

- Operational: dense, fast, restrained; information leads.
- Editorial: generous space, strong type scale, reading-first.
- Premium: layered surfaces, considered motion, muted depth.
- Bold: oversized type, hard contrast, deliberate grid breaks.

Write the direction as rules, not adjectives. "Minimal" is not a rule;
"one accent, 8px spacing base, no shadows, 14px minimum body" is.

Name the defaults you are rejecting and why. Common unmade decisions:

- the framework's default blue as the accent with no brand reason;
- glass, blur, or cyan-to-purple gradients used as decoration;
- a uniform three-card row when the items are not peers;
- shadows on every surface, so elevation signals nothing;
- an eyebrow, heading, and description stacked above every section;
- placeholder copy and empty states that were never designed;
- Inter or the system stack doing no typographic work;
- a palette with one flood color, gray, and no semantic states.

### 3. Define tokens before components

Record concrete values in the project's design-token file or a short
`DESIGN.md`:

- color: every background layer, text level, border, accent, and semantic
  state (success, warning, danger, focus);
- typography: display and body families, fallbacks, sizes, weights, line
  heights, and a floor for CJK body text;
- spacing: a base unit and a small scale, used consistently;
- radius, border weight, and shadow levels;
- motion durations, easing, and reduced-motion behavior;
- breakpoints and layout constraints.

Every component consumes tokens. A hard-coded hex value or one-off spacing
value is a token request, not a local decision.

### 4. Build structure and states

Order the work:

1. Semantic structure and content hierarchy.
2. Layout at the smallest target width.
3. Responsive expansion to tablet and desktop.
4. Interactive states: hover, focus-visible, active, disabled, loading.
5. Empty, error, and success states.
6. Motion, only where it communicates state or causality.

Use native controls (`button`, `a`, `input`, `select`, `dialog`) before
rebuilding them. Keep dimensions stable between states so the layout does not
jump. Use realistic content, including long labels and translated strings.

### 5. Accessibility gate

- Every interactive element is reachable and operable by keyboard.
- Focus is visible and never removed without an equivalent replacement.
- Every input has a label; every image has appropriate alt text.
- Headings form a logical outline; landmarks and page regions are semantic.
- Text and meaningful icons meet contrast requirements (4.5:1 for body text).
- Color is not the only carrier of state or meaning.
- `prefers-reduced-motion` disables non-essential animation.
- Touch targets are large enough and spaced.
- Errors are announced and associated with the field that caused them.
- A screen reader can follow the reading order and understand dynamic updates.

Use `grep` to find missing labels, `div` click handlers, `outline: none`, and
motion without a reduced-motion branch. Use `task` for independent read-only
audits of component states, token drift, and accessibility patterns; verify
each finding in the source before reporting it.

### 6. Performance

State a budget before optimizing: LCP, INP, or CLS with a device and network
profile, and the route being measured. Attribute the metric to the element or
interaction that causes it. Avoid shipping large fonts, unoptimized images,
layout-shifting embeds, and client-side work that can be static.

Keep performance claims tied to the profile they were measured on. A lab pass
is not a field-data claim about real users.

### 7. Verify in the browser if possible

After the last UI edit:

1. Start the app with the repository's documented command.
2. Open the target route with a browser or the project's existing browser
   tests, if either is available in the environment.
3. Check at 375px, 768px, and 1440px, plus any project-specific breakpoint.
4. Exercise empty, loading, error, long-content, and keyboard-only paths.
5. Compare against the reference or the token contract.
6. Fix, then re-check. A pass from before the last edit is stale.

If no browser or rendering capability is available, say so explicitly. Do not
claim visual verification. Report what was inspected statically and what must
be checked by a human or CI with a browser.

## Completion checklist

- [ ] Brief and primary direction stated.
- [ ] Tokens defined and reused; no one-off colors or spacing.
- [ ] Content hierarchy and semantic structure are clear.
- [ ] Target widths checked with no horizontal overflow.
- [ ] Empty, loading, error, disabled, hover, focus, and active states exist.
- [ ] Keyboard and screen-reader basics verified.
- [ ] Contrast and reduced-motion rules applied.
- [ ] Performance budget named where relevant.
- [ ] Rendered result verified at target widths, or the gap is reported.
- [ ] No placeholder copy, lorem ipsum, or unfinished state left behind.

## Failure modes

- Declaring it beautiful without ever rendering it.
- A one-note palette with weak hierarchy and equal-weight cards.
- Rebuilding native controls and losing keyboard behavior.
- Removing focus outlines because they look untidy.
- Motion that ignores reduced-motion preferences.
- Text that overflows at 375px or clips CJK glyphs.
- Designing only the happy path and leaving error/empty states blank.
- Copying a component registry item without checking license, tokens, and
  behavior.
- Optimizing a metric without knowing which element produces it.
- Reporting "looks good" with no viewport, route, or evidence.
