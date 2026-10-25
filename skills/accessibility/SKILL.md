---
name: accessibility
description: "Review interfaces against WCAG: semantics, keyboard, focus, contrast, motion, and screen readers."
aliases: a11y, wcag, accessibility-review
license: MIT
source: oh-my-hermes (MIT) — adapted
---

# Accessibility

Accessibility is whether people can actually use the interface, not whether a
checker returned zero violations. Review the rendered result, walk it with a
keyboard, inspect the accessibility tree, and cite the WCAG criterion behind each
finding.

## Use this skill when

- Reviewing a page, flow, component, or design system for accessibility.
- A user reports keyboard, focus, screen-reader, contrast, or zoom problems.
- Adding forms, dialogs, menus, carousels, tables, or custom controls.
- Preparing evidence for a WCAG 2.2 AA review or an accessibility fix.

## Do not use this skill when

- The task is purely visual polish with no interaction or semantic impact.
  Combine with a visual review if layout also matters.
- You have no rendered page, snapshot, or component evidence and cannot obtain
  it. Prepare the checklist and mark the verdict as blocked, not passed.
- The user wants a legal compliance certification. Report observed evidence and
  scope; certification belongs to a qualified reviewer.

## Workflow

### 1. Scope the review

Name the pages or components, the critical user tasks, the target viewport
matrix, and the standard (usually WCAG 2.2 AA). List the evidence available:
source markup, `browser_snapshot` accessibility tree, `browser_screenshot`,
keyboard walk, or a manual screen-reader session. A pass requires evidence from
the rendered surface, not only source code.

### 2. Inspect semantics first

Use `browser_snapshot` when available to read roles, names, states, and heading
structure. Otherwise inspect the rendered HTML or component tree.

- One `h1`, a logical heading outline, and landmark regions
  (`header`, `nav`, `main`, `footer`).
- Buttons are `<button>` for actions; links are `<a href>` for navigation.
- Lists use list markup; tables use caption, header cells, and scope.
- Images have useful `alt` text, or empty `alt` when decorative.
- Form controls have a programmatic label, not just a placeholder.
- Custom widgets have the correct role, name, value, and state.
- ARIA is used to fill a real semantic gap, never to patch a native element.

### 3. Walk the keyboard path

Put the pointer aside. Starting from the URL:

- Tab reaches every interactive control in a logical order.
- The focus indicator is visible and high-contrast on every background.
- There are no keyboard traps. Escape closes menus and dialogs.
- A skip link reaches the main content.
- Dialogs move focus inside, keep it there, and restore it on close.
- Menus, tabs, and grids support the expected arrow-key behavior.
- Disabled or hidden controls are not focusable.
- Custom drag, hover, or pointer-only actions have a keyboard equivalent.

Record the sequence and the exact step where it breaks.

### 4. Check names, labels, and errors

- Every control has an accessible name matching its visible label.
- Required fields are indicated in text, not color alone.
- Help text and errors are programmatically associated with the field.
- Error summaries identify the field and the correction.
- Focus moves to the error summary or first invalid field after submit.
- Status messages use a live region with an appropriate politeness setting.
- Icon-only buttons have an accessible name.
- Link text makes sense out of context; avoid repeated "click here".

### 5. Measure contrast and visual state

Check text at 4.5:1 for normal text and 3:1 for large text, plus 3:1 for UI
component boundaries, icons, and focus indicators. Verify with a contrast tool
rather than eyeballing. Test hover, focus, active, disabled, and error states.
Do not use color as the only way to convey status. Check that content remains
readable at 200% text size and 400% zoom, and that pages reflow to 320 CSS pixels
without horizontal scrolling, except for content that genuinely needs two
dimensions such as a data table.

### 6. Check motion and timing

- Respect `prefers-reduced-motion`; disable or shorten non-essential animation.
- No content flashes more than three times per second.
- Auto-playing media can be paused, stopped, or muted.
- Time limits can be extended or turned off where possible.
- Carousels do not advance while the user is interacting, and controls are
  keyboard reachable.
- Parallax and scroll effects do not block reading or trigger motion sickness.

### 7. Verify with the accessibility tree and a screenshot

`browser_snapshot` is the primary evidence for roles, names, and focus order.
`browser_screenshot` records visual issues such as clipped text, invisible focus,
or low contrast; it does not prove the accessibility tree. When a screen reader
is available, listen to the name, role, value, and announcement for each custom
control. Automated checks are useful for finding candidates, but they cannot
prove keyboard or screen-reader usability.

### 8. Report findings with impact

Use one block per finding:

```
[Serious] Dialog does not return focus after close
WCAG: 2.4.3 Focus Order
Location: /settings, "Delete workspace" dialog
Evidence: browser_snapshot shows focus on body after Escape; keyboard walk loses position.
Impact: keyboard and screen-reader users restart navigation from the top.
Fix: store the trigger element and restore focus on close.
```

Group by severity (critical, serious, moderate, minor), cite the criterion, and
distinguish automated detection from manual observation. After fixes, retest the
same path and report what is resolved, carried, or new.

## Completion checklist

- [ ] Pages, tasks, viewports, standard, and evidence named.
- [ ] Heading, landmark, list, table, and control semantics checked.
- [ ] Full keyboard path walked with visible focus and no traps.
- [ ] Accessible names, labels, errors, and status messages verified.
- [ ] Contrast measured for text, controls, and focus states.
- [ ] Reduced motion, timing, zoom, reflow, and target size checked.
- [ ] Accessibility tree inspected; screenshot used only for visual evidence.
- [ ] Findings cite WCAG criterion, location, impact, and fix.
- [ ] No pass claimed without rendered evidence.

## Failure modes

- Treating an automated scan's zero violations as WCAG compliance.
- Reviewing source code only and never rendering the page.
- Adding `aria-label` while leaving a non-semantic `div` as the control.
- A visible label that is not programmatically associated with the input.
- Focus styles removed because the outline looked ugly.
- Testing with a mouse and calling the flow keyboard accessible.
- Reporting contrast from a screenshot without measuring.
- Claiming screen-reader support without listening to the announcement.
