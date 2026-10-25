---
name: git
description: "Commit and branch safely: inspect diffs, resolve merge conflicts, keep changes small, and never discard user work."
aliases: git-workflow, commits
license: MIT
source: oh-my-hermes (MIT) — adapted
---

# Git

Git is the record of other people's work as much as yours. Inspect before
acting, stage precisely, commit one logical change at a time, and treat every
destructive command as someone else's lost work.

## Use this skill when

- The user asks to commit, branch, stage, merge, rebase, or clean up history.
- A merge or rebase conflict must be resolved.
- A commit broke something and the history needs inspection or a safe revert.
- You are about to record work and need the repository's commit convention.

## Do not use this skill when

- The user only asks what changed. Use read-only `git status` and `git diff`.
- The task is a code review; review the diff instead of committing it.
- The user asks to discard work. Stop and confirm explicitly before any
  operation that can lose uncommitted changes.

## First moves

Always inspect before writing to the repository:

```sh
git status --short
git branch --show-current
git log --oneline -10
git diff --stat
git diff
git diff --cached
git remote -v
```

Identify:

- the current branch and whether it is dedicated to this task;
- staged, unstaged, and untracked files;
- unrelated changes already present in the working tree;
- the repository's subject style and trailers from recent commits;
- whether a commit hook or signing requirement exists.

Never assume a dirty tree belongs to you. Leave unrelated files alone and say
so in the final report.

## Branching

- If the current branch is not dedicated to the task, create a focused branch
  before editing. Follow the repository's naming convention.
- Do not commit directly to a protected or shared branch when the project uses
  pull requests.
- Do not switch branches while the user has uncommitted work that would be
  carried over or conflict. Inspect first and ask if the intent is unclear.

## Staging

Stage by path, not by sweep:

```sh
git add path/to/changed_file.py tests/test_changed_file.py
git diff --cached
```

Avoid `git add -A` and `git add .` unless every change in the tree is part of
this commit. Never stage secrets, local configuration, generated caches, or
large binaries. If a file is already tracked and should not be, fix the
ignore rule separately rather than deleting the user's file.

Before committing, read the staged diff end to end. Confirm that the diff
contains only the intended change and no debug output, temporary paths,
credentials, or unrelated formatting.

## Commit boundaries

One commit should represent one logical change that can be understood and
reverted on its own. Split when the changes:

- mix a refactor with a behavior change;
- mix unrelated features or bug fixes;
- touch generated files without their source;
- contain a large formatting-only movement plus logic edits.

Do not split so finely that the tree never works between commits. Each commit
in a series should pass the checks that apply to it.

## Commit messages

Read the last ten subjects and the repository's contribution guide first.
Follow the local convention rather than importing another one.

A useful default:

```
Short imperative subject (<= 72 characters)

Why the change exists, what behavior changes, and any constraint that
shaped the approach. Mention the user-visible effect, not a restatement
of the diff.

Tested: exact command(s) that were actually run
Not-tested: known gaps, if the repository uses these trailers
```

Rules:

- Subject in the imperative mood: "Fix retry cap", not "Fixed" or "Fixes".
- Explain why, not just which files moved.
- List only commands that actually ran, with their observed result.
- Do not claim tests, review, CI, or merge status that was not observed.
- Do not add trailers, co-authors, or sign-offs the repository does not use.
- Never bypass hooks with `--no-verify` unless the user explicitly asks and
  understands the consequence.
- Never put tokens, private URLs, customer data, or raw transcripts in a
  commit message.

Commit only when the user asks. Creating a branch or editing files is not
permission to commit or push.

## History safety

Never run these against the user's work without an explicit, informed request:

- `git reset --hard`
- `git checkout -- <path>` or `git restore <path>`
- `git clean -f`, `git clean -fd`, or `git clean -fdx`
- `git stash` as a way to hide someone else's changes
- `git push --force` or `--force-with-lease`
- `git rebase` or `git commit --amend` on published history
- deleting branches or tags

If a rewrite is genuinely required, create a backup branch or note the reflog
entry first, state exactly what will be rewritten, and get confirmation. Force
pushing a shared branch affects everyone who fetched it.

To undo a published commit, prefer `git revert` over rewriting history.

## Conflicts

1. Run `git status` and list every conflicted file.
2. Read both sides and the common ancestor; understand what each intended.
3. Resolve by preserving both intents where possible, not by choosing a side
   wholesale.
4. Regenerate generated or lock files from their producer after the merge;
   never hand-pick conflict markers in them.
5. Remove every conflict marker and run the relevant tests.
6. Stage only the resolved files and complete the merge or rebase.

If the resolution becomes uncertain, abort the merge or rebase to return to a
known state. That is safer than guessing.

## Recovery

- A lost commit is usually in `git reflog`; inspect before doing anything else.
- A bad merge can be aborted before it is committed.
- A bad local commit can be corrected with a new commit; rewriting is a last
  resort.
- If the tree is in an unknown state, stop and report the observed status
  rather than experimenting with destructive commands.

## Completion checklist

- [ ] Status and diff inspected before staging.
- [ ] Unrelated or user-owned changes left untouched.
- [ ] Only intended paths staged, verified with `git diff --cached`.
- [ ] One logical change per commit.
- [ ] Subject and body follow the repository's convention.
- [ ] Tested/not-tested claims match observed runs.
- [ ] No secrets, debug output, or generated junk committed.
- [ ] No destructive command used without explicit approval.
- [ ] Push performed only if requested.

## Failure modes

- Sweeping the whole tree into a commit with `git add -A`.
- Discarding a dirty file to make a diff clean.
- Amending or force-pushing someone else's published commit.
- A giant commit named "updates" with no explanation.
- Bypassing a failing hook instead of fixing the failure.
- Resolving a conflict by accepting one side without reading the other.
- Committing generated files without their source.
- Reporting a commit or push that was prepared but never executed.
