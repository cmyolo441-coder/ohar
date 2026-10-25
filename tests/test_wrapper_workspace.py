from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ohar.types import ApprovalRequest, ToolContext
from ohar.wrapper import workspace as mod
from ohar.wrapper.records import (
    guarded_update,
    read_events,
    read_record,
    session_dir,
    write_record,
)
from ohar.wrapper.state import load_state, open_state

HAS_GIT = shutil.which("git") is not None


def _git(cwd: Path, *argv: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(cwd), *argv],
        capture_output=True,
        text=True,
        errors="replace",
        check=True,
        env={**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"},
    )
    return completed.stdout.strip()


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name).resolve()
        self.home = self.base / "home"
        self.ws = self.base / "ws"
        self.ws.mkdir(parents=True)
        env = mock.patch.dict(os.environ, {"OHAR_HOME": str(self.home)})
        env.start()
        self.addCleanup(env.stop)
        self.addCleanup(self.tmp.cleanup)
        self.ctx = ToolContext(workspace=self.ws, approval_mode="read-only")

    def repo(self, name: str = "repo", *, commit: bool = False) -> Path:
        root = self.ws / name
        root.mkdir(parents=True, exist_ok=True)
        _git(root, "init", "-q")
        (root / "README.md").write_text("hello\n", encoding="utf-8")
        if commit:
            _git(root, "add", "README.md")
            _git(root, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", "init")
        return root


@unittest.skipUnless(HAS_GIT, "git is not installed")
class ProbeTests(_Base):
    def test_probe_reports_a_committed_repository(self) -> None:
        root = self.repo(commit=True)
        info = mod.probe(root)
        self.assertTrue(info.is_repo)
        self.assertEqual(Path(info.root).resolve(), root.resolve())
        self.assertEqual(info.branch, _git(root, "symbolic-ref", "--quiet", "--short", "HEAD"))
        self.assertFalse(info.detached)
        self.assertRegex(info.revision, r"^[0-9a-f]{12}$")
        self.assertEqual(info.dirty, 0)
        self.assertEqual(info.worktrees, ())

    def test_probe_counts_uncommitted_changes(self) -> None:
        root = self.repo(commit=True)
        (root / "extra.txt").write_text("x\n", encoding="utf-8")
        (root / "README.md").write_text("changed\n", encoding="utf-8")
        self.assertEqual(mod.probe(root).dirty, 2)

    def test_probe_handles_an_unborn_branch_without_inventing_a_revision(self) -> None:
        root = self.repo()
        info = mod.probe(root)
        self.assertTrue(info.is_repo)
        self.assertEqual(info.revision, "")
        self.assertFalse(info.detached)
        self.assertEqual(info.dirty, 1)

    def test_probe_lists_known_worktrees_but_never_creates_one(self) -> None:
        root = self.repo(commit=True)
        second = self.base / "side"
        _git(root, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "worktree", "add", "-q", str(second), "-b", "side")
        info = mod.probe(root)
        self.assertEqual(len(info.worktrees), 1)
        self.assertEqual(Path(info.worktrees[0]).resolve(), second.resolve())
        linked = mod.probe(second).worktrees
        self.assertEqual([Path(path).name for path in linked], [root.name])

    def test_probe_of_a_plain_directory_is_not_a_repository(self) -> None:
        plain = self.ws / "plain"
        plain.mkdir()
        info = mod.probe(plain)
        self.assertFalse(info.is_repo)
        self.assertEqual(Path(info.root).resolve(), plain.resolve())
        self.assertEqual((info.branch, info.revision, info.dirty, info.worktrees), ("", "", 0, ()))

    def test_probe_of_a_file_uses_its_directory(self) -> None:
        root = self.repo(commit=True)
        info = mod.probe(root / "README.md")
        self.assertTrue(info.is_repo)

    def test_probe_fails_closed_for_missing_and_odd_inputs(self) -> None:
        missing = mod.probe(self.ws / "nope")
        self.assertFalse(missing.is_repo)
        self.assertEqual(missing.root, str(self.ws / "nope"))
        for value in ("", "   ", None, 3, ["a"]):
            info = mod.probe(value)  # type: ignore[arg-type]
            self.assertFalse(info.is_repo)
            self.assertEqual(info.revision, "")

    def test_probe_fails_closed_when_git_is_absent_or_hangs(self) -> None:
        root = self.repo(commit=True)
        with mock.patch.object(mod.subprocess, "run", side_effect=FileNotFoundError("no git")):
            info = mod.probe(root)
        self.assertFalse(info.is_repo)
        self.assertEqual((info.branch, info.revision, info.dirty, info.worktrees), ("", "", 0, ()))
        with mock.patch.object(
            mod.subprocess, "run", side_effect=subprocess.TimeoutExpired(cmd="git", timeout=1)
        ):
            self.assertFalse(mod.probe(root).is_repo)
        with mock.patch.object(
            mod.subprocess,
            "run",
            return_value=subprocess.CompletedProcess(args=["git"], returncode=128, stdout="", stderr=""),
        ):
            self.assertFalse(mod.probe(root).is_repo)

    def test_probe_never_uses_a_shell(self) -> None:
        root = self.repo(commit=True)
        with mock.patch.object(mod.subprocess, "run", wraps=mod.subprocess.run) as run:
            mod.probe(root)
        self.assertTrue(run.call_args_list)
        for call in run.call_args_list:
            self.assertIsInstance(call.args[0], list)
            self.assertNotIn("shell", call.kwargs)

    def test_sanitize_and_bounds_on_hostile_repository_output(self) -> None:
        root = self.repo(commit=True)
        hostile = "\x1b[31mbr\x00anch\nwith newline"

        def fake(argv, cwd, timeout):
            if argv[:2] == ["rev-parse", "--show-toplevel"]:
                return str(cwd) + "\n"
            if argv[0] == "symbolic-ref":
                return hostile + "\n"
            if argv[:2] == ["rev-parse", "--verify"]:
                return "a" * 40 + "\n"
            if argv[0] == "status":
                return "\n".join(["?. " + "x" for _ in range(mod.MAX_DIRTY + 50)]) + "\n"
            if argv[0] == "worktree":
                return "".join(f"worktree /p/{i}\nworktree refs: whatever\n" for i in range(mod.MAX_WORKTREES + 4))
            return None

        with mock.patch.object(mod, "_run_git", fake):
            info = mod.probe(root)
        self.assertNotIn("\x1b", info.branch)
        self.assertNotIn("\n", info.branch)
        self.assertEqual(info.revision, "a" * 12)
        self.assertEqual(info.dirty, mod.MAX_DIRTY)
        self.assertEqual(len(info.worktrees), mod.MAX_WORKTREES)


class RecipeTests(unittest.TestCase):
    def info(self, **changes: object) -> mod.WorkspaceInfo:
        values: dict[str, object] = {
            "root": "/tmp/proj",
            "is_repo": True,
            "branch": "main",
            "detached": False,
            "revision": "0123456789ab",
            "dirty": 2,
            "worktrees": ("/tmp/proj-wt",),
        }
        values.update(changes)
        return mod.WorkspaceInfo(**values)  # type: ignore[arg-type]

    def test_recipes_are_copy_only_and_bounded(self) -> None:
        recipes = mod.launch_recipes(self.info())
        self.assertLessEqual(len(recipes), mod.MAX_RECIPES)
        self.assertEqual([recipe.recipe_id for recipe in recipes][:2], ["open_workspace", "inspect_changes"])
        self.assertTrue(all(recipe.command for recipe in recipes))
        self.assertTrue(all(recipe.when_to_use for recipe in recipes))
        isolate = next(recipe for recipe in recipes if recipe.recipe_id == "isolate_worktree")
        self.assertIn("-b main-ohar", isolate.command)
        self.assertIn("OHAR never creates the worktree", isolate.when_to_use)
        confirm = next(recipe for recipe in recipes if recipe.recipe_id == "confirm_revision")
        self.assertIn("0123456789ab", confirm.command)

    def test_recipe_paths_are_shell_quoted(self) -> None:
        root = "/tmp/our project'; touch /tmp/pwned; 'x"
        recipes = mod.launch_recipes(self.info(root=root))
        quoted = shlex.quote(root)
        self.assertIn(quoted, recipes[0].command)
        for recipe in recipes:
            self.assertTrue(recipe.command.startswith("cd ") or f"-C {quoted}" in recipe.command)
            self.assertNotIn(root, recipe.command)
        detached = mod.launch_recipes(self.info(root=root, detached=True, branch=""))
        isolate = next(recipe for recipe in detached if recipe.recipe_id == "isolate_worktree")
        self.assertIn("--detach", isolate.command)
        self.assertIn(quoted, isolate.command)

    def test_untrusted_revision_is_not_passed_through(self) -> None:
        recipes = mod.launch_recipes(self.info(revision="; rm -rf /"))
        confirm = next(recipe for recipe in recipes if recipe.recipe_id == "confirm_revision")
        self.assertNotIn("rm -rf", confirm.command)

    def test_non_repository_yields_only_the_cd_recipe(self) -> None:
        recipes = mod.launch_recipes(self.info(is_repo=False, branch="", revision="", dirty=0, worktrees=()))
        self.assertEqual([recipe.recipe_id for recipe in recipes], ["open_workspace"])

    def test_to_dict_is_json_friendly(self) -> None:
        payload = self.info().to_dict()
        self.assertEqual(
            set(payload),
            {"root", "is_repo", "branch", "detached", "revision", "dirty", "worktrees"},
        )
        self.assertIsInstance(payload["worktrees"], list)
        json.dumps(payload)

    def test_workspace_line_shapes(self) -> None:
        line = mod.workspace_line(self.info())
        self.assertIn("main@0123456789ab", line)
        self.assertIn("2 dirty", line)
        self.assertIn("1 worktree", line)
        self.assertIn("·", line)
        self.assertNotIn("\n", mod.workspace_line(self.info(), glyphs="ascii"))
        self.assertNotIn("·", mod.workspace_line(self.info(), glyphs="ascii"))
        self.assertIn("not a repository", mod.workspace_line(self.info(is_repo=False)))
        self.assertIn("detached", mod.workspace_line(self.info(branch="", detached=True)))
        long_line = mod.workspace_line(self.info(root="/tmp/" + "x" * 400))
        self.assertLessEqual(len(long_line), mod.MAX_LINE_CHARS + 4)


@unittest.skipUnless(HAS_GIT, "git is not installed")
class BindTests(_Base):
    def test_bind_records_state_record_and_event(self) -> None:
        root = self.repo(commit=True)
        (root / "dirty.txt").write_text("d\n", encoding="utf-8")
        state = mod.bind("s1", root, self.ctx)
        self.assertEqual(Path(state.workspace).resolve(), root.resolve())
        again = load_state("s1")
        self.assertIsNotNone(again)
        self.assertEqual(Path(again.workspace).resolve(), root.resolve())

        record = read_record(mod.WORKSPACE_RECORD, "s1")
        self.assertEqual(record["record_type"], mod.WORKSPACE_RECORD_TYPE)
        self.assertEqual(record["binding_schema_version"], mod.WORKSPACE_SCHEMA_VERSION)
        self.assertTrue(record["workspace"]["revision"])
        self.assertEqual(record["workspace"]["dirty"], 1)

        events, errors = read_events("s1")
        self.assertEqual(errors, [])
        names = [event["event"] for event in events]
        self.assertIn(mod.BOUND_EVENT, names)
        bound = next(event for event in events if event["event"] == mod.BOUND_EVENT)
        self.assertEqual(bound["data"]["dirty"], 1)

    def test_bind_keeps_an_existing_goal_and_rebinds_the_path(self) -> None:
        first = self.repo("one", commit=True)
        second = self.repo("two", commit=True)
        open_state("s2", goal="Ship the parser", workspace="")
        state = mod.bind("s2", first, self.ctx)
        self.assertEqual(state.goal, "Ship the parser")
        self.assertEqual(Path(state.workspace).resolve(), first.resolve())
        rebound = mod.bind("s2", second, self.ctx)
        self.assertEqual(Path(rebound.workspace).resolve(), second.resolve())
        self.assertEqual(read_record(mod.WORKSPACE_RECORD, "s2")["path"], str(second))

    def test_bind_refuses_paths_outside_the_workspace(self) -> None:
        outside = self.base / "outside"
        outside.mkdir()
        with self.assertRaises(mod.WorkspaceError):
            mod.bind("s3", outside, self.ctx)
        with self.assertRaises(mod.WorkspaceError):
            mod.bind("s3", "../outside", self.ctx)
        with self.assertRaises(mod.WorkspaceError):
            mod.bind("s3", "", self.ctx)
        with self.assertRaises(mod.WorkspaceError):
            mod.bind("s3", None, self.ctx)  # type: ignore[arg-type]
        with self.assertRaises(mod.WorkspaceError):
            mod.bind("s3", self.ws / "missing", self.ctx)
        a_file = self.ws / "file.txt"
        a_file.write_text("x", encoding="utf-8")
        with self.assertRaises(mod.WorkspaceError):
            mod.bind("s3", a_file, self.ctx)
        with self.assertRaises(ValueError):
            mod.bind("bad id", self.ws, self.ctx)

    def test_bind_requires_a_tool_context(self) -> None:
        root = self.repo(commit=True)
        with self.assertRaises(mod.WorkspaceError):
            mod.bind("s4", root, None)  # type: ignore[arg-type]

    def test_bind_never_mutates_the_repository(self) -> None:
        root = self.repo(commit=True)
        before = _git(root, "rev-parse", "HEAD")
        mod.bind("s5", root, self.ctx)
        self.assertEqual(_git(root, "rev-parse", "HEAD"), before)
        self.assertEqual(len(_git(root, "worktree", "list").splitlines()), 1)
        self.assertEqual(sorted(path.name for path in root.iterdir()), [".git", "README.md"])


@unittest.skipUnless(HAS_GIT, "git is not installed")
class BindingTests(_Base):
    def test_unknown_session_reports_unknown(self) -> None:
        payload = mod.binding("nobody")
        self.assertEqual(payload["status"], "unknown")
        self.assertEqual(payload["path"], "")
        self.assertEqual(payload["next_action"], "bind_workspace")
        self.assertEqual(payload["launch_recipes"], [])
        self.assertIn("worktree creation", payload["claim_boundary"])
        self.assertIn("revision_observed", payload["not_evidence_until_observed"])

    def test_bound_session_is_an_observed_record(self) -> None:
        root = self.repo(commit=True)
        (root / "note.txt").write_text("n\n", encoding="utf-8")
        mod.bind("s1", root, self.ctx)
        payload = mod.binding("s1")
        self.assertEqual(payload["status"], "ready_observed")
        self.assertEqual(payload["path"], str(root))
        self.assertEqual(payload["dirty"], 1)
        self.assertRegex(payload["revision"], r"^[0-9a-f]{12}$")
        self.assertEqual(payload["next_action"], "isolate_work_in_worktree")
        self.assertNotIn("not_evidence_until_observed", payload)
        self.assertTrue(payload["bound_at"])
        self.assertEqual(
            set(payload["launch_recipes"][0]),
            {"recipe_id", "label", "command", "when_to_use"},
        )
        self.assertIn(str(root), payload["launch_recipes"][0]["command"])

    def test_detached_observed_session_asks_for_confirmation(self) -> None:
        root = self.repo(commit=True)
        self._write_record("s-d", root, detached=True, dirty=0, revision="0123456789ab")
        payload = mod.binding("s-d")
        self.assertEqual(payload["status"], "ready_observed")
        self.assertEqual(payload["next_action"], "confirm_detached_revision")
        isolate = next(r for r in payload["launch_recipes"] if r["recipe_id"] == "isolate_worktree")
        self.assertIn("--detach", isolate["command"])

    def _write_record(self, session_id: str, root: Path, **changes: object) -> None:
        state = load_state(session_id)
        if state is None:
            open_state(session_id, goal="Work", workspace=str(root))
        record = {
            "record_type": mod.WORKSPACE_RECORD_TYPE,
            "binding_schema_version": mod.WORKSPACE_SCHEMA_VERSION,
            "session_id": session_id,
            "path": str(root),
            "bound_at": "2026-10-25T00:00:00+00:00",
            "workspace": {
                "root": str(root),
                "is_repo": True,
                "branch": "main",
                "detached": False,
                "revision": "0123456789ab",
                "dirty": 0,
                "worktrees": [],
            },
        }
        record["workspace"].update(changes)
        guarded_update(
            mod.WORKSPACE_RECORD, session_id, lambda _current: record, operation="test_bind"
        )

    def test_corrupt_record_degrades_to_filesystem_only(self) -> None:
        root = self.repo(commit=True)
        mod.bind("s2", root, self.ctx)
        (session_dir("s2") / (mod.WORKSPACE_RECORD)).write_text("{ not json", encoding="utf-8")
        payload = mod.binding("s2")
        self.assertEqual(payload["status"], "filesystem_only")
        self.assertEqual(payload["next_action"], "observe_workspace_revision")
        self.assertIn("revision_observed", payload["not_evidence_until_observed"])
        self.assertEqual(payload["revision"], mod.probe(root).revision)

    def test_a_foreign_record_is_not_read_as_a_binding(self) -> None:
        root = self.repo(commit=True)
        open_state("s3", goal="Work", workspace=str(root))
        write_record(mod.WORKSPACE_RECORD, "s3", {"workspace": {"root": str(root), "is_repo": True}})
        payload = mod.binding("s3")
        self.assertEqual(payload["status"], "filesystem_only")

    def test_a_deleted_workspace_reads_as_missing(self) -> None:
        root = self.repo(commit=True)
        self._write_record("s4", root)
        shutil.rmtree(root)
        payload = mod.binding("s4")
        self.assertEqual(payload["status"], "missing")
        self.assertEqual(payload["next_action"], "rebind_workspace")
        self.assertEqual(payload["revision"], "0123456789ab")

    def test_recorded_path_is_confined_when_a_context_is_given(self) -> None:
        escape = self.base / "elsewhere"
        escape.mkdir()
        self._write_record("s5", escape)
        payload = mod.binding("s5", self.ctx)
        self.assertEqual(payload["status"], "unknown")
        self.assertEqual(payload["path"], "")

    def test_dirty_count_from_a_record_is_clamped(self) -> None:
        root = self.repo(commit=True)
        self._write_record("s6", root, dirty=10_000, branch="\x1b[31mmain\x1b[0m")
        payload = mod.binding("s6")
        self.assertEqual(payload["dirty"], mod.MAX_DIRTY)
        self.assertNotIn("\x1b", payload["branch"])
        self.assertNotIn("\x1b", payload["line"])

    def test_relative_recorded_path_resolves_through_the_context(self) -> None:
        root = self.repo("rel", commit=True)
        open_state("s7", goal="Work", workspace="")
        guarded_update(
            mod.WORKSPACE_RECORD,
            "s7",
            lambda _current: {
                "record_type": mod.WORKSPACE_RECORD_TYPE,
                "path": "rel",
                "workspace": {"root": str(root), "is_repo": True, "branch": "main", "detached": False, "revision": "0" * 12, "dirty": 0, "worktrees": []},
                "bound_at": "2026-10-25T00:00:00+00:00",
            },
            operation="test_bind",
        )
        payload = mod.binding("s7", self.ctx)
        self.assertEqual(payload["status"], "ready_observed")
        self.assertEqual(payload["path"], str(root))


@unittest.skipUnless(HAS_GIT, "git is not installed")
class ToolTests(_Base):
    def test_workspace_status_tool_is_read_only(self) -> None:
        tool = mod.make_workspace_tool(self.ctx, "t1")
        self.assertEqual(tool.name, "workspace_status")
        self.assertTrue(tool.read_only)
        self.assertFalse(tool.destructive)
        self.assertEqual(tool.parameters["additionalProperties"], False)
        spec = tool.api_spec()["function"]
        self.assertIn("workspace bound to this session", spec["description"])

    def test_tool_reports_the_binding_without_running_anything(self) -> None:
        root = self.repo(commit=True)
        (root / "w.txt").write_text("w\n", encoding="utf-8")
        mod.bind("t2", root, self.ctx)
        result = mod.make_workspace_tool(self.ctx, "t2").handler({}, self.ctx)
        self.assertTrue(result.ok)
        self.assertEqual(result.meta["status"], "ready_observed")
        self.assertEqual(result.meta["next_action"], "isolate_work_in_worktree")
        self.assertGreaterEqual(result.meta["recipes"], 2)
        self.assertIn("copy-only recipes", result.content)
        self.assertIn("worktree add", result.content)
        self.assertIn("next action:", result.content)
        self.assertIn("read-only projection", result.content)
        self.assertEqual(sorted(path.name for path in root.iterdir()), [".git", "README.md", "w.txt"])

    def test_tool_rejects_arguments_and_surfaces_failures(self) -> None:
        tool = mod.make_workspace_tool(self.ctx, "t3")
        refused = tool.handler({"path": "/etc"}, self.ctx)
        self.assertFalse(refused.ok)
        self.assertIn("nothing was run", refused.content)
        with mock.patch.object(mod, "binding", side_effect=mod.WorkspaceError("boom")):
            failed = tool.handler({}, self.ctx)
        self.assertFalse(failed.ok)
        self.assertIn("boom", failed.content)
        self.assertEqual(failed.meta, {})

    def test_tool_text_on_an_unbound_session(self) -> None:
        result = mod.make_workspace_tool(self.ctx, "t4").handler({}, self.ctx)
        self.assertTrue(result.ok)
        self.assertIn("not a repository", result.content)
        self.assertIn("bind_workspace", result.content)

    def test_tool_never_asks_for_approval(self) -> None:
        asked: list[ApprovalRequest] = []
        ctx = ToolContext(
            workspace=self.ws,
            approval_mode="ask",
            confirm=lambda request: asked.append(request) or False,
        )
        root = self.repo(commit=True)
        mod.bind("t5", root, ctx)
        result = mod.make_workspace_tool(ctx, "t5").handler({}, ctx)
        self.assertTrue(result.ok)
        self.assertEqual(asked, [])


if __name__ == "__main__":
    unittest.main()
