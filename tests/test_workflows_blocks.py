from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from ohar.types import ToolContext, ToolResult
from ohar.workflows.blocks import (
    DECISION_OUTCOMES,
    ENFORCEMENT,
    MAX_CLASS_SHAPE,
    MAX_DIRTY_PATHS,
    REASON_DOMAINS,
    RECOVERY_ACTIONS,
    WORK_CLAIMS,
    BlockedWork,
    BlockLog,
    RecoveryAnchor,
    RecoveryAnchors,
    enforcement_for,
    make_block_tool,
    recovery_action_for,
    request_class_shape,
    request_fingerprint,
    subject_class_digest,
    workspace_digest_of,
)
from ohar.workflows.journal import ActivityJournal
from ohar.workflows.store import WorkflowError

SECRET_PATH = "src/billing/token_store.py"
SECRET_PROMPT = "rotate the signing key and paste the private key here"


class TempBase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.root = self.tmp / "home"
        self.workspace = self.tmp / "repo"
        self.workspace.mkdir()
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))


class VocabularyTests(unittest.TestCase):
    def test_closed_vocabularies(self) -> None:
        self.assertEqual(len(set(REASON_DOMAINS)), len(REASON_DOMAINS))
        self.assertEqual(set(DECISION_OUTCOMES), {"allowed", "blocked", "cancelled", "completed", "failed"})
        self.assertEqual(set(WORK_CLAIMS), {"not_attempted", "attempted_not_completed", "completed"})
        self.assertEqual(set(ENFORCEMENT), {"enforced", "declared_not_enforced"})
        self.assertEqual(
            set(RECOVERY_ACTIONS),
            {"answer_gate", "allow_action", "leave_read_only", "retry_with_scope", "run_recovery_recipe",
             "no_recovery_available"},
        )
        self.assertEqual((MAX_CLASS_SHAPE, MAX_DIRTY_PATHS), (32, 20))

    def test_recovery_action_is_a_closed_id_for_every_pair(self) -> None:
        self.assertEqual(recovery_action_for("approval_denied", "approval_denied"), "allow_action")
        self.assertEqual(recovery_action_for("approval_denied", "approval_expired"), "answer_gate")
        self.assertEqual(recovery_action_for("read_only_policy", "read_only_mode"), "leave_read_only")
        self.assertEqual(recovery_action_for("workspace_escape", "path_outside_workspace"), "retry_with_scope")
        self.assertEqual(recovery_action_for("gate_open", "gate_pending"), "answer_gate")
        self.assertEqual(recovery_action_for("verification_failed", "check_failed"), "run_recovery_recipe")
        self.assertEqual(recovery_action_for("cancelled", "operator_cancel"), "no_recovery_available")
        self.assertEqual(recovery_action_for("budget_exhausted", "token_count_exceeded"), "retry_with_scope")
        # A code nobody has mapped still gets the domain's action; an unknown
        # pair gets the one id that promises nothing.
        self.assertEqual(recovery_action_for("provider_error", "brand_new_code"), "retry_with_scope")
        self.assertEqual(recovery_action_for("made_up_domain", "made_up_code"), "no_recovery_available")

    def test_enforcement_separates_control_points_from_reports(self) -> None:
        self.assertEqual(enforcement_for("tool_gate"), "enforced")
        self.assertEqual(enforcement_for("policy"), "enforced")
        self.assertEqual(enforcement_for("model"), "declared_not_enforced")
        self.assertEqual(enforcement_for("operator_report"), "declared_not_enforced")
        # An unknown source is never promoted to an enforcer.
        self.assertEqual(enforcement_for("whatever_the_file_says"), "declared_not_enforced")


class ClassShapeTests(unittest.TestCase):
    def test_shape_names_classes_never_values(self) -> None:
        shape = request_class_shape(
            {"path": SECRET_PATH, "command": "pytest -q", "content": SECRET_PROMPT, "limit": 4, "force": True}
        )
        self.assertIn("field:path:set", shape)
        self.assertIn("field:command:set", shape)
        self.assertIn("field:content:set", shape)
        self.assertIn("field:scalar:set", shape)
        self.assertIn("field:vocabulary:true", shape)
        dump = json.dumps(shape)
        self.assertNotIn(SECRET_PATH, dump)
        self.assertNotIn("pytest", dump)
        self.assertNotIn("rotate", dump)

    def test_shape_is_sorted_and_order_independent(self) -> None:
        first = request_class_shape({"path": "a.py", "content": "one"})
        second = request_class_shape({"content": "two", "path": "b.py"})
        self.assertEqual(first, second)

    def test_closed_vocabulary_values_may_be_carried_verbatim(self) -> None:
        shape = request_class_shape({"mode": "read-only", "risk": "high"})
        self.assertIn("field:vocabulary:read-only", shape)
        self.assertIn("field:vocabulary:high", shape)

    def test_secret_shaped_keys_never_leak_a_shape_of_their_own(self) -> None:
        shape = request_class_shape({"api_token": "hunter2", "path": SECRET_PATH})
        self.assertIn("field:secret:set", shape)
        self.assertNotIn("hunter2", json.dumps(shape))

    def test_shape_is_bounded_and_keeps_every_distinct_label(self) -> None:
        wide = {f"field_{index:03d}": "v" for index in range(60)}
        wide["path"] = SECRET_PATH
        wide["command"] = "ls"
        shape = request_class_shape(wide)
        self.assertLessEqual(len(shape), MAX_CLASS_SHAPE)
        self.assertIn("field:path:set", shape)
        self.assertIn("field:command:set", shape)

    def test_non_mapping_and_empty_containers(self) -> None:
        self.assertEqual(request_class_shape(None), [])
        self.assertEqual(request_class_shape("not a mapping"), [])  # type: ignore[arg-type]
        self.assertEqual(request_class_shape({}), [])
        self.assertIn("field:empty:empty", request_class_shape({"files": [], "meta": {}}))

    def test_fingerprint_groups_repeat_blocks_and_hides_the_request(self) -> None:
        one = request_class_shape({"path": "src/a.py", "content": "first"})
        two = request_class_shape({"content": "second", "path": "src/b.py"})
        self.assertEqual(request_fingerprint("write_file", one), request_fingerprint("write_file", two))
        self.assertNotEqual(request_fingerprint("write_file", one), request_fingerprint("read_file", one))
        self.assertTrue(request_fingerprint("write_file", one).startswith("fp-"))
        self.assertNotIn("src/a.py", request_fingerprint("write_file", one))

    def test_fingerprint_refuses_to_hash_unknown_labels(self) -> None:
        smuggled = ["field:path:set", "content:excluded", SECRET_PATH, SECRET_PROMPT, "field:zzz:set"]
        clean = ["field:path:set"]
        self.assertEqual(request_fingerprint("write_file", smuggled), request_fingerprint("write_file", clean))

    def test_subject_handle_cannot_name_the_subject(self) -> None:
        shape = request_class_shape({"path": "x"})
        self.assertEqual(subject_class_digest("src/one.py", shape), subject_class_digest("src/two.py", shape))
        self.assertNotEqual(subject_class_digest("src/one.py", shape), subject_class_digest(7, shape))


class BlockLogTests(TempBase):
    def setUp(self) -> None:
        super().setUp()
        self.log = BlockLog(self.root, self.workspace)

    def _record(self, **kwargs: object) -> BlockedWork:
        base: dict[str, object] = {
            "source": "tool_gate",
            "reason_domain": "approval_denied",
            "reason_code": "approval_denied",
            "action": "write_file",
        }
        base.update(kwargs)
        return self.log.record(**base)  # type: ignore[arg-type]

    def test_record_mints_derived_claims_and_keeps_no_bytes(self) -> None:
        record = self._record(
            request={"path": SECRET_PATH, "content": SECRET_PROMPT},
            subject=SECRET_PATH,
            session_id="s1",
            goal_id="g1",
        )
        self.assertEqual(record.outcome, "blocked")
        self.assertEqual(record.work_claim, "not_attempted")
        self.assertEqual(record.enforcement, "enforced")
        self.assertEqual(record.recovery_action, "allow_action")
        self.assertEqual(record.session_id, "s1")
        self.assertTrue(record.decision_id.startswith("dec-"))
        dumped = json.dumps(record.to_dict())
        self.assertNotIn(SECRET_PATH, dumped)
        self.assertNotIn(SECRET_PROMPT, dumped)
        raw = self.log.path.read_text(encoding="utf-8")
        self.assertNotIn(SECRET_PATH, raw)
        self.assertNotIn(SECRET_PROMPT, raw)

    def test_unknown_reason_code_folds_to_unspecified(self) -> None:
        record = self._record(reason_domain="tool_failure", reason_code="brand_new_code")
        self.assertEqual(record.reason_code, "unspecified")
        self.assertEqual(record.recovery_action, "retry_with_scope")

    def test_unsupported_vocabulary_is_refused(self) -> None:
        with self.assertRaises(WorkflowError):
            self._record(reason_domain="mood_ring")
        with self.assertRaises(WorkflowError):
            self._record(outcome="mostly_blocked")

    def test_outcome_drives_the_work_claim(self) -> None:
        self.assertEqual(self._record(outcome="allowed").work_claim, "not_attempted")
        self.assertEqual(self._record(outcome="cancelled").work_claim, "attempted_not_completed")
        self.assertEqual(self._record(outcome="completed").work_claim, "completed")

    def test_declaring_source_stays_declared(self) -> None:
        reported = self._record(source="model")
        self.assertEqual(reported.enforcement, "declared_not_enforced")

    def test_recent_history_and_counts(self) -> None:
        self._record(session_id="s1")
        self._record(reason_domain="read_only_policy", reason_code="read_only_mode", action="edit_file",
                    session_id="s2")
        newest = self.log.recent()
        self.assertEqual([item.action for item in newest], ["edit_file", "write_file"])
        self.assertEqual([item.action for item in self.log.recent(session_id="s1")], ["write_file"])
        self.assertEqual([item.reason_domain for item in self.log.recent(reason_domain="gate_open")], [])
        self.assertEqual(len(self.log.recent(limit=1)), 1)
        self.assertEqual([item.at for item in self.log.history("edit_file")], [newest[0].at])
        self.assertEqual(self.log.history("nope"), [])
        counts = self.log.counts()
        self.assertEqual(counts["approval_denied"], 1)
        self.assertEqual(counts["read_only_policy"], 1)
        self.assertEqual(counts["total"], 2)

    def test_for_decision_chains_repeat_refusals_of_one_question(self) -> None:
        first = self._record(request={"path": "a.py"}, subject="a.py")
        second = self._record(request={"path": "b.py"}, subject="b.py")
        other = self._record(reason_code="approval_expired")
        self.assertEqual(first.decision_id, second.decision_id)
        self.assertNotEqual(first.decision_id, other.decision_id)
        self.assertEqual(len(self.log.for_decision(first.decision_id)), 2)
        self.assertEqual(self.log.for_decision(""), [])

    def test_rows_and_headers(self) -> None:
        record = self._record()
        rows = self.log.rows([record])
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]), len(self.log.headers()))
        self.assertEqual(self.log.headers()[2], "domain")
        self.assertIn("allow_action", rows[0])

    def test_store_survives_a_new_instance(self) -> None:
        self._record()
        again = BlockLog(self.root, self.workspace)
        self.assertEqual(len(again.recent()), 1)

    def test_other_workspace_cannot_read_these_records(self) -> None:
        self._record()
        other = self.tmp / "other"
        other.mkdir()
        self.assertEqual(BlockLog(self.root, other).recent(), [])

    def test_journal_supplies_the_gate_it_observed_for_this_action(self) -> None:
        journal = ActivityJournal(self.root, self.workspace)
        journal.record("gate_opened", subject="gate-9", detail={"action": "write_file", "record": "gt-9"})
        journal.record("gate_opened", subject="gate-1", detail={"action": "shell", "record": "gt-1"})
        log = BlockLog(self.root, self.workspace, journal=journal)
        self.assertEqual(self._record_with(log).gate_id, "gt-9")

    def _record_with(self, log: BlockLog) -> BlockedWork:
        return log.record(
            source="gate", reason_domain="gate_open", reason_code="gate_open", action="write_file"
        )

    def test_explicit_gate_id_wins_and_a_broken_journal_never_breaks_the_record(self) -> None:
        record = self._record(gate_id="gate-7")
        self.assertEqual(record.gate_id, "gate-7")
        quiet = BlockLog(self.root, self.workspace, journal=None)
        self.assertEqual(quiet.record(source="gate", reason_domain="gate_open", reason_code="gate_open",
                                      action="shell").gate_id, "")

    def test_from_tool_result_records_refusals_only(self) -> None:
        self.assertIsNone(self.log.from_tool_result("read_file", {"path": "a"}, ToolResult.success("fine")))
        self.assertIsNone(self.log.from_tool_result("read_file", {"path": "a"}, ToolResult.failure("bad json")))
        denied = self.log.from_tool_result("write_file", {"path": SECRET_PATH},
                                           ToolResult.failure("Action denied.", denied=True))
        read_only = self.log.from_tool_result("shell", {"command": "ls"},
                                              ToolResult.failure("Read-only mode: shell is disabled.", denied=True))
        escape = self.log.from_tool_result("read_file", {"path": "../../etc/shadow"},
                                           ToolResult.failure("Path '../../etc/shadow' is outside the workspace."))
        gate = self.log.from_tool_result("edit_file", {"path": "a"}, ToolResult.failure("a decision gate is open"))
        self.assertEqual([item.reason_domain for item in (denied, read_only, escape, gate)],
                         ["approval_denied", "read_only_policy", "workspace_escape", "gate_open"])
        self.assertTrue(all(item.enforcement == "enforced" for item in (denied, read_only, escape, gate)))
        self.assertEqual(len(self.log.recent()), 4)

    def test_control_characters_cannot_hide_a_refusal(self) -> None:
        record = self.log.from_tool_result(
            "write_file", {"path": "a"}, ToolResult.failure("\x1b[31mdenied\x1b[0m by the gate", denied=True)
        )
        self.assertIsNotNone(record)
        self.assertEqual(record.reason_domain, "approval_denied")
        self.assertNotIn("\x1b", json.dumps(record.to_dict()))


class StoreHygieneTests(TempBase):
    def test_record_file_is_private_and_line_bounded(self) -> None:
        log = BlockLog(self.root, self.workspace)
        log.record(source="tool_gate", reason_domain="approval_denied", reason_code="approval_denied",
                   action="write_file", request={"path": SECRET_PATH}, subject=SECRET_PATH)
        path = log.path
        self.assertEqual(oct(path.stat().st_mode & 0o777), "0o600")
        self.assertTrue(path.read_text(encoding="utf-8").strip().startswith("{"))
        self.assertLess(len(path.read_bytes().splitlines()[-1]), 65_536)

    def test_a_planted_enforcement_claim_is_repaired_through_the_store(self) -> None:
        log = BlockLog(self.root, self.workspace)
        record = log.record(source="model", reason_domain="gate_open", reason_code="gate_open", action="shell")
        lines = [json.loads(line) for line in log.path.read_text(encoding="utf-8").splitlines()]
        # Written straight to the store: a hand-edited record must not be able to
        # promote its own enforcement, work claim, or recovery.
        forged = dict(lines[-1])
        forged.update({"id": "block-forged", "source": "model", "enforcement": "enforced",
                       "work_claim": "completed", "recovery_action": "run_recovery_recipe"})
        log._store.append(forged)
        restored = {item.id: item for item in log.recent(limit=10)}
        self.assertEqual(restored["block-forged"].enforcement, "declared_not_enforced")
        self.assertEqual(restored["block-forged"].work_claim, "not_attempted")
        self.assertEqual(restored["block-forged"].recovery_action, "answer_gate")
        self.assertEqual(restored[record.id].enforcement, "declared_not_enforced")

    def test_underscores_and_path_shapes_never_become_actions(self) -> None:
        log = BlockLog(self.root, self.workspace)
        record = log.record(source="tool_gate", reason_domain="workspace_escape", reason_code="symlink_escape",
                           action="/etc/passwd; rm -rf /", subject="/etc/passwd")
        self.assertNotIn("/etc/passwd", json.dumps(record.to_dict()))
        self.assertTrue(record.action.startswith("ref:"))


class BlockedWorkReadTests(TempBase):
    def test_reads_rederive_claims_a_file_could_inflate(self) -> None:
        record = BlockedWork.from_record(
            {
                "id": "block-1",
                "at": "2026-01-01T00:00:00+00:00",
                "source": "model",
                "reason_domain": "invented_domain",
                "reason_code": "invented_code",
                "action": "write_file",
                "subject_digest": "sub-abc",
                "class_shape": ["field:path:set", SECRET_PATH, "field:nope:set"],
                "fingerprint": "not-a-fingerprint",
                "outcome": "allowed",
                "work_claim": "completed",
                "enforcement": "enforced",
                "recovery_action": "run_recovery_recipe",
                "prompt": SECRET_PROMPT,
            }
        )
        self.assertEqual(record.reason_domain, "tool_failure")
        self.assertEqual(record.reason_code, "unspecified")
        self.assertEqual(record.outcome, "allowed")
        self.assertEqual(record.work_claim, "not_attempted")
        self.assertEqual(record.enforcement, "declared_not_enforced")
        self.assertEqual(record.recovery_action, "retry_with_scope")
        self.assertEqual(record.class_shape, ["field:path:set"])
        self.assertTrue(record.fingerprint.startswith("fp-"))
        dumped = json.dumps(record.to_dict())
        self.assertNotIn(SECRET_PATH, dumped)
        self.assertNotIn(SECRET_PROMPT, dumped)
        self.assertNotIn("prompt", dumped)

    def test_round_trip_is_stable(self) -> None:
        log = BlockLog(self.root, self.workspace)
        made = log.record(source="policy", reason_domain="workspace_escape", reason_code="symlink_escape",
                          action="edit_file", request={"path": "a"}, subject="a")
        again = BlockedWork.from_record(made.to_dict())
        self.assertEqual(again.to_dict()["class_shape"], made.class_shape)
        self.assertEqual(again.fingerprint, made.fingerprint)
        self.assertEqual(again.decision_id, made.decision_id)

    def test_garbage_input_degrades_to_a_blocked_record(self) -> None:
        record = BlockedWork.from_record({})  # type: ignore[arg-type]
        self.assertEqual(record.outcome, "blocked")
        self.assertEqual(record.reason_domain, "tool_failure")
        self.assertTrue(record.id.startswith("block-"))


class BlockToolTests(TempBase):
    def setUp(self) -> None:
        super().setUp()
        self.log = BlockLog(self.root, self.workspace)
        self.tool = make_block_tool(self.log)
        self.ctx = ToolContext(workspace=self.workspace)

    def test_tool_is_read_only_and_paged_bounded(self) -> None:
        self.assertTrue(self.tool.read_only)
        self.assertFalse(self.tool.destructive)
        self.assertEqual(self.tool.name, "blocks")
        self.log.record(source="tool_gate", reason_domain="gate_open", reason_code="gate_open", action="shell")
        capped = self.tool.handler({"view": "why", "limit": 9999}, self.ctx)
        self.assertTrue(capped.ok)
        self.assertLessEqual(capped.meta["count"], 50)

    def test_why_lists_the_answer_and_the_recovery(self) -> None:
        self.log.record(source="tool_gate", reason_domain="approval_denied", reason_code="approval_denied",
                        action="write_file", request={"path": SECRET_PATH}, subject=SECRET_PATH,
                        session_id="s1")
        result = self.tool.handler({"view": "why"}, self.ctx)
        self.assertTrue(result.ok)
        self.assertIn("approval_denied", result.content)
        self.assertIn("allow_action", result.content)
        self.assertIn("fingerprint:", result.content)
        self.assertNotIn(SECRET_PATH, result.content)

    def test_last_renders_one_record_in_full(self) -> None:
        self.log.record(source="policy", reason_domain="read_only_policy", reason_code="read_only_mode",
                        action="shell", subject="shell", session_id="s1")
        result = self.tool.handler({"view": "last"}, self.ctx)
        self.assertIn("Recovery:", result.content)
        self.assertIn("decision_id:", result.content)
        self.assertIn("claim boundary: not_attempted", result.content)

    def test_history_needs_an_action(self) -> None:
        self.log.record(source="tool_gate", reason_domain="tool_failure", reason_code="unspecified", action="shell")
        self.assertIn("History for shell", self.tool.handler({"view": "history", "action": "shell"}, self.ctx).content)
        self.assertFalse(self.tool.handler({"view": "history"}, self.ctx).ok)
        self.assertIn("No block history", self.tool.handler({"view": "history", "action": "ghost"}, self.ctx).content)

    def test_empty_store_and_bad_view(self) -> None:
        self.assertIn("Nothing has been blocked", self.tool.handler({}, self.ctx).content)
        self.assertIn("Nothing has been blocked", self.tool.handler({"view": "last"}, self.ctx).content)
        self.assertFalse(self.tool.handler({"view": "nonsense"}, self.ctx).ok)


class RecoveryAnchorTests(TempBase):
    def setUp(self) -> None:
        super().setUp()
        self.anchors = RecoveryAnchors(self.root)

    def _git(self, revision: str = "a" * 40) -> Path:
        git = self.workspace / ".git"
        (git / "refs" / "heads").mkdir(parents=True)
        (git / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
        (git / "refs" / "heads" / "main").write_text(revision + "\n", encoding="utf-8")
        (git / "index").write_bytes(b"index")
        old = time.time() - 600
        os.utime(git / "index", (old, old))
        return git

    def test_nothing_to_observe_is_a_prepared_anchor_that_promises_nothing(self) -> None:
        anchor = self.anchors.take(self.workspace, action="refactor the billing module")
        self.assertEqual(anchor.status, "prepared")
        self.assertEqual(anchor.base_revision, "")
        self.assertEqual(anchor.recipe, [])
        self.assertFalse(anchor.recoverable)
        self.assertEqual(anchor.dirty_state["state"], "not_observed")
        guidance = self.anchors.guidance(anchor.id, workspace=self.workspace)
        self.assertEqual(len(guidance), 1)
        self.assertIn("not a recoverable baseline", guidance[0])
        self.assertNotIn("git restore", guidance[0])

    def test_observed_baseline_writes_a_recipe_it_never_runs(self) -> None:
        self._git()
        (self.workspace / "notes.md").write_text("touched after the index", encoding="utf-8")
        anchor = self.anchors.take(self.workspace, action="refactor", note="ask before any delete")
        self.assertEqual(anchor.status, "baseline_observed")
        self.assertEqual(anchor.base_revision, "a" * 40)
        self.assertTrue(anchor.recoverable)
        joined = " ".join(anchor.recipe)
        self.assertIn("git restore --source " + "a" * 40, joined)
        self.assertIn("ask before any delete", joined)
        self.assertIn("refactor", joined)
        self.assertIn("OHAR runs none of these commands", joined)
        self.assertEqual(anchor.dirty_state["state"], "candidates_found")
        self.assertIn("notes.md", anchor.dirty_state["paths"])
        self.assertNotIn("index", anchor.dirty_state["paths"])
        self.assertEqual(self.anchors.guidance(anchor.id, workspace=self.workspace), anchor.recipe)

    def test_baseline_survives_a_new_instance(self) -> None:
        self._git(revision="c" * 40)
        anchor = self.anchors.take(self.workspace)
        again = RecoveryAnchors(self.root)
        self.assertEqual(again.latest(self.workspace).id, anchor.id)
        self.assertEqual(again.get(anchor.id).base_revision, "c" * 40)
        self.assertTrue(again.guidance(anchor.id, workspace=self.workspace))

    def test_packed_refs_and_detached_head(self) -> None:
        git = self.workspace / ".git"
        git.mkdir(parents=True)
        (git / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
        (git / "packed-refs").write_text("# pack\n" + "d" * 40 + " refs/heads/main\n", encoding="utf-8")
        (git / "index").write_bytes(b"x")
        self.assertEqual(self.anchors.take(self.workspace).base_revision, "d" * 40)
        (git / "HEAD").write_text("e" * 40 + "\n", encoding="utf-8")
        self.assertEqual(self.anchors.take(self.workspace).base_revision, "e" * 40)

    def test_unreadable_or_malformed_head_stays_prepared(self) -> None:
        git = self.workspace / ".git"
        git.mkdir(parents=True)
        (git / "HEAD").write_text("ref: refs/heads/missing\n", encoding="utf-8")
        (git / "index").write_bytes(b"x")
        anchor = self.anchors.take(self.workspace)
        self.assertEqual(anchor.status, "prepared")
        self.assertFalse(anchor.recoverable)

    def test_dirty_scan_is_bounded_and_reports_truncation(self) -> None:
        self._git()
        for index in range(MAX_DIRTY_PATHS + 6):
            (self.workspace / f"f{index:02d}.txt").write_text("x", encoding="utf-8")
        anchor = self.anchors.take(self.workspace)
        self.assertLessEqual(len(anchor.dirty_state["paths"]), MAX_DIRTY_PATHS)
        self.assertTrue(anchor.dirty_state["paths_truncated"])

    def test_clean_by_mtime_is_not_the_same_as_not_observed(self) -> None:
        self._git()
        (self.workspace / "old.txt").write_text("x", encoding="utf-8")
        old = time.time() - 1_200
        os.utime(self.workspace / "old.txt", (old, old))
        anchor = self.anchors.take(self.workspace)
        self.assertEqual(anchor.dirty_state["state"], "clean_by_mtime")
        self.assertTrue(any("likely nothing to save" in step for step in anchor.recipe))

    def test_guidance_for_another_workspace_is_refused(self) -> None:
        self._git()
        anchor = self.anchors.take(self.workspace)
        other = self.tmp / "elsewhere"
        other.mkdir()
        self.assertIsNone(self.anchors.latest(other))
        refused = self.anchors.guidance(anchor.id, workspace=other)
        self.assertEqual(len(refused), 1)
        self.assertIn("different workspace", refused[0])
        joined = " ".join(refused)
        self.assertNotIn("git restore", joined)
        self.assertNotIn("a" * 40, joined)

    def test_unknown_anchor_id_is_refused_not_invented(self) -> None:
        self.assertIn("No recovery anchor", self.anchors.guidance("anchor-nope", workspace=self.workspace)[0])

    def test_workspace_handle_is_stable_and_names_nothing(self) -> None:
        digest = workspace_digest_of(self.workspace)
        self.assertEqual(digest, workspace_digest_of(Path(str(self.workspace) + "/")))
        self.assertNotIn("repo", digest)
        self.assertNotEqual(digest, workspace_digest_of(self.tmp))

    def test_read_repairs_a_planted_recoverable_claim(self) -> None:
        planted = RecoveryAnchor(
            id="anchor-1",
            at="2026-01-01T00:00:00+00:00",
            workspace_digest=workspace_digest_of(self.workspace),
            status="prepared",
            base_revision="a" * 40,
            dirty_state={"state": "clean", "paths": ["/etc/shadow"]},
            recipe=["git restore --source a" * 40],
            recoverable=True,
        )
        restored = RecoveryAnchor.from_record(planted.to_dict())
        self.assertEqual(restored.status, "prepared")
        self.assertEqual(restored.base_revision, "")
        self.assertEqual(restored.recipe, [])
        self.assertFalse(restored.recoverable)
        self.assertEqual(restored.dirty_state["state"], "not_observed")
        self.assertEqual(restored.dirty_state["paths"], [])


if __name__ == "__main__":
    unittest.main()
