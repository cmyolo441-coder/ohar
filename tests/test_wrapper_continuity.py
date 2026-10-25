from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from typing import Any

from ohar.sessions import SessionMeta
from ohar.terminal import display_width, make_theme, strip_ansi
from ohar.wrapper import continuity
from ohar.wrapper.continuity import (
    CONTINUITY_SCHEMA_VERSION,
    MAX_LINE_CHARS,
    Continuity,
    ContinuityLine,
    build_continuity,
    render_continuity,
    resume_status,
)
from ohar.wrapper.evidence import EvidenceItem
from ohar.wrapper.state import SessionState

STAMP = "2026-01-05T00:00:00+00:00"


def make_meta(**overrides: Any) -> SessionMeta:
    fields: dict[str, Any] = {
        "id": "s-1",
        "title": "Port the ladder",
        "created_at": STAMP,
        "updated_at": STAMP,
        "message_count": 0,
        "cwd": "",
        "model": "openai/gpt-test",
        "lane": "main",
    }
    fields.update(overrides)
    return SessionMeta(**fields)


def make_state(**overrides: Any) -> SessionState:
    fields: dict[str, Any] = {
        "session_id": "s-1",
        "status": "open",
        "goal": "",
        "workspace": "",
        "model": "openai/gpt-test",
        "lane": "main",
        "updated_at": STAMP,
    }
    fields.update(overrides)
    return SessionState(**fields)


def make_item(result: str = "observed", *, digest: str = "sha256:aa") -> EvidenceItem:
    return EvidenceItem(
        kind="command", subject="pytest -q", digest=digest, result=result, exit_code=0, at=STAMP
    )


def tool_call(call_id: str = "c1") -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": call_id, "function": {"name": "shell", "arguments": "{}"}}],
    }


def tool_result(call_id: str = "c1") -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call_id, "name": "shell", "content": "ok"}


class ResumeStatusTests(unittest.TestCase):
    def test_nothing_recorded_is_fresh(self) -> None:
        self.assertEqual(resume_status(state=make_state(), messages=(), evidence=()), "fresh")
        self.assertEqual(resume_status(state=None, messages=(), evidence=()), "fresh")

    def test_recorded_progress_without_tail_is_still_safe(self) -> None:
        state = make_state(status="needs_review", goal="carry on")
        self.assertEqual(resume_status(state=state, messages=(), evidence=()), "conversation_safe")

    def test_complete_transcript_and_evidence_are_safe(self) -> None:
        messages = [{"role": "user", "content": "hi"}, tool_call(), tool_result()]
        self.assertEqual(
            resume_status(state=make_state(status="working"), messages=messages, evidence=[make_item()]),
            "conversation_safe",
        )

    def test_dangling_tool_call_blocks(self) -> None:
        messages = [{"role": "user", "content": "hi"}, tool_call("c9")]
        self.assertEqual(
            resume_status(state=make_state(status="working"), messages=messages, evidence=()), "blocked"
        )

    def test_unanswered_call_mid_transcript_loses_context(self) -> None:
        messages = [tool_call("c1"), {"role": "user", "content": "next"}, {"role": "assistant", "content": "ok"}]
        self.assertEqual(
            resume_status(state=make_state(status="working"), messages=messages, evidence=()),
            "context_lost",
        )

    def test_orphan_tool_result_loses_context(self) -> None:
        messages = [{"role": "user", "content": "hi"}, tool_result("trimmed-away")]
        self.assertEqual(
            resume_status(state=make_state(status="working"), messages=messages, evidence=()),
            "context_lost",
        )

    def test_state_without_record_but_with_messages_loses_context(self) -> None:
        self.assertEqual(
            resume_status(state=None, messages=[{"role": "user", "content": "hi"}], evidence=()),
            "context_lost",
        )

    def test_blocked_and_terminal_journeys_block(self) -> None:
        for status, expected in (
            ("blocked", "blocked"),
            ("done", "blocked"),
            ("cancelled", "blocked"),
        ):
            with self.subTest(status=status):
                self.assertEqual(
                    resume_status(state=make_state(status=status), messages=[{"role": "user", "content": "hi"}], evidence=[make_item()]),
                    expected,
                )

    def test_malformed_inputs_block(self) -> None:
        cases: tuple[dict[str, Any], ...] = (
            {"state": {"status": "working"}, "messages": (), "evidence": ()},
            {"state": make_state(status="not_a_status"), "messages": (), "evidence": ()},
            {"state": make_state(), "messages": "not-a-transcript", "evidence": ()},
            {"state": make_state(), "messages": [{"content": "no role"}], "evidence": ()},
            {"state": make_state(), "messages": [42], "evidence": ()},
            {"state": make_state(), "messages": (), "evidence": [make_item(), {"result": "observed"}]},
            {"state": make_state(), "messages": (), "evidence": "command"},
        )
        for case in cases:
            with self.subTest(case=case):
                self.assertEqual(resume_status(**case), "blocked")


class BuildContinuityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.workspace = self.base / "workspace"
        self.workspace.mkdir()
        self.moved = self.base / "moved"
        self.moved.mkdir()
        self.gone = self.base / "gone"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def build(self, **overrides: Any) -> Continuity:
        kwargs: dict[str, Any] = {
            "meta": make_meta(cwd=str(self.workspace)),
            "state": make_state(workspace=str(self.workspace)),
            "workspace": self.workspace,
        }
        kwargs.update(overrides)
        return build_continuity(**kwargs)

    def test_workspace_drift_ladder(self) -> None:
        same = self.build()
        self.assertEqual(same.workspace_drift, "same")
        moved = self.build(workspace=self.moved)
        self.assertEqual(moved.workspace_drift, "moved")
        missing = self.build(state=make_state(workspace=str(self.gone)))
        self.assertEqual(missing.workspace_drift, "missing")
        unknown = self.build(meta=make_meta(cwd=""), state=make_state(workspace=""))
        self.assertEqual(unknown.workspace_drift, "unknown")

    def test_missing_workspace_outranks_a_safe_conversation(self) -> None:
        continuity_row = self.build(
            state=make_state(status="working", workspace=str(self.gone)),
            evidence=[make_item()],
            messages=[{"role": "user", "content": "hi"}],
        )
        self.assertEqual(continuity_row.resume_status, "conversation_safe")
        self.assertEqual(continuity_row.next_action, "relocate_workspace")
        self.assertIn("gone", continuity_row.lines[0].text)

    def test_lane_drift_compares_records_not_assumptions(self) -> None:
        self.assertEqual(self.build().lane_drift, "same")
        drifted = self.build(state=make_state(lane="fast", model="anthropic/x"))
        self.assertEqual(drifted.lane_drift, "moved")
        self.assertEqual(self.build(state=None).lane_drift, "unknown")
        self.assertEqual(
            self.build(state=make_state(lane="", model="")).lane_drift,
            "unknown",
        )

    def test_memory_state_is_never_reassuring_by_default(self) -> None:
        self.assertEqual(self.build(memory_records=3).memory, "available")
        self.assertEqual(self.build(memory_records=0).memory, "not_included")
        for value in (None, -1, True, "3"):
            with self.subTest(value=value):
                self.assertEqual(self.build(memory_records=value).memory, "unknown")

    def test_plan_counts_read_the_planner_shape(self) -> None:
        plan = {
            "steps": [
                {"step": "Inspect", "status": "completed"},
                {"step": "Port", "status": "in_progress"},
                {"step": "Test", "status": "pending"},
            ],
            "updated_at": STAMP,
            "counts": {"pending": 1, "in_progress": 1, "completed": 1},
        }
        self.assertEqual(self.build(plan=plan).plan_counts, {"pending": 1, "in_progress": 1, "completed": 1})
        without_counts = self.build(plan={"steps": plan["steps"]})
        self.assertEqual(without_counts.plan_counts, {"pending": 1, "in_progress": 1, "completed": 1})
        empty = self.build(plan=None)
        self.assertEqual(empty.plan_counts, {"pending": 0, "in_progress": 0, "completed": 0})

    def test_malformed_plan_degrades_to_zero_and_says_so(self) -> None:
        for plan in ({"steps": "nope"}, {"steps": [{"step": "a", "status": "nearly"}]}, "not-a-mapping", {"counts": {"pending": -2}}):
            with self.subTest(plan=plan):
                row = self.build(plan=plan)
                self.assertEqual(row.plan_counts, {"pending": 0, "in_progress": 0, "completed": 0})
                context = [line for line in row.lines if line.domain == "context"]
                self.assertTrue(context)
                self.assertIn("plan", context[0].text.lower())

    def test_in_progress_steps_without_evidence_get_a_line_and_an_action(self) -> None:
        plan = {"steps": [{"step": "Port", "status": "in_progress"}]}
        row = self.build(state=make_state(status="working", workspace=str(self.workspace)), plan=plan)
        context = [line for line in row.lines if line.domain == "context"]
        self.assertEqual(len(context), 1)
        self.assertIn("no observed evidence", context[0].text)
        self.assertEqual(row.next_action, "review_unbacked_plan_steps")
        backed = self.build(
            state=make_state(status="working", workspace=str(self.workspace)),
            plan=plan,
            evidence=[make_item()],
            memory_records=1,
        )
        self.assertEqual(backed.next_action, "continue_conversation")

    def test_trimmed_tail_is_reported_from_the_recorded_count(self) -> None:
        row = self.build(
            meta=make_meta(cwd=str(self.workspace), message_count=9),
            messages=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}],
        )
        context = [line for line in row.lines if line.domain == "context"]
        self.assertTrue(any("Context was trimmed" in line.text for line in context))
        self.assertLessEqual(len(row.lines), 4)

    def test_three_lines_without_context_and_four_with(self) -> None:
        plain = self.build()
        self.assertEqual([line.domain for line in plain.lines], ["workspace", "resume", "memory"])
        rich = self.build(
            state=make_state(status="working", workspace=str(self.workspace)),
            plan={"steps": [{"step": "Port", "status": "in_progress"}]},
        )
        self.assertEqual([line.domain for line in rich.lines], ["workspace", "resume", "memory", "context"])
        self.assertTrue(all(isinstance(line, ContinuityLine) for line in rich.lines))

    def test_blocked_resume_is_the_most_severe_action(self) -> None:
        row = self.build(
            state=make_state(status="done", workspace=str(self.gone)),
            messages=[tool_call("c1")],
        )
        self.assertEqual(row.resume_status, "blocked")
        self.assertEqual(row.next_action, "review_persisted_records")

    def test_malformed_meta_produces_no_claim(self) -> None:
        row = build_continuity(meta=None, state=make_state(), workspace=self.workspace)  # type: ignore[arg-type]
        self.assertEqual(row.resume_status, "blocked")
        self.assertEqual(row.workspace_drift, "unknown")
        self.assertEqual(row.lane_drift, "unknown")
        self.assertEqual(row.memory, "unknown")
        self.assertEqual(row.next_action, "repair_session_record")
        self.assertEqual(len(row.lines), 3)

    def test_untrusted_text_is_sanitized_and_bounded(self) -> None:
        messy = self.base / "\x1b[31mevil\x00 dir" / ("nested" * 60)
        row = self.build(state=make_state(workspace=str(messy)))
        self.assertEqual(row.workspace_drift, "missing")
        for line in row.lines:
            self.assertNotIn("\x1b", line.text)
            self.assertNotIn("\x00", line.text)
            self.assertLessEqual(len(line.text), MAX_LINE_CHARS)
        self.assertLessEqual(len(row.headline), MAX_LINE_CHARS)

    def test_to_dict_shape(self) -> None:
        row = self.build(memory_records=2, plan={"steps": [{"step": "Port", "status": "pending"}]})
        payload = row.to_dict()
        self.assertEqual(
            set(payload),
            {
                "schema_version",
                "resume_status",
                "workspace_drift",
                "lane_drift",
                "plan_counts",
                "memory",
                "headline",
                "lines",
                "next_action",
                "claim_boundary",
            },
        )
        self.assertEqual(payload["schema_version"], CONTINUITY_SCHEMA_VERSION)
        self.assertEqual(continuity.CONTINUITY_SCHEMA_VERSION, "ohar_continuity/v1")
        self.assertEqual(payload["memory"], "available")
        self.assertEqual(payload["lines"][0], {"domain": "workspace", "text": row.lines[0].text})
        self.assertNotIsInstance(payload["lines"], tuple)
        self.assertIn("does not resume a session", payload["claim_boundary"])

    def test_projection_never_writes_a_record(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            previous = os.environ.get("OHAR_HOME")
            os.environ["OHAR_HOME"] = home
            try:
                self.build(
                    state=make_state(status="working", workspace=str(self.workspace)),
                    messages=[{"role": "user", "content": "hi"}, tool_call()],
                    evidence=[make_item()],
                )
                resume_status(state=make_state(), messages=(), evidence=())
                rendered = list(Path(home).rglob("*"))
            finally:
                if previous is None:
                    os.environ.pop("OHAR_HOME", None)
                else:
                    os.environ["OHAR_HOME"] = previous
            self.assertEqual(rendered, [])


class RenderContinuityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tmp.name) / "ws"
        self.workspace.mkdir()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def row(self, **overrides: Any) -> Continuity:
        kwargs: dict[str, Any] = {
            "meta": make_meta(cwd=str(self.workspace)),
            "state": make_state(workspace=str(self.workspace), status="working", goal="carry on"),
            "workspace": self.workspace,
            "evidence": [make_item()],
            "memory_records": 1,
        }
        kwargs.update(overrides)
        return build_continuity(**kwargs)

    def test_every_rendered_line_fits_the_width(self) -> None:
        plain = make_theme("aurora", color=False)
        for width in (24, 40, 76):
            for glyphs in ("unicode", "ascii"):
                text = render_continuity(self.row(), plain, width=width, glyphs=glyphs)
                with self.subTest(width=width, glyphs=glyphs):
                    self.assertTrue(text)
                    for line in text.splitlines():
                        self.assertLessEqual(display_width(strip_ansi(line)), width)

    def test_plain_theme_emits_no_escape_sequences(self) -> None:
        plain = make_theme("aurora", color=False)
        text = render_continuity(self.row(), plain, width=70)
        self.assertNotIn("\x1b", text)
        self.assertIn("Continuity", text)
        self.assertIn("workspace:", text)
        self.assertIn("continue_conversation", text)

    def test_color_and_blocked_states(self) -> None:
        color = make_theme("aurora", color=True, depth="truecolor")
        self.assertIn("\x1b[38;2;", render_continuity(self.row(), color, width=70))
        blocked = render_continuity(self.row(state=make_state(status="blocked")), color, width=70)
        self.assertIn("review_persisted_records", blocked)
        self.assertIn("×", blocked)

    def test_ascii_glyphs_replace_unicode_marks(self) -> None:
        plain = make_theme("aurora", color=False)
        text = render_continuity(self.row(), plain, width=70, glyphs="ascii")
        self.assertIn("+", text.splitlines()[0])
        self.assertNotIn("●", text)

    def test_degenerate_width_still_renders(self) -> None:
        plain = make_theme("aurora", color=False)
        for width in (0, -5, "wide"):
            with self.subTest(width=width):
                text = render_continuity(self.row(), plain, width=width)  # type: ignore[arg-type]
                self.assertTrue(text.splitlines())


if __name__ == "__main__":
    unittest.main()
