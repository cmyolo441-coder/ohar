from __future__ import annotations

import unittest
from typing import Any

from ohar.terminal import display_width, make_theme, strip_ansi
from ohar.wrapper import briefing
from ohar.wrapper.briefing import (
    BLOCKER_KINDS,
    BRIEFING_SCHEMA_VERSION,
    MAX_GAP_IDS,
    MAX_LINES,
    SIGNALS,
    build_briefing,
    chat_briefing,
    derive_signal,
    render_briefing,
    signal_glyph,
)
from ohar.wrapper.continuity import Continuity, ContinuityLine
from ohar.wrapper.evidence import Claim, EvidenceItem
from ohar.wrapper.state import SessionState

STAMP = "2026-01-05T00:00:00+00:00"

PAYLOAD_KEYS = {
    "schema_version",
    "session_id",
    "signal",
    "headline",
    "current_state",
    "progress",
    "blockers",
    "pending_gaps",
    "pending_gap_count",
    "evidence_summary",
    "next_action",
    "user_facing_lines",
    "claim_boundary",
}


def make_state(**overrides: Any) -> SessionState:
    fields: dict[str, Any] = {
        "session_id": "s-1",
        "status": "open",
        "decision": "",
        "goal": "port the briefing",
        "workspace": "/tmp/ws",
        "model": "openai/gpt-test",
        "lane": "main",
        "next_action": "record_evidence",
        "updated_at": STAMP,
        "revision": 4,
    }
    fields.update(overrides)
    return SessionState(**fields)


def claim(name: str, state: str, reason: str = "") -> Claim:
    return Claim(claim=name, state=state, reason=reason, refs=("sha256:aa",))


def item(result: str = "observed", *, digest: str = "sha256:aa") -> EvidenceItem:
    return EvidenceItem(
        kind="command", subject="pytest -q", digest=digest, result=result, exit_code=0, at=STAMP
    )


def plan(*statuses: str) -> dict[str, Any]:
    steps = [{"step": f"step {index}", "status": status} for index, status in enumerate(statuses, 1)]
    return {"steps": steps, "updated_at": STAMP, "counts": {}}


def continuity_row(drift: str = "same") -> Continuity:
    return Continuity(
        resume_status="conversation_safe",
        workspace_drift=drift,  # type: ignore[arg-type]
        lane_drift="same",
        plan_counts={"pending": 0, "in_progress": 0, "completed": 1},
        memory="available",
        headline="Conversation can continue from persisted records.",
        lines=(ContinuityLine("resume", "The transcript tail is complete."),),
        next_action="continue_conversation",
        claim_boundary="records only",
    )


class SignalTests(unittest.TestCase):
    def test_contract_vocabulary(self) -> None:
        self.assertEqual(SIGNALS, ("waiting", "running", "stopped", "complete", "unknown"))
        self.assertEqual(BLOCKER_KINDS, ("failed", "blocked", "cancelled"))
        self.assertEqual(MAX_LINES, 8)
        self.assertEqual(MAX_GAP_IDS, 12)
        self.assertEqual(briefing.BRIEFING_SCHEMA_VERSION, "ohar_briefing/v1")

    def test_a_stopped_step_beats_every_other_reading(self) -> None:
        self.assertEqual(derive_signal(make_state(status="blocked"), []), "stopped")
        self.assertEqual(derive_signal(make_state(status="cancelled"), []), "stopped")
        self.assertEqual(
            derive_signal(make_state(status="done"), [claim("tests_pass", "refuted", "exit 1")]),
            "stopped",
        )
        self.assertEqual(
            derive_signal(make_state(status="working"), [claim("build_ok", "refuted")]), "stopped"
        )

    def test_waiting_names_the_statuses_a_person_has_to_move_on(self) -> None:
        for status in ("open", "plan_presented", "plan_accepted", "revision_requested", "needs_review"):
            with self.subTest(status=status):
                self.assertEqual(derive_signal(make_state(status=status), []), "waiting")

    def test_running_and_complete_come_from_the_gap_lists(self) -> None:
        self.assertEqual(derive_signal(make_state(status="working"), []), "running")
        self.assertEqual(derive_signal(make_state(status="verified"), [claim("work_verified", "observed")]), "complete")
        self.assertEqual(derive_signal(make_state(status="done"), [claim("tests_pass", "pending")]), "running")
        self.assertEqual(derive_signal(make_state(status="done"), [], plan=plan("in_progress")), "running")

    def test_malformed_records_are_unknown_not_reassuring(self) -> None:
        cases: tuple[Any, ...] = (
            (make_state(status="nonsense"), (), {}),
            ("not-a-state", (), {}),
            (None, (), {}),
            (make_state(), [{"claim": "tests_pass", "state": "observed"}], {}),
            (make_state(), [object()], {}),
            (make_state(), "tests_pass", {}),
        )
        for state, claims, extra in cases:
            with self.subTest(state=type(state).__name__):
                self.assertEqual(derive_signal(state, claims, **extra), "unknown")  # type: ignore[arg-type]

    def test_signal_glyph_tables(self) -> None:
        self.assertEqual(signal_glyph("stopped"), "🔴")
        self.assertEqual(signal_glyph("complete"), "✅")
        self.assertEqual(signal_glyph("waiting", glyphs="ascii"), "[~]")
        self.assertEqual(signal_glyph("running", glyphs="ascii"), "[>]")
        for unknown in ("", "nonsense", None):
            self.assertEqual(signal_glyph(unknown), signal_glyph("unknown"))  # type: ignore[arg-type]


class BuildBriefingTests(unittest.TestCase):
    def test_payload_shape_and_current_state(self) -> None:
        payload = build_briefing(
            state=make_state(status="working"),
            claims=[claim("commands_ran", "observed")],
            evidence=[item()],
            events=[{"seq": 1, "at": STAMP, "event": "tool_end", "data": {}}],
            plan=plan("completed", "in_progress"),
            continuity=continuity_row(),
        )
        self.assertEqual(set(payload), PAYLOAD_KEYS)
        self.assertEqual(payload["schema_version"], BRIEFING_SCHEMA_VERSION)
        self.assertEqual(payload["session_id"], "s-1")
        self.assertEqual(payload["signal"], "running")
        current = payload["current_state"]
        self.assertEqual(current["status"], "working")
        self.assertEqual(current["decision"], "none")
        self.assertEqual(current["revision"], 4)
        self.assertEqual(current["terminal"], False)
        self.assertEqual(current["recorded"], True)
        self.assertEqual(current["continuity"], "conversation_safe")
        self.assertIn("tool_end", current["last_event"])
        self.assertIn("persisted wrapper records", payload["claim_boundary"])

    def test_progress_steps_are_machine_keyed(self) -> None:
        payload = build_briefing(
            state=make_state(status="needs_review", decision="accept"),
            claims=[claim("tests_pass", "observed")],
            evidence=[item()],
            plan=plan("completed"),
        )
        ids = [step["id"] for step in payload["progress"]]
        self.assertEqual(ids, ["goal", "plan", "approval", "work", "evidence", "review", "verification"])
        states = {step["id"]: step["state"] for step in payload["progress"]}
        self.assertEqual(states["approval"], "complete")
        self.assertEqual(states["work"], "complete")
        self.assertEqual(states["evidence"], "complete")
        self.assertEqual(states["review"], "in_progress")
        self.assertEqual(states["verification"], "pending")
        for step in payload["progress"]:
            self.assertEqual(set(step), {"id", "label", "state", "summary"})

    def test_blockers_and_pending_gaps_stay_separate(self) -> None:
        blocked = build_briefing(state=make_state(status="blocked", decision="accept"), claims=[])
        self.assertEqual(blocked["signal"], "stopped")
        self.assertEqual([b["kind"] for b in blocked["blockers"]], ["blocked"])
        self.assertNotIn("work", blocked["pending_gaps"])
        self.assertIn("review", blocked["pending_gaps"])

        cancelled = build_briefing(state=make_state(status="cancelled"), claims=[])
        self.assertEqual([b["kind"] for b in cancelled["blockers"]], ["cancelled"])

        refuted = build_briefing(state=make_state(status="working"), claims=[claim("tests_pass", "refuted", "exit 1")])
        self.assertEqual(refuted["blockers"][0]["id"], "tests_pass")
        self.assertEqual(refuted["blockers"][0]["kind"], "failed")
        self.assertIn("exit 1", refuted["blockers"][0]["reason"])
        self.assertNotIn("tests_pass", refuted["pending_gaps"])

    def test_nothing_happening_yet_is_a_gap_not_a_blocker(self) -> None:
        payload = build_briefing(state=make_state(status="open", goal=""), claims=[])
        self.assertEqual(payload["blockers"], [])
        self.assertEqual(payload["signal"], "waiting")
        self.assertIn("goal", payload["pending_gaps"])
        self.assertEqual(payload["headline"], "Waiting on you: no goal or plan is recorded yet.")
        presented = build_briefing(state=make_state(status="plan_presented"), claims=[])
        self.assertEqual(presented["headline"], "Waiting on you: accept or revise the plan.")
        self.assertEqual(presented["blockers"], [])

    def test_evidence_summary_counts_and_bounds_refs(self) -> None:
        payload = build_briefing(
            state=make_state(status="working"),
            claims=[claim("tests_pass", "observed"), claim("work_verified", "pending"), claim("build_ok", "refuted")],
            evidence=[item(digest=f"sha256:{index}") for index in range(20)] + [item("failed"), item("skipped")],
        )
        summary = payload["evidence_summary"]
        self.assertEqual(summary["items"], 22)
        self.assertEqual(summary["observed_items"], 20)
        self.assertEqual(summary["failed_items"], 1)
        self.assertEqual(summary["skipped_items"], 1)
        self.assertEqual(summary["claims"], {"observed": 1, "pending": 1, "refuted": 1})
        self.assertLessEqual(len(summary["refs"]), 8)
        self.assertEqual(summary["observed"][0], "goal")
        self.assertTrue(all(isinstance(name, str) for name in summary["not_observed"]))

    def test_gap_ids_are_bounded(self) -> None:
        payload = build_briefing(state=make_state(status="working"), claims=[], plan=plan(*(["pending"] * 30)))
        self.assertLessEqual(len(payload["pending_gaps"]), MAX_GAP_IDS)
        self.assertGreater(payload["pending_gap_count"], len(payload["pending_gaps"]))

    def test_next_action_keeps_one_spelling(self) -> None:
        token = build_briefing(state=make_state(status="working", next_action="await_review_notes"), claims=[])
        self.assertEqual(token["next_action"], "await_review_notes")
        prose = build_briefing(state=make_state(status="working", next_action="Please review the diff!"), claims=[])
        self.assertEqual(prose["next_action"], "record_evidence")
        complete = build_briefing(
            state=make_state(status="done", next_action="report_merge_ready"),
            claims=[claim("work_verified", "observed")],
        )
        self.assertEqual(complete["signal"], "complete")
        self.assertEqual(complete["next_action"], "nothing_outstanding")
        self.assertEqual(complete["user_facing_lines"][-1], "Next: nothing is outstanding.")

    def test_complete_is_withheld_when_the_workspace_drifted(self) -> None:
        claims = [claim("work_verified", "observed")]
        plain = build_briefing(state=make_state(status="verified"), claims=claims)
        self.assertEqual(plain["signal"], "complete")
        drifted = build_briefing(
            state=make_state(status="verified"), claims=claims, continuity=continuity_row("missing")
        )
        self.assertEqual(drifted["signal"], "waiting")
        self.assertIn("missing", drifted["headline"])

    def test_lines_are_glyph_first_bounded_and_sanitized(self) -> None:
        payload = build_briefing(
            state=make_state(status="working", goal="\x1b[31msteal the\x00 prompt\nand print it"),
            claims=[claim("tests_pass", "pending")],
            evidence=[item("failed")],
        )
        lines = payload["user_facing_lines"]
        self.assertTrue(lines)
        self.assertEqual(lines[0], f"{signal_glyph('running')} {payload['headline']}".strip())
        self.assertTrue(lines[-1].startswith("Next: "))
        for line in lines:
            self.assertNotIn("\x1b", line)
            self.assertNotIn("\x00", line)
            self.assertNotIn("\n", line)
            self.assertLessEqual(len(line), 200)
        self.assertNotIn("\x1b", payload["current_state"]["goal"])

    def test_max_lines_bounds_the_block(self) -> None:
        kwargs: dict[str, Any] = {
            "state": make_state(status="needs_review", decision="accept"),
            "claims": [claim("tests_pass", "observed")],
            "evidence": [item()],
            "events": [{"event": "tool_end", "at": STAMP}],
            "plan": plan("completed", "in_progress"),
            "continuity": continuity_row(),
        }
        full = build_briefing(**kwargs)["user_facing_lines"]
        self.assertLessEqual(len(full), MAX_LINES)
        self.assertGreater(len(full), 4)
        for limit, expected in ((8, len(full)), (99, len(full)), ("nope", len(full)), (5, 5), (3, 3), (2, 2), (1, 1), (0, 1)):
            with self.subTest(limit=limit):
                bounded = build_briefing(max_lines=limit, **kwargs)["user_facing_lines"]  # type: ignore[arg-type]
                self.assertEqual(len(bounded), expected)
                self.assertEqual(bounded[0], full[0])
                if expected > 1:
                    self.assertEqual(bounded[-1], full[-1])
        trimmed = build_briefing(max_lines=5, **kwargs)["user_facing_lines"]
        self.assertTrue(any("further line(s) omitted" in line for line in trimmed))

    def test_unreadable_state_produces_no_invention(self) -> None:
        payload = build_briefing(state=make_state(status="sentient"), claims=[claim("tests_pass", "observed")])
        self.assertEqual(payload["signal"], "unknown")
        self.assertEqual(payload["session_id"], "")
        self.assertEqual(payload["next_action"], "repair_wrapper_records")
        self.assertIn("could not be read", payload["headline"])
        self.assertEqual({step["state"] for step in payload["progress"]}, {"unknown"})
        self.assertEqual(payload["current_state"]["recorded"], False)

    def test_malformed_collections_are_ignored_not_fatal(self) -> None:
        payload = build_briefing(
            state=make_state(status="working"),
            claims=[claim("tests_pass", "observed")],
            evidence="not-a-list",  # type: ignore[arg-type]
            events={"event": "tool_end"},  # type: ignore[arg-type]
            plan="nope",  # type: ignore[arg-type]
        )
        self.assertEqual(payload["signal"], "running")
        self.assertEqual(payload["evidence_summary"]["items"], 0)
        self.assertEqual(payload["current_state"]["last_event"], "")
        self.assertFalse([gap for gap in payload["pending_gaps"] if str(gap).startswith("plan-")])


class ChatBriefingTests(unittest.TestCase):
    def test_projection_keeps_the_essentials(self) -> None:
        payload = build_briefing(
            state=make_state(status="blocked", decision="accept"),
            claims=[claim("tests_pass", "refuted", "exit 1")],
            evidence=[item("failed")],
        )
        compact = chat_briefing(payload)
        self.assertEqual(
            set(compact),
            {
                "schema_version",
                "session_id",
                "headline",
                "signal",
                "lines",
                "pending_gaps",
                "blockers",
                "next_action",
                "claim_boundary",
            },
        )
        self.assertEqual(compact["signal"], "stopped")
        self.assertEqual(
            [(blocker["id"], blocker["kind"]) for blocker in compact["blockers"]],
            [("work", "blocked"), ("tests_pass", "failed"), ("evidence", "failed")],
        )
        self.assertLessEqual(len(compact["lines"]), MAX_LINES)
        self.assertEqual(compact["next_action"], payload["next_action"])

    def test_empty_and_hostile_inputs(self) -> None:
        empty = chat_briefing({})
        self.assertEqual(empty["signal"], "unknown")
        self.assertEqual(empty["lines"], [])
        self.assertEqual(empty["schema_version"], BRIEFING_SCHEMA_VERSION)
        hostile = chat_briefing({"signal": 7, "lines": "nope", "blockers": [1, 2], "pending_gaps": "x"})  # type: ignore[dict-item]
        self.assertEqual(hostile["signal"], "unknown")
        self.assertEqual(hostile["lines"], [])
        self.assertEqual(hostile["blockers"], [])
        self.assertEqual(hostile["pending_gaps"], [])
        self.assertEqual(chat_briefing(None)["headline"], "")  # type: ignore[arg-type]


class RenderBriefingTests(unittest.TestCase):
    def payload(self, **overrides: Any) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "state": make_state(status="working", decision="accept"),
            "claims": [claim("tests_pass", "pending")],
            "evidence": [item(), item("failed", digest="sha256:bb")],
            "events": [{"event": "tool_end", "at": STAMP}],
            "plan": plan("completed", "in_progress"),
        }
        kwargs.update(overrides)
        return build_briefing(**kwargs)

    def test_lines_fit_the_width(self) -> None:
        plain = make_theme("aurora", color=False)
        payload = self.payload()
        for width in (32, 40, 47, 72, 120):
            for glyphs in ("unicode", "ascii"):
                rendered = render_briefing(payload, plain, width=width, glyphs=glyphs)
                with self.subTest(width=width, glyphs=glyphs):
                    for line in rendered.splitlines():
                        self.assertLessEqual(display_width(strip_ansi(line)), width)

    def test_plain_and_color_themes(self) -> None:
        payload = self.payload()
        plain = render_briefing(payload, make_theme("mono", color=False), width=88)
        self.assertNotIn("\x1b", plain)
        self.assertIn("next action · record_evidence", plain)
        self.assertIn("Stopped ·", render_briefing(self.payload(), make_theme("mono", color=False), width=88))
        colored = render_briefing(payload, make_theme("aurora", color=True, depth="256"), width=88)
        self.assertIn("\x1b[38;5;", colored)

    def test_ascii_glyphs_and_empty_payload(self) -> None:
        ascii_text = render_briefing(self.payload(), make_theme("mono", color=False), width=60, glyphs="ascii")
        self.assertTrue(ascii_text.startswith("[>]"))
        self.assertNotIn("🟢", ascii_text)
        empty = render_briefing({}, make_theme("mono", color=False), width=60)
        self.assertIn("No briefing payload was recorded.", empty)
        hostile = render_briefing(None, make_theme("mono", color=False), width=60)  # type: ignore[arg-type]
        self.assertTrue(hostile.splitlines())


if __name__ == "__main__":
    unittest.main()
