from __future__ import annotations

import os
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest import mock

from ohar.sessions import SessionStore
from ohar.terminal import display_width, make_theme, strip_ansi
from ohar.wrapper import records as wrapper_records
from ohar.wrapper.briefing import SIGNALS
from ohar.wrapper.evidence import Claim, EvidenceItem
from ohar.wrapper.evidence import record as record_evidence
from ohar.wrapper.mission import (
    DEFAULT_STALE_MINUTES,
    JOURNEY_STATES,
    MISSION_SCHEMA_VERSION,
    autonomy,
    build_mission,
    journey_state,
    list_missions,
    mission_rows,
    recovery,
    render_mission,
)
from ohar.wrapper.state import STATE_RECORD, SessionState, note, open_state
from ohar.wrapper.state import record_decision
from ohar.wrapper.state import transition

LADDER = ("open", "plan_presented", "plan_accepted", "working", "needs_review", "verified", "done")
STALE_SECONDS = (DEFAULT_STALE_MINUTES + 30) * 60


def stamp(offset_seconds: float = 0.0) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)).isoformat(timespec="seconds")


def make_state(**overrides: Any) -> SessionState:
    fields: dict[str, Any] = {
        "session_id": "s-1",
        "status": "open",
        "goal": "port mission control",
        "workspace": "/tmp/ws",
        "model": "openai/gpt-test",
        "lane": "main",
        "updated_at": stamp(),
    }
    fields.update(overrides)
    return SessionState(**fields)


def claim(name: str, state: str, reason: str = "") -> Claim:
    return Claim(claim=name, state=state, reason=reason, refs=("sha256:aa",))


def plant(session_id: str, **overrides: Any) -> None:
    """Write one state record directly, so timestamps can be older than this run."""
    row = make_state(session_id=session_id, **overrides)
    wrapper_records.write_record(STATE_RECORD, session_id, row.to_dict())


class WrapperHomeCase(unittest.TestCase):
    """Every record read or write stays inside a temporary OHAR_HOME."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.home = self.base / "home"
        self.workspace = self.base / "workspace"
        self.workspace.mkdir()
        self.patcher = mock.patch.dict(os.environ, {"OHAR_HOME": str(self.home)})
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(mock.patch.stopall)

    def drive(self, session_id: str, status: str, *, skill: str = "") -> SessionState:
        state, _ = open_state(session_id, goal="port mission control", workspace=str(self.workspace))
        for step in LADDER[1 : LADDER.index(status) + 1]:
            state = record_decision(session_id, "accept") if step == "plan_accepted" else transition(session_id, step)
        if skill:
            state = note(session_id, skill=skill)
        return state


class JourneyTests(unittest.TestCase):
    def test_contract_vocabulary(self) -> None:
        self.assertEqual(
            JOURNEY_STATES,
            (
                "drafted",
                "approved",
                "working",
                "blocked",
                "review",
                "verified",
                "finished",
                "cancelled",
                "broken_link",
            ),
        )
        self.assertEqual(MISSION_SCHEMA_VERSION, "ohar_mission/v1")
        self.assertEqual(DEFAULT_STALE_MINUTES, 60)

    def test_the_ladder_walks_the_recorded_statuses(self) -> None:
        cases = (
            ("open", "waiting", "drafted"),
            ("plan_presented", "waiting", "drafted"),
            ("revision_requested", "waiting", "drafted"),
            ("plan_accepted", "waiting", "approved"),
            ("working", "running", "working"),
            ("blocked", "stopped", "blocked"),
            ("needs_review", "waiting", "review"),
            ("verified", "complete", "verified"),
            ("done", "complete", "finished"),
            ("cancelled", "stopped", "cancelled"),
        )
        for status, signal, expected in cases:
            with self.subTest(status=status):
                rows = [claim("work_verified", "observed")] if expected in ("verified", "finished") else []
                self.assertEqual(journey_state(make_state(status=status), rows, signal=signal), expected)

    def test_mismatched_linkage_fails_to_broken_link(self) -> None:
        mismatches = (
            ("working", "waiting"),
            ("working", "complete"),
            ("blocked", "running"),
            ("blocked", "waiting"),
            ("open", "complete"),
            ("open", "running"),
            ("done", "waiting"),
            ("cancelled", "running"),
            ("needs_review", "running"),
        )
        for status, signal in mismatches:
            with self.subTest(status=status, signal=signal):
                self.assertEqual(journey_state(make_state(status=status), [], signal=signal), "broken_link")

    def test_refuted_verification_under_a_finished_ladder_is_a_broken_link(self) -> None:
        for status in ("verified", "done"):
            with self.subTest(status=status):
                refuted = [claim("work_verified", "refuted", "postcondition failed")]
                self.assertEqual(journey_state(make_state(status=status), refuted, signal="stopped"), "broken_link")
        # The same refutation one rung earlier is a plain stop, not a broken link.
        self.assertEqual(
            journey_state(make_state(status="working"), [claim("work_verified", "refuted")], signal="stopped"),
            "working",
        )

    def test_unreadable_records_are_a_broken_link(self) -> None:
        self.assertEqual(journey_state(make_state(status="sentient"), [], signal="waiting"), "broken_link")
        self.assertEqual(journey_state(None, [], signal="waiting"), "broken_link")  # type: ignore[arg-type]
        self.assertEqual(journey_state(make_state(), [{"claim": "tests_pass"}], signal="waiting"), "broken_link")
        self.assertEqual(journey_state(make_state(), "tests_pass", signal="waiting"), "broken_link")  # type: ignore[arg-type]

    def test_every_signal_is_used_by_the_ladder(self) -> None:
        reachable: set[str] = set()
        for status in ("open", "working", "blocked", "needs_review", "verified", "done", "cancelled"):
            for signal in SIGNALS:
                if journey_state(make_state(status=status), [], signal=signal) != "broken_link":
                    reachable.add(signal)
        self.assertEqual(reachable, set(SIGNALS) - {"unknown"})


class RecoveryTests(unittest.TestCase):
    def test_each_rung_carries_its_own_hint(self) -> None:
        seen: dict[str, str] = {}
        for name in JOURNEY_STATES:
            row = recovery(name, resume_safe=name in ("drafted", "working"))
            self.assertEqual(set(row), {"status", "resume_safe", "next_action", "hint"})
            self.assertTrue(all(isinstance(value, str) and value for value in row.values()))
            seen[row["next_action"]] = name
        self.assertEqual(len(seen), len(JOURNEY_STATES))
        self.assertEqual(
            {recovery(name, resume_safe=False)["status"] for name in ("blocked", "broken_link")},
            {"recovery_blocked"},
        )
        self.assertEqual(recovery("working", resume_safe=True)["resume_safe"], "true")
        self.assertEqual(recovery("working", resume_safe=False)["resume_safe"], "false")
        self.assertEqual(recovery("blocked", resume_safe=True)["next_action"], "resolve_blocker")
        self.assertEqual(recovery("review", resume_safe=False)["status"], "awaiting_review")

    def test_unknown_rungs_recover_nothing(self) -> None:
        for name in ("", "merged", "sentient"):
            with self.subTest(name=name):
                row = recovery(name, resume_safe=True)
                self.assertEqual(row["status"], "recovery_blocked")
                self.assertEqual(row["next_action"], "repair_wrapper_records")
                self.assertEqual(row["resume_safe"], "true")


class AutonomyTests(WrapperHomeCase):
    def test_first_use_is_guided(self) -> None:
        self.drive("s-new", "plan_presented")
        row = autonomy("s-new")
        self.assertEqual(row["mode"], "guided_first_use")
        self.assertEqual(row["accepted_count"], 0)
        self.assertEqual(row["threshold"], 2)
        self.assertEqual(row["next_action"], "explain_before_next_step")
        self.assertEqual(row["record_errors"], 0)
        self.assertTrue(row["confirmation"]["required"])
        self.assertIn("Approvals still gate", row["confirmation"]["boundary"])
        self.assertIn("recorded accept decisions", row["claim_boundary"])

    def test_accepted_patterns_read_back_from_the_records(self) -> None:
        for name in ("s-a", "s-b"):
            self.drive(name, "plan_accepted", skill="porting")
        row = autonomy("s-a")
        self.assertEqual(row["mode"], "autonomous_pattern")
        self.assertEqual(row["skill"], "porting")
        self.assertEqual(row["accepted_count"], 2)
        self.assertEqual(row["next_action"], "continue_recorded_pattern")
        self.assertEqual(row["confirmation"]["policy"], "pattern_based_autonomy")
        self.assertGreater(row["observed_events"], 0)

    def test_a_different_skill_is_not_an_accepted_pattern(self) -> None:
        self.drive("s-a", "plan_accepted", skill="one")
        self.drive("s-b", "plan_accepted", skill="two")
        self.assertEqual(autonomy("s-a", threshold=2)["mode"], "guided_first_use")
        self.assertEqual(autonomy("s-a", threshold=1)["mode"], "autonomous_pattern")

    def test_without_a_skill_only_this_session_counts(self) -> None:
        self.drive("s-a", "plan_accepted")
        self.drive("s-b", "plan_accepted")
        self.assertEqual(autonomy("s-a")["accepted_count"], 1)
        self.assertEqual(autonomy("s-a")["mode"], "guided_first_use")

    def test_unusable_identifiers_stay_guided(self) -> None:
        self.drive("s-a", "plan_accepted")
        for bad in ("../escape", "", "with space", "a" * 300, 7):
            with self.subTest(bad=bad):
                row = autonomy(bad)  # type: ignore[arg-type]
                self.assertEqual(row["mode"], "guided_first_use")
                self.assertEqual(row["session_id"], "")
                self.assertGreaterEqual(row["record_errors"], 1)
                self.assertEqual(row["observed_events"], 0)

    def test_thresholds_are_clamped(self) -> None:
        self.drive("s-a", "plan_accepted")
        for bad in (0, -3, "two", True, None, 10_000):
            with self.subTest(bad=bad):
                self.assertGreaterEqual(autonomy("s-a", threshold=bad)["threshold"], 1)  # type: ignore[arg-type]
        self.assertEqual(autonomy("s-a", threshold=10_000)["threshold"], 50)


class BuildMissionTests(WrapperHomeCase):
    def test_aggregates_only_persisted_records(self) -> None:
        self.drive("s-1", "working")
        record_evidence(
            "s-1",
            EvidenceItem(kind="command", subject="pytest -q", digest="sha256:aa", result="observed", exit_code=0, at=stamp()),
        )
        store = SessionStore(self.base / "sessions")
        session = store.create(str(self.workspace), "openai/gpt-test", "main")
        session.messages.append({"role": "user", "content": "carry on"})
        store.save(session)

        payload = build_mission("s-1", store=store, now=time.time())
        self.assertEqual(
            set(payload),
            {
                "schema_version",
                "session_id",
                "journey",
                "signal",
                "status",
                "title",
                "workspace",
                "model",
                "lane",
                "updated_at",
                "age_seconds",
                "stale",
                "terminal",
                "broken_link",
                "untracked",
                "record_errors",
                "claims",
                "continuity",
                "recovery",
                "autonomy",
                "headline",
                "next_action",
                "claim_boundary",
            },
        )
        self.assertEqual(payload["session_id"], "s-1")
        self.assertEqual(payload["journey"], {"state": "working"})
        self.assertEqual(payload["signal"], "running")
        self.assertEqual(payload["workspace"], str(self.workspace))
        self.assertEqual(payload["claims"]["allowed_claims"], ["commands_ran", "tests_pass"])
        self.assertIn("work_verified", payload["claims"]["blocked_claims"])
        self.assertEqual(payload["continuity"]["resume_status"], "conversation_safe")
        self.assertEqual(payload["continuity"]["workspace_drift"], "same")
        self.assertEqual(payload["recovery"]["status"], "running_observed")
        self.assertEqual(payload["recovery"]["resume_safe"], "true")
        self.assertEqual(payload["autonomy"]["mode"], "guided_first_use")
        self.assertIn("in progress", payload["headline"].lower())
        self.assertEqual(payload["next_action"], "record_observed_evidence")
        self.assertIn("read-only local projection", payload["claim_boundary"])
        self.assertFalse(payload["terminal"])
        self.assertFalse(payload["stale"])

    def test_a_vanished_workspace_is_reported_not_assumed(self) -> None:
        plant("s-2", status="working", workspace=str(self.base / "vanished"), updated_at=stamp())
        payload = build_mission("s-2", now=time.time())
        self.assertEqual(payload["continuity"]["workspace_drift"], "missing")
        self.assertEqual(payload["continuity"]["next_action"], "relocate_workspace")

    def test_staleness_and_terminality_come_from_the_timestamp(self) -> None:
        plant("s-3", status="working", updated_at=stamp(-STALE_SECONDS))
        payload = build_mission("s-3", now=time.time())
        self.assertTrue(payload["stale"])
        self.assertGreater(payload["age_seconds"], DEFAULT_STALE_MINUTES * 60)
        self.assertFalse(payload["terminal"])

        plant("s-4", status="cancelled", updated_at=stamp())
        closed = build_mission("s-4", now=time.time())
        self.assertTrue(closed["terminal"])
        self.assertEqual(closed["journey"], {"state": "cancelled"})
        self.assertEqual(closed["signal"], "stopped")
        self.assertEqual(closed["continuity"]["resume_status"], "blocked")
        self.assertEqual(closed["recovery"]["status"], "cancelled_observed")
        self.assertEqual(closed["recovery"]["resume_safe"], "false")
        self.assertFalse(closed["stale"])

    def test_unrecorded_sessions_claim_nothing(self) -> None:
        payload = build_mission("s-ghost", now=time.time())
        self.assertEqual(payload["journey"], {"state": "broken_link"})
        self.assertEqual(payload["signal"], "unknown")
        self.assertEqual(payload["status"], "unrecorded")
        self.assertTrue(payload["untracked"])
        self.assertTrue(payload["broken_link"])
        self.assertEqual(payload["headline"], "No wrapper state is recorded for this session.")
        self.assertEqual(payload["next_action"], "repair_wrapper_records")
        self.assertEqual(payload["claims"]["allowed_claims"], [])
        self.assertEqual(payload["continuity"]["workspace_drift"], "unknown")
        self.assertIsNone(payload["age_seconds"])

    def test_hostile_identifiers_are_refused(self) -> None:
        for bad in ("", "../escape", "with space", None, 7):
            with self.subTest(bad=bad):
                row = build_mission(bad, now=time.time())  # type: ignore[arg-type]
                self.assertEqual(row["journey"], {"state": "broken_link"})
                self.assertEqual(row["session_id"], "")
                self.assertTrue(row["broken_link"])

    def test_mission_reading_never_writes_a_record(self) -> None:
        self.drive("s-5", "working")
        store = SessionStore(self.base / "sessions")
        session = store.create(str(self.workspace), "openai/gpt-test", "main")
        session.messages.append({"role": "user", "content": "hi"})
        store.save(session)
        before = _files(self.home)
        self.assertTrue(before)
        build_mission("s-5", store=store)
        build_mission(session.meta.id, store=store)
        list_missions(store)
        autonomy("s-5")
        self.assertEqual(_files(self.home), before)


class ListMissionsTests(WrapperHomeCase):
    def test_rows_join_states_to_metas(self) -> None:
        store = SessionStore(self.base / "sessions")
        tracked = store.create(str(self.workspace), "openai/gpt-test", "main")
        store.save(tracked)
        self.drive(tracked.meta.id, "working")
        plain = store.create(str(self.workspace), "openai/gpt-test", "main")
        plain.messages.append({"role": "user", "content": "just chatting"})
        store.save(plain)

        rows = {row["session_id"]: row for row in list_missions(store)}
        self.assertEqual(set(rows), {tracked.meta.id, plain.meta.id})
        self.assertEqual(rows[tracked.meta.id]["journey"], "working")
        self.assertEqual(rows[tracked.meta.id]["signal"], "running")
        self.assertFalse(rows[tracked.meta.id]["untracked"])
        self.assertEqual(rows[plain.meta.id]["journey"], "broken_link")
        self.assertTrue(rows[plain.meta.id]["untracked"])
        self.assertEqual(rows[plain.meta.id]["status"], "unrecorded")
        self.assertEqual(rows[plain.meta.id]["next_action"], "repair_wrapper_records")
        self.assertIn("read-only local projection", rows[plain.meta.id]["claim_boundary"])

    def test_old_and_terminal_sessions_are_flagged(self) -> None:
        store = SessionStore(self.base / "sessions")
        old = store.create(str(self.workspace), "openai/gpt-test", "main")
        store.save(old)
        plant(old.meta.id, status="working", updated_at=stamp(-STALE_SECONDS))
        closed = store.create(str(self.workspace), "openai/gpt-test", "main")
        store.save(closed)
        plant(closed.meta.id, status="cancelled", updated_at=stamp())

        rows = {row["session_id"]: row for row in list_missions(store)}
        self.assertTrue(rows[old.meta.id]["stale"])
        self.assertFalse(rows[old.meta.id]["terminal"])
        self.assertEqual(rows[old.meta.id]["journey"], "working")
        self.assertTrue(rows[closed.meta.id]["terminal"])
        self.assertFalse(rows[closed.meta.id]["stale"])

    def test_limits_are_bounded_and_bad_stores_are_ignored(self) -> None:
        store = SessionStore(self.base / "sessions")
        for _ in range(4):
            store.save(store.create(str(self.workspace), "openai/gpt-test", "main"))
        self.assertEqual(len(list_missions(store, limit=2)), 2)
        self.assertLessEqual(len(list_missions(store, limit=0)), 4)
        self.assertLessEqual(len(list_missions(store, limit=10_000)), 100)
        for bad in (None, "store", object()):
            with self.subTest(store=bad):
                self.assertEqual(list_missions(bad), [])  # type: ignore[arg-type]


class MissionRowsTests(unittest.TestCase):
    def test_columns_and_flags(self) -> None:
        moment = datetime.now(timezone.utc).timestamp()
        rows = [
            {
                "session_id": "s-old",
                "signal": "waiting",
                "journey": "drafted",
                "status": "plan_presented",
                "updated_at": datetime.fromtimestamp(moment - 3900, timezone.utc).isoformat(timespec="seconds"),
                "title": "Port mission control",
            },
            {
                "session_id": "s-now",
                "signal": "complete",
                "journey": {"state": "finished"},
                "status": "done",
                "updated_at": datetime.fromtimestamp(moment, timezone.utc).isoformat(timespec="seconds"),
                "terminal": True,
                "headline": "Work is finished; nothing is outstanding.",
            },
            {"session_id": "s-unknown", "signal": "nonsense", "age_seconds": 60},
            "not-a-mapping",
        ]
        table = mission_rows(rows, now=moment)
        self.assertEqual(len(table), 3)
        self.assertTrue(all(len(row) == 7 for row in table))
        self.assertEqual(table[0][:2], ["s-old", "🟡"])
        self.assertEqual(table[0][2:5], ["drafted", "plan_presented", "1h"])
        self.assertEqual(table[0][5], "stale")
        self.assertEqual(table[0][6], "Port mission control")
        self.assertEqual(table[1][2:6], ["finished", "done", "now", "terminal"])
        self.assertEqual(table[1][6], "Work is finished; nothing is outstanding.")
        self.assertEqual(table[2][2], "broken_link")
        self.assertEqual(table[2][4], "1m")

    def test_age_and_timestamp_fallbacks(self) -> None:
        moment = datetime.now(timezone.utc).timestamp()
        recorded = mission_rows([{"session_id": "s", "updated_at": "not-a-timestamp", "age_seconds": 90000}], now=moment)
        self.assertEqual(recorded[0][4], "1d")
        missing = mission_rows([{"session_id": "s"}], now=moment)
        self.assertEqual(missing[0][4], "unknown")
        self.assertEqual(missing[0][2], "broken_link")
        self.assertEqual(missing[0][5], "broken_link")
        self.assertEqual(mission_rows([], now=moment), [])
        self.assertEqual(mission_rows([{"session_id": "\x1b[31mghost\x00"}], now=moment)[0][0], "ghost")
        self.assertEqual(mission_rows([{"session_id": "s"}])[0][4], "unknown")


class RenderMissionTests(WrapperHomeCase):
    def payload(self) -> dict[str, Any]:
        cached = getattr(self, "_payload", None)
        if cached is not None:
            return cached
        self.drive("s-1", "working")
        record_evidence(
            "s-1",
            EvidenceItem(kind="command", subject="pytest -q", digest="sha256:aa", result="observed", exit_code=0, at=stamp()),
        )
        self._payload = build_mission("s-1", now=time.time())
        return self._payload

    def test_every_line_fits_the_width(self) -> None:
        plain = make_theme("aurora", color=False)
        payload = self.payload()
        for width in (24, 40, 64, 100):
            for glyphs in ("unicode", "ascii"):
                rendered = render_mission(payload, plain, width=width, glyphs=glyphs)
                with self.subTest(width=width, glyphs=glyphs):
                    for line in rendered.splitlines():
                        self.assertLessEqual(display_width(strip_ansi(line)), width)

    def test_content_is_drawn_from_the_payload(self) -> None:
        rendered = render_mission(self.payload(), make_theme("mono", color=False), width=96)
        self.assertNotIn("\x1b", rendered)
        self.assertIn("Mission · s-1", rendered)
        self.assertIn("working", rendered)
        self.assertIn("running_observed", rendered)
        self.assertIn("2 allowed / 4 blocked", rendered)
        self.assertIn("guided_first_use", rendered)
        self.assertIn("hint ·", rendered)
        self.assertIn("read-only local projection", rendered.replace("\n", " "))
        colored = render_mission(self.payload(), make_theme("aurora", color=True, depth="truecolor"), width=96)
        self.assertIn("\x1b[38;2;", colored)

    def test_broken_and_empty_payloads(self) -> None:
        plain = make_theme("mono", color=False)
        broken = render_mission(
            {"session_id": "s", "journey": {"state": "broken_link"}, "recovery": recovery("broken_link", resume_safe=False)},
            plain,
            width=60,
            glyphs="ascii",
        )
        self.assertIn("x Mission - s", broken)
        self.assertTrue(broken.startswith("x "))
        self.assertIn("repair_wrapper_records", broken)
        self.assertIn("broken_link", render_mission({}, plain, width=60))
        self.assertTrue(render_mission(None, plain, width=60).splitlines())  # type: ignore[arg-type]

    def test_degenerate_widths_still_render(self) -> None:
        plain = make_theme("mono", color=False)
        payload = self.payload()
        for width in (0, -8, "wide", None):
            with self.subTest(width=width):
                self.assertTrue(render_mission(payload, plain, width=width).splitlines())  # type: ignore[arg-type]


def _files(root: Path) -> set[str]:
    return {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}


if __name__ == "__main__":
    unittest.main()
