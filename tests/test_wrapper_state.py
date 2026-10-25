from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from ohar.wrapper import records, state


class StateMixin(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.previous_home = os.environ.get("OHAR_HOME")
        os.environ["OHAR_HOME"] = str(Path(self.tmp.name) / "home")

    def tearDown(self) -> None:
        if self.previous_home is None:
            os.environ.pop("OHAR_HOME", None)
        else:
            os.environ["OHAR_HOME"] = self.previous_home
        self.tmp.cleanup()

    def open(self, session_id: str = "s1") -> state.SessionState:
        opened, resumed = state.open_state(session_id, goal="port the wrapper", workspace="/work")
        self.assertFalse(resumed)
        return opened

    def events(self, session_id: str = "s1") -> list[str]:
        items, errors = records.read_events(session_id, limit=records.MAX_EVENTS)
        self.assertEqual(errors, [])
        return [str(item["event"]) for item in items]


class OpenStateTests(StateMixin):
    def test_open_creates_a_gated_record_and_event(self) -> None:
        opened = self.open()
        self.assertEqual(opened.status, "open")
        self.assertEqual(opened.revision, 1)
        self.assertEqual(opened.next_action, state.NEXT_ACTIONS["open"])
        self.assertEqual(opened.decision, "")
        self.assertFalse(opened.terminal)
        self.assertEqual(self.events(), ["session_started"])
        self.assertEqual(state.load_state("s1"), opened)

    def test_goal_and_workspace_are_cleaned_and_bounded(self) -> None:
        opened, _ = state.open_state(
            "s2",
            goal="\x1b[31m" + "goal " * 200 + "\nsecond line\x07",
            workspace="/tmp\x00/a",
            model="qwen3:8b",
            lane="main",
        )
        self.assertEqual(len(opened.goal), state.MAX_GOAL_CHARS)
        self.assertNotIn("\x1b", opened.goal)
        self.assertNotIn("\n", opened.goal)
        self.assertEqual(opened.workspace, "/tmp/a")
        self.assertEqual((opened.model, opened.lane), ("qwen3:8b", "main"))

    def test_resume_returns_the_existing_record_untouched(self) -> None:
        opened = self.open()
        state.transition("s1", "plan_presented")
        resumed_state, resumed = state.open_state("s1", goal="other", workspace="other")
        self.assertTrue(resumed)
        self.assertEqual(resumed_state.status, "plan_presented")
        self.assertEqual(resumed_state.revision, opened.revision + 1)
        self.assertNotEqual(resumed_state.goal, "other")
        self.assertEqual(self.events(), ["session_started", "status_plan_presented", "session_resumed"])

    def test_invalid_session_id_is_refused(self) -> None:
        for bad in ("../escape", "", "a/b"):
            with self.subTest(bad=bad):
                with self.assertRaises(state.SessionStateError):
                    state.open_state(bad, goal="g", workspace="/w")

    def test_load_state_returns_none_when_absent_or_corrupt(self) -> None:
        self.assertIsNone(state.load_state("never"))
        self.open()
        (records.session_dir("s1") / state.STATE_RECORD).write_text("{oops", encoding="utf-8")
        self.assertIsNone(state.load_state("s1"))
        self.assertEqual(records.read_record(state.STATE_RECORD, "s1"), {})


class TransitionTests(StateMixin):
    def test_the_happy_path_walks_the_whole_vocabulary(self) -> None:
        self.open()
        walk = [
            "plan_presented",
            "plan_accepted",
            "working",
            "blocked",
            "working",
            "needs_review",
            "verified",
            "done",
        ]
        current = None
        for status in walk:
            current = state.transition("s1", status)
            self.assertEqual(current.status, status)
            self.assertEqual(current.next_action, state.NEXT_ACTIONS[status])
        self.assertTrue(current.terminal)
        self.assertEqual(current.revision, len(walk) + 1)
        self.assertEqual(self.events(), ["session_started", *[f"status_{status}" for status in walk]])

    def test_illegal_jumps_are_refused_with_the_allowed_set(self) -> None:
        self.open()
        with self.assertRaises(state.SessionStateError) as caught:
            state.transition("s1", "working")
        message = str(caught.exception)
        self.assertIn("open -> working", message)
        self.assertIn("plan_presented", message)
        state.transition("s1", "plan_presented")
        with self.assertRaises(state.SessionStateError):
            state.transition("s1", "verified")
        with self.assertRaises(state.SessionStateError):
            state.transition("s1", "plan_presented")

    def test_unknown_status_and_fields_are_refused(self) -> None:
        self.open()
        for bad in ("", "PLAN_PRESENTED", "done ", None, 7):
            with self.subTest(bad=bad):
                with self.assertRaises(state.SessionStateError):
                    state.transition("s1", bad)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            state.transition("s1", "plan_presented", status="working")  # type: ignore[misc]
        with self.assertRaises(state.SessionStateError):
            state.transition("s1", "plan_presented", bogus="x")
        with self.assertRaises(state.SessionStateError):
            state.transition("s1", "plan_presented", goal=object())  # type: ignore[arg-type]

    def test_transition_carries_explicit_fields(self) -> None:
        self.open()
        moved = state.transition("s1", "plan_presented", skill="code-review", next_action="review_plan")
        self.assertEqual(moved.skill, "code-review")
        self.assertEqual(moved.next_action, "review_plan")
        self.assertEqual(moved.goal, "port the wrapper")
        self.assertEqual(moved.revision, 2)

    def test_transition_reuses_replayed_mutation_ids(self) -> None:
        self.open()
        first = state.transition("s1", "plan_presented", mutation_id="click-1")
        again = state.transition("s1", "plan_presented", mutation_id="click-1")
        self.assertEqual(again.revision, first.revision)
        self.assertEqual(self.events(), ["session_started", "status_plan_presented"])

    def test_stale_expected_revision_is_refused(self) -> None:
        self.open()
        with self.assertRaises(state.SessionStateError):
            state.transition("s1", "plan_presented", expected_revision=42)
        current = state.transition("s1", "plan_presented", expected_revision=1)
        self.assertEqual(current.revision, 2)

    def test_transition_requires_an_opened_state(self) -> None:
        with self.assertRaises(state.SessionStateError):
            state.transition("ghost", "plan_presented")


class DecisionTests(StateMixin):
    def test_accept_moves_the_plan_forward(self) -> None:
        self.open()
        state.transition("s1", "plan_presented")
        accepted = state.record_decision("s1", "accept")
        self.assertEqual(accepted.status, "plan_accepted")
        self.assertEqual(accepted.decision, "accept")
        self.assertEqual(accepted.next_action, state.NEXT_ACTIONS["plan_accepted"])
        self.assertEqual(self.events(), ["session_started", "status_plan_presented", "decision_accept"])

    def test_revise_and_reaccept_round_trip(self) -> None:
        self.open()
        state.transition("s1", "plan_presented")
        revised = state.record_decision("s1", "revise")
        self.assertEqual((revised.status, revised.decision), ("revision_requested", "revise"))
        with self.assertRaises(state.SessionStateError):
            state.record_decision("s1", "revise")
        accepted = state.record_decision("s1", "accept")
        self.assertEqual(accepted.status, "plan_accepted")

    def test_cancel_from_any_open_status_is_terminal(self) -> None:
        for status in ("open", "plan_presented", "working"):
            session_id = f"cancel-{status}"
            state.open_state(session_id, goal="g", workspace="/w")
            if status != "open":
                state.transition(session_id, "plan_presented")
            if status == "working":
                state.record_decision(session_id, "accept")
                state.transition(session_id, "working")
            closed = state.record_decision(session_id, "cancel")
            self.assertEqual(closed.status, "cancelled")
            self.assertTrue(closed.terminal)
            with self.assertRaises(state.SessionStateError):
                state.record_decision(session_id, "cancel")

    def test_decisions_outside_the_gate_are_refused(self) -> None:
        self.open()
        state.transition("s1", "plan_presented")
        state.record_decision("s1", "accept")
        with self.assertRaises(state.SessionStateError):
            state.record_decision("s1", "accept")
        with self.assertRaises(state.SessionStateError):
            state.record_decision("s1", "maybe")
        state.transition("s1", "working")
        with self.assertRaises(state.SessionStateError) as caught:
            state.record_decision("s1", "revise")
        self.assertIn("while status is working", str(caught.exception))
        self.assertEqual(self.events(), ["session_started", "status_plan_presented", "decision_accept", "status_working"])

    def test_replayed_decision_appends_one_event(self) -> None:
        self.open()
        state.transition("s1", "plan_presented")
        first = state.record_decision("s1", "accept", mutation_id="tap-accept")
        again = state.record_decision("s1", "accept", mutation_id="tap-accept")
        self.assertEqual(again.revision, first.revision)
        self.assertEqual(self.events().count("decision_accept"), 1)


class NoteAndSaveTests(StateMixin):
    def test_note_refreshes_hints_without_a_status_jump(self) -> None:
        self.open()
        noted = state.note("s1", model="qwen3:8b", skill="refactor", next_action="present_plan")
        self.assertEqual(noted.status, "open")
        self.assertEqual(noted.model, "qwen3:8b")
        self.assertEqual(noted.next_action, "present_plan")
        self.assertEqual(noted.revision, 2)
        self.assertEqual(self.events(), ["session_started"])
        self.assertEqual(state.next_action_for(noted), "present_plan")

    def test_note_requires_a_value_and_refuses_terminal_states(self) -> None:
        self.open()
        with self.assertRaises(state.SessionStateError):
            state.note("s1")
        with self.assertRaises(TypeError):
            state.note("s1", status="done")  # type: ignore[call-arg]
        with self.assertRaises(state.SessionStateError):
            state.note("never-opened", skill="x")
        state.transition("s1", "plan_presented")
        state.record_decision("s1", "cancel")
        with self.assertRaises(state.SessionStateError):
            state.note("s1", lane="main")

    def test_save_state_replaces_wholesale_and_gates_revisions(self) -> None:
        opened = self.open()
        saved = state.save_state(opened, mutation_id="save-1")
        self.assertEqual(saved.revision, 2)
        replayed = state.save_state(opened, mutation_id="save-1")
        self.assertEqual(replayed.revision, 2)
        with self.assertRaises(state.SessionStateError):
            state.save_state(replayed, expected_revision=9)
        with self.assertRaises(state.SessionStateError):
            state.save_state("not a state")  # type: ignore[arg-type]

    def test_save_state_creates_when_absent(self) -> None:
        draft = state.SessionState(session_id="fresh", status="open", goal="g", workspace="/w")
        saved = state.save_state(draft)
        self.assertEqual(saved.revision, 1)
        self.assertEqual(saved.updated_at, records.read_record(state.STATE_RECORD, "fresh")["updated_at"])


class CloseTests(StateMixin):
    def walk(self, session_id: str = "s1") -> None:
        state.open_state(session_id, goal="g", workspace="/w")
        state.transition(session_id, "plan_presented")
        state.record_decision(session_id, "accept")
        state.transition(session_id, "working")
        state.transition(session_id, "verified")

    def test_close_state_finishes_a_verified_journey(self) -> None:
        self.walk()
        closed = state.close_state("s1")
        self.assertEqual(closed.status, "done")
        self.assertTrue(closed.terminal)
        self.assertEqual(self.events()[-1], "status_done")

    def test_close_state_is_idempotent_on_a_terminal_record(self) -> None:
        self.walk()
        closed = state.close_state("s1", "cancelled")
        again = state.close_state("s1")
        self.assertEqual(again.status, "cancelled")
        self.assertEqual(again.revision, closed.revision)

    def test_close_state_refuses_non_terminal_and_missing_records(self) -> None:
        self.walk()
        with self.assertRaises(state.SessionStateError):
            state.close_state("s1", "working")
        with self.assertRaises(state.SessionStateError):
            state.close_state("nobody")

    def test_no_write_touches_a_terminal_state(self) -> None:
        self.walk()
        state.close_state("s1")
        for call in (
            lambda: state.transition("s1", "working"),
            lambda: state.record_decision("s1", "cancel"),
            lambda: state.note("s1", skill="nope"),
            lambda: state.save_state(state.SessionState(session_id="s1", status="working", updated_at="x")),
        ):
            with self.assertRaises(state.SessionStateError):
                call()
        self.assertEqual(state.load_state("s1").status, "done")


class SessionStateModelTests(StateMixin):
    def test_round_trip_through_dict(self) -> None:
        opened = self.open()
        restored = state.SessionState.from_dict(opened.to_dict())
        self.assertEqual(restored, opened)
        payload = opened.to_dict()
        self.assertEqual(payload["record_type"], state.RECORD_TYPE)
        self.assertEqual(payload["schema_version"], records.SCHEMA_VERSION)
        self.assertEqual(payload["record_revision"], opened.revision)

    def test_from_dict_is_fail_closed(self) -> None:
        good = self.open().to_dict()
        cases = {
            "status": {**good, "status": "nonsense"},
            "decision": {**good, "decision": "maybe"},
            "goal_type": {**good, "goal": 3},
            "goal_too_long": {**good, "goal": "x" * (state.MAX_GOAL_CHARS + 1)},
            "session_id": {**good, "session_id": "../sneaky"},
            "revision": {**good, "record_revision": -2},
            "updated_at": {**good, "updated_at": "yesterday"},
            "schema": {**good, "schema_version": 9},
            "record_type": {**good, "record_type": "other"},
        }
        for label, payload in cases.items():
            with self.subTest(label):
                with self.assertRaises(state.SessionStateError):
                    state.SessionState.from_dict(payload)
        for not_a_mapping in (None, [], "state"):
            with self.assertRaises(state.SessionStateError):
                state.SessionState.from_dict(not_a_mapping)  # type: ignore[arg-type]

    def test_from_dict_accepts_missing_optional_strings(self) -> None:
        payload = self.open().to_dict()
        for key in ("decision", "goal", "workspace", "model", "lane", "skill", "next_action"):
            payload.pop(key)
        restored = state.SessionState.from_dict(payload)
        self.assertEqual(restored.decision, "")
        self.assertEqual(restored.goal, "")

    def test_terminal_property_matches_the_vocabulary(self) -> None:
        for status in state.TERMINAL_STATUSES:
            self.assertTrue(state.SessionState(session_id="s", status=status).terminal)
        for status in state.STATUSES:
            if status not in state.TERMINAL_STATUSES:
                self.assertFalse(state.SessionState(session_id="s", status=status).terminal)

    def test_vocabulary_tables_are_closed(self) -> None:
        self.assertEqual(set(state.TRANSITIONS), set(state.STATUSES))
        self.assertEqual(set(state.DECISION_GATES), set(state.STATUSES))
        for status, allowed in state.TRANSITIONS.items():
            for target in allowed:
                self.assertIn(target, state.STATUSES)
            if status in state.TERMINAL_STATUSES:
                self.assertEqual(allowed, frozenset())
            else:
                self.assertTrue(allowed, f"{status} can never go anywhere")
        self.assertEqual(len(state.STATUSES), 10)
        self.assertEqual(state.DECISIONS, ("accept", "revise", "cancel"))
        for decision, target in state.DECISION_TARGETS.items():
            self.assertIn(target, state.STATUSES)
            for status, gated in state.DECISION_GATES.items():
                if decision in gated:
                    self.assertIn(target, state.TRANSITIONS[status])

    def test_next_action_for_falls_back_to_the_status(self) -> None:
        self.assertEqual(state.next_action_for(state.SessionState(session_id="s", status="working")), "record_observed_evidence")
        self.assertEqual(state.next_action_for(state.SessionState(session_id="s", status="working", next_action="run_tests")), "run_tests")
        self.assertEqual(state.next_action_for(state.SessionState(session_id="s", status="nonsense", next_action="   ")), "show_status")


class ListStatesTests(StateMixin):
    def test_lists_newest_first_and_skips_unusable_dirs(self) -> None:
        self.assertEqual(state.list_states(), [])
        for index in range(3):
            session_id = f"session-{index}"
            state.open_state(session_id, goal=f"g{index}", workspace="/w")
        state.transition("session-2", "plan_presented")
        (records.session_dir("session-1") / state.STATE_RECORD).write_text("{ broken", encoding="utf-8")
        (records.wrapper_root() / "not a session").mkdir(exist_ok=True)
        (records.wrapper_root() / "stray.txt").write_text("x", encoding="utf-8")

        found = state.list_states()
        self.assertEqual([item.session_id for item in found], ["session-0", "session-2"])
        # Same-second timestamps make natural ordering a coin flip; pin one record.
        pinned = records.read_record(state.STATE_RECORD, "session-2")
        pinned["updated_at"] = "2099-01-01T00:00:00+00:00"
        records.write_record(state.STATE_RECORD, "session-2", pinned)

        found = state.list_states()
        self.assertEqual([item.session_id for item in found], ["session-2", "session-0"])
        self.assertEqual(found[0].status, "plan_presented")
        self.assertEqual([item.session_id for item in state.list_states(limit=1)], ["session-2"])
        for bad in (0, -1, "many", None):
            self.assertGreaterEqual(len(state.list_states(limit=bad)), 1)  # type: ignore[arg-type]

    def test_list_does_not_create_the_wrapper_root(self) -> None:
        root = records.wrapper_root(create=False)
        self.assertEqual(state.list_states(), [])
        self.assertFalse(root.exists())


if __name__ == "__main__":
    unittest.main()
