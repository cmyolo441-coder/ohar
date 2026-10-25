from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ohar.skills import SkillLibrary
from ohar.wrapper import routehints
from ohar.wrapper.records import append_event, read_events
from ohar.wrapper.state import close_state, load_state, open_state


class _Home(unittest.TestCase):
    """Every wrapper test runs against an OHAR_HOME inside a temp directory."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.home = self.base / "home"
        self.skills_root = self.base / "skills"
        self.skills_root.mkdir(parents=True)
        self._env = mock.patch.dict(os.environ, {"OHAR_HOME": str(self.home)})
        self._env.start()
        self.addCleanup(self._env.stop)
        self.addCleanup(self.tmp.cleanup)

    def add_skill(
        self,
        name: str,
        description: str,
        *,
        aliases: tuple[str, ...] = (),
        body: str = "Do the thing.",
    ) -> None:
        directory = self.skills_root / name
        directory.mkdir(parents=True, exist_ok=True)
        front = [f"name: {name}", f"description: {description}"]
        if aliases:
            front.append("aliases: [" + ", ".join(aliases) + "]")
        text = "---\n" + "\n".join(front) + "\n---\n\n" + body + "\n"
        (directory / "SKILL.md").write_text(text, encoding="utf-8")

    def library(self) -> SkillLibrary:
        return SkillLibrary.discover(self.skills_root)


class HintTests(_Home):
    def test_named_skill_is_ranked_first_and_carries_the_next_action(self) -> None:
        self.add_skill("code-review", "Review a diff for defects")
        self.add_skill("changelog", "Summarize released commits")
        hints = routehints.hint_for(
            "run code-review over this diff", self.library(), limit=2
        )
        self.assertTrue(hints)
        top = hints[0]
        self.assertEqual(top.skill, "code-review")
        self.assertEqual(top.label, "Code Review")
        self.assertEqual(top.next_action, "load_skill")
        self.assertEqual(top.submit_text, "/skill code-review")
        self.assertIn("named in the request", top.reason)
        self.assertGreaterEqual(top.score, routehints.MIN_HINT_SCORE)
        self.assertIn(top.confidence, ("high", "medium", "low"))

    def test_history_weight_can_flip_the_order_and_is_bounded(self) -> None:
        self.add_skill("alpha-one", "shared description words")
        self.add_skill("beta-two", "shared description words")
        library = self.library()
        request = "alpha-one beta-two"

        plain = routehints.hint_for(request, library, limit=2)
        self.assertEqual([hint.skill for hint in plain], ["alpha-one", "beta-two"])
        boosted = routehints.hint_for(
            request, library, limit=2, history={"beta-two": 3}
        )
        self.assertEqual([hint.skill for hint in boosted], ["beta-two", "alpha-one"])
        self.assertAlmostEqual(
            boosted[0].score - plain[1].score, routehints.HISTORY_WEIGHT, places=3
        )
        self.assertIn("your usual pick (3 accepted)", boosted[0].reason)
        self.assertTrue(all(hint.score <= 1.0 for hint in boosted))

    def test_single_accepted_pick_is_reported_as_a_single_pick(self) -> None:
        self.add_skill("alpha-one", "shared description words")
        self.add_skill("beta-two", "shared description words")
        hints = routehints.hint_for(
            "alpha-one beta-two", self.library(), limit=2, history={"beta-two": 1}
        )
        reasons = {hint.skill: hint.reason for hint in hints}
        self.assertIn("1 accepted pick", reasons["beta-two"])
        self.assertNotIn("accepted", reasons["alpha-one"])

    def test_weak_matches_below_the_floor_are_dropped(self) -> None:
        self.add_skill("zeta", "quantum tunneling spectroscopy")
        self.add_skill("omega", "brick oven proofing schedule")
        hints = routehints.hint_for(
            "quantum spectroscopy and oven proofing notes", self.library(), limit=3
        )
        self.assertTrue(all(hint.score >= routehints.MIN_HINT_SCORE for hint in hints))
        self.assertLessEqual(len(hints), routehints.MAX_HINTS)

    def test_limit_and_empty_input_are_bounded(self) -> None:
        self.add_skill("alpha-one", "shared description words")
        self.add_skill("beta-two", "shared description words")
        self.add_skill("gamma-three", "shared description words")
        library = self.library()
        self.assertEqual(routehints.hint_for("", library), [])
        self.assertEqual(routehints.hint_for("   ", library), [])
        self.assertLessEqual(
            len(routehints.hint_for("alpha-one beta-two gamma-three", library, limit=0)),
            routehints.MAX_HINTS,
        )
        self.assertLessEqual(
            len(routehints.hint_for("alpha-one beta-two gamma-three", library, limit=99)),
            routehints.MAX_HINTS,
        )
        self.assertLessEqual(
            len(routehints.hint_for("alpha-one beta-two", library, limit="bad")),
            routehints.MAX_HINTS,
        )

    def test_adjacent_names_exclude_the_hint_itself(self) -> None:
        self.add_skill("alpha-one", "shared description words")
        self.add_skill("beta-two", "shared description words")
        self.add_skill("gamma-three", "shared description words")
        hints = routehints.hint_for(
            "alpha-one beta-two gamma-three", self.library(), limit=1
        )
        self.assertEqual(len(hints), 1)
        self.assertNotIn(hints[0].skill, hints[0].adjacent)
        self.assertEqual(set(hints[0].adjacent), {"beta-two", "gamma-three"})
        self.assertLessEqual(len(hints[0].adjacent), routehints.MAX_ADJACENT + 1)

    def test_catalog_question_is_answered_from_the_skill_index(self) -> None:
        self.add_skill("alpha-one", "shared description words")
        self.add_skill("beta-two", "shared description words")
        hints = routehints.hint_for("which skills do you have?", self.library(), limit=2)
        self.assertTrue(hints)
        self.assertEqual([hint.skill for hint in hints], ["alpha-one", "beta-two"])
        for hint in hints:
            self.assertEqual(hint.next_action, "show_skills")
            self.assertEqual(hint.confidence, "high")
            self.assertEqual(hint.submit_text, f"/skill {hint.skill}")
        self.assertEqual(hints[0].adjacent, ("beta-two",))
        self.assertEqual(hints[1].adjacent, ("alpha-one",))
        self.assertIn("listed from the installed skill index", hints[0].reason)


class AnswersDirectlyTests(_Home):
    def test_greetings_thanks_catalog_and_short_turns_answer_directly(self) -> None:
        for text in ("hi", "Hello!", "hey there", "thanks a lot", "thank you", " Ty ", "h", "x"):
            self.assertTrue(routehints.answers_directly(text, []), text)

    def test_catalog_questions_answer_directly(self) -> None:
        for text in (
            "which skills do you have?",
            "what skills exist",
            "list available skills",
            "show installed capabilities",
            "/skills",
        ):
            self.assertTrue(routehints.answers_directly(text, []), text)

    def test_a_real_request_with_hints_uses_the_rail(self) -> None:
        self.add_skill("code-review", "Review a diff for defects")
        hint = routehints.hint_for("run code-review on this diff", self.library())
        self.assertTrue(hint)
        self.assertFalse(routehints.answers_directly("run code-review on this diff", hint))
        self.assertTrue(routehints.answers_directly("run code-review on this diff", []))


class HistoryTests(_Home):
    def test_record_choice_counts_only_accepted_picks(self) -> None:
        open_state("s1", goal="Ship it", workspace="/tmp")
        open_state("s2", goal="Other", workspace="/tmp")
        routehints.record_choice("s1", "Alpha-One", accepted=True)
        routehints.record_choice("s1", "alpha-one", accepted=True)
        routehints.record_choice("s1", "beta-two", accepted=False)
        routehints.record_choice("s2", "gamma-three", accepted=True)
        self.assertEqual(
            routehints.preference_history(),
            {"alpha-one": 2, "gamma-three": 1},
        )
        self.assertEqual(load_state("s1").skill, "alpha-one")

    def test_history_ignores_unrelated_and_malformed_events(self) -> None:
        append_event("s3", "session_started", {"goal": "unrelated"})
        append_event("s3", routehints.CHOICE_EVENT, {"accepted": "yes"})
        append_event("s3", routehints.CHOICE_EVENT, {"skill": "delta", "accepted": "no"})
        append_event("s3", routehints.CHOICE_EVENT, {"skill": "delta", "accepted": True})
        events, errors = read_events("s3")
        self.assertEqual(len(events), 4)
        self.assertEqual(errors, [])
        self.assertEqual(routehints.preference_history(), {"delta": 1})

    def test_record_choice_survives_a_closed_journey_and_a_missing_state(self) -> None:
        open_state("s4", goal="Done soon", workspace="/tmp")
        close_state("s4", "cancelled")
        routehints.record_choice("s4", "alpha-one", accepted=True)
        routehints.record_choice("no-such-session", "alpha-one", accepted=True)
        routehints.record_choice("s4", "   ", accepted=True)
        self.assertEqual(load_state("s4").skill, "")
        self.assertEqual(routehints.preference_history(), {"alpha-one": 2})

    def test_preference_history_bounds_and_missing_root(self) -> None:
        self.assertEqual(routehints.preference_history(limit=0), {})
        self.assertEqual(routehints.preference_history(limit="bad"), {})
        self.assertEqual(routehints.preference_history(), {})
        root = routehints.wrapper_root(create=True)
        (root / "loose-file").write_text("x", encoding="utf-8")
        self.assertEqual(routehints.preference_history(), {})

    def test_preference_history_skips_sessions_it_cannot_read(self) -> None:
        routehints.record_choice("good-one", "alpha-one", accepted=True)
        loose = routehints.wrapper_root(create=True) / "bad name"
        loose.mkdir()
        (loose / "events.jsonl").write_text(
            json.dumps({"event": routehints.CHOICE_EVENT, "data": {"skill": "beta", "accepted": True}})
            + "\n",
            encoding="utf-8",
        )
        self.assertEqual(routehints.preference_history(), {"alpha-one": 1})


class RenderTests(_Home):
    def test_hint_line_names_the_rail_and_falls_back_to_the_catalog(self) -> None:
        self.add_skill("code-review", "Review a diff for defects")
        self.add_skill("changelog", "Summarize released commits")
        hints = routehints.hint_for(
            "code-review then changelog please", self.library(), limit=2
        )
        line = routehints.hint_line(hints)
        self.assertNotIn("\n", line)
        self.assertIn("·", line)
        self.assertIn("Code Review", line)
        self.assertIn("next: load_skill", line)
        ascii_line = routehints.hint_line(hints, glyphs="ascii")
        self.assertNotIn("·", ascii_line)
        self.assertIn("-", ascii_line)
        empty = routehints.hint_line([])
        self.assertIn("/skills", empty)
        self.assertIn("no strong skill hint", empty)

    def test_hint_line_strips_control_characters_and_stays_bounded(self) -> None:
        directory = self.skills_root / "wide-skill"
        directory.mkdir(parents=True)
        (directory / "SKILL.md").write_text(
            "---\nname: \x1b[31mwide-skill\x1b[0m\ndescription: "
            + "wide " * 200
            + "\n---\n\nbody\n",
            encoding="utf-8",
        )
        hints = routehints.hint_for("wide-skill " * 40, self.library(), limit=3)
        self.assertTrue(hints)
        self.assertNotIn("\x1b", hints[0].skill)
        self.assertNotIn("\x1b", hints[0].label)
        line = routehints.hint_line(hints)
        self.assertNotIn("\x1b", line)
        self.assertLessEqual(len(line), routehints.MAX_HINT_LINE_CHARS + 4)

    def test_hints_payload_is_metadata_only(self) -> None:
        self.add_skill("code-review", "Review a diff for defects")
        request = "run code-review on the billing diff, my password is hunter2"
        payload = routehints.hints_payload(request, self.library())
        self.assertEqual(payload["schema_version"], routehints.HINTS_SCHEMA_VERSION)
        self.assertEqual(payload["message_length"], len(request))
        self.assertEqual(
            payload["prompt_digest"],
            hashlib.sha256(request.encode("utf-8")).hexdigest(),
        )
        self.assertTrue(payload["prompt_reference"].startswith("sha256:"))
        self.assertIn(f"{len(request)} chars", payload["prompt_reference"])
        self.assertEqual(payload["privacy"]["mode"], "metadata_only")
        self.assertIs(payload["privacy"]["raw_prompt_stored"], False)
        self.assertIs(payload["privacy"]["raw_prompt_echoed"], False)
        self.assertIn("route hint", payload["privacy"]["stored_fields"])
        self.assertIn("not skill", payload["claim_boundary"])
        self.assertEqual(payload["mode"], "guided_first_use")
        self.assertEqual(payload["accepted_pattern_count"], 0)
        dumped = json.dumps(payload)
        self.assertNotIn("hunter2", dumped)
        self.assertNotIn("billing", dumped)

    def test_hints_payload_reports_an_autonomous_pattern(self) -> None:
        self.add_skill("code-review", "Review a diff for defects")
        payload = routehints.hints_payload(
            "run code-review on this diff",
            self.library(),
            history={"code-review": routehints.PATTERN_THRESHOLD},
        )
        self.assertEqual(payload["mode"], "autonomous_pattern")
        self.assertEqual(payload["accepted_pattern_count"], routehints.PATTERN_THRESHOLD)
        self.assertTrue(payload["hints"])
        self.assertEqual(payload["hints"][0]["skill"], "code-review")
        self.assertEqual(payload["hints"][0]["adjacent"], [])

    def test_hints_payload_marks_a_catalog_question(self) -> None:
        self.add_skill("alpha-one", "shared description words")
        payload = routehints.hints_payload("which skills exist?", self.library())
        self.assertIs(payload["catalog_question"], True)
        self.assertIs(payload["answers_directly"], True)
        self.assertEqual(payload["hints"][0]["next_action"], "show_skills")

    def test_route_hint_is_immutable_and_json_shaped(self) -> None:
        self.add_skill("alpha-one", "shared description words")
        hint = routehints.hint_for("alpha-one", self.library())[0]
        self.assertIsInstance(hint.adjacent, tuple)
        with self.assertRaises(Exception):
            hint.skill = "other"  # type: ignore[misc]
        payload = routehints.hints_payload("alpha-one", self.library())
        self.assertEqual(
            set(payload["hints"][0]),
            {
                "skill",
                "label",
                "reason",
                "next_action",
                "confidence",
                "score",
                "submit_text",
                "adjacent",
            },
        )


if __name__ == "__main__":
    unittest.main()
