from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ohar.sessions import Session, SessionMeta
from ohar.wrapper import handoff as mod
from ohar.wrapper.evidence import Claim
from ohar.wrapper.records import read_events
from ohar.wrapper.state import SessionState, load_state, open_state


def _state(**changes: object) -> SessionState:
    values: dict[str, object] = {
        "session_id": "h1",
        "status": "working",
        "decision": "accept",
        "goal": "Make the parser recover from a truncated record",
        "workspace": "/tmp/project",
        "model": "anthropic/sonnet",
        "lane": "main",
        "skill": "code-review",
        "next_action": "run tests",
        "revision": 3,
    }
    values.update(changes)
    return SessionState(**values)  # type: ignore[arg-type]


MESSAGES: list[dict[str, object]] = [
    {"role": "system", "content": "ignored"},
    {"role": "user", "content": [{"type": "text", "text": "fix the parser, please"}]},
    {"role": "assistant", "content": "Working on the record reader.", "tool_calls": [{"id": "1"}]},
    {"role": "tool", "content": "tool output", "tool_call_id": "1"},
    {"role": "assistant", "content": "Repaired the \x1b[31mreader\x1b[0m\x00 tail."},
]

PLAN = {
    "steps": [
        {"step": "Read the record", "status": "completed"},
        {"step": "Recover the tail", "status": "in_progress"},
        {"step": "Run tests", "status": "pending"},
    ],
    "counts": {"completed": 1, "in_progress": 1, "pending": 1},
}


class GateTests(unittest.TestCase):
    def test_fence_marker_grows_past_the_longest_run(self) -> None:
        self.assertEqual(mod.fence_marker(""), "```")
        self.assertEqual(mod.fence_marker("plain text"), "```")
        self.assertEqual(mod.fence_marker("a ``` b"), "````")
        self.assertEqual(mod.fence_marker("``````"), "`" * 7)
        self.assertEqual(mod.fence_marker(None), "```")  # type: ignore[arg-type]

    def test_digest_prompt_is_stable_and_bounded(self) -> None:
        digest = mod.digest_prompt("fix the parser")
        self.assertEqual(digest, hashlib.sha256(b"fix the parser").hexdigest())
        self.assertEqual(len(digest), 64)
        self.assertEqual(mod.digest_prompt(None), mod.digest_prompt(""))
        self.assertNotEqual(digest, mod.digest_prompt("fix the parser."))

    def test_prompt_reference_never_carries_the_prompt(self) -> None:
        digest = "533d406a4863" + "0" * 52
        self.assertEqual(mod.prompt_reference(digest, 412), "sha256:533d406a4863 · 412 chars")
        self.assertEqual(mod.prompt_reference(digest, None), "sha256:533d406a4863")
        self.assertEqual(mod.prompt_reference(digest, -1), "sha256:533d406a4863")
        self.assertEqual(mod.prompt_reference(digest, True), "sha256:533d406a4863")
        self.assertEqual(mod.prompt_reference(" sha256:533d-406a! ", 3), "sha256:sha256533d40 · 3 chars")
        self.assertEqual(mod.prompt_reference("  ".join("abc"), None), "sha256:abc")
        self.assertEqual(mod.prompt_reference("", 3), mod.UNKNOWN)
        self.assertEqual(mod.prompt_reference(None, None), mod.UNKNOWN)  # type: ignore[arg-type]

    def test_provenance_prints_every_field_and_degrades_gaps_to_unknown(self) -> None:
        payload = mod.build_provenance(
            model="anthropic/sonnet",
            lane="main",
            skill="code-review",
            status="working",
            task="fix the parser",
            prompt="fix the parser, please",
            evidence_class="observed",
        )
        self.assertEqual(payload["schema_version"], mod.PROVENANCE_SCHEMA_VERSION)
        self.assertEqual(payload["model_route_resolved"], True)
        self.assertEqual(payload["warnings"], [])
        self.assertEqual(
            payload["field_order"],
            ["skill", "model", "status", "task", "prompt", "evidence_class"],
        )
        fields = payload["fields"]
        self.assertEqual(sorted(fields), sorted(payload["field_order"]))
        self.assertEqual(fields["skill"], "code-review")
        self.assertEqual(fields["model"], "main (anthropic/sonnet)")
        self.assertEqual(fields["status"], "working")
        self.assertTrue(fields["prompt"].startswith("sha256:"))
        self.assertIn("22 chars", fields["prompt"])
        self.assertNotIn("parser", fields["prompt"])
        self.assertEqual(fields["evidence_class"], "observed")
        self.assertEqual(payload["prompt_preview"], "fix the parser, please")

    def test_every_absent_field_reads_unknown_with_warnings(self) -> None:
        payload = mod.build_provenance(model="", lane="", evidence_class="")
        fields = payload["fields"]
        for key in ("skill", "model", "status", "task", "prompt", "evidence_class"):
            self.assertEqual(fields[key], mod.UNKNOWN, key)
        self.assertIs(payload["model_route_resolved"], False)
        self.assertNotIn("prompt_preview", payload)
        self.assertEqual(
            payload["warnings"],
            sorted(
                [
                    mod.WARNING_MISSING_MODEL_LABEL,
                    mod.WARNING_UNRESOLVED_MODEL_ROUTE,
                    mod.WARNING_MISSING_STATUS,
                    mod.WARNING_MISSING_PROMPT_REFERENCE,
                    mod.WARNING_UNKNOWN_EVIDENCE_CLASS,
                ]
            ),
        )
        self.assertEqual(payload["warnings"], mod.provenance_warnings(payload))

    def test_three_model_states_are_distinguishable(self) -> None:
        resolved = mod.build_provenance(model="gpt-x", lane="main")
        default_lane = mod.build_provenance(model="", lane="main")
        nothing = mod.build_provenance(model="", lane="")
        self.assertEqual(resolved["fields"]["model"], "main (gpt-x)")
        self.assertEqual(default_lane["fields"]["model"], "main (no model route)")
        self.assertIn(mod.WARNING_UNRESOLVED_MODEL_ROUTE, default_lane["warnings"])
        self.assertNotIn(mod.WARNING_MISSING_MODEL_LABEL, default_lane["warnings"])
        self.assertEqual(nothing["fields"]["model"], mod.UNKNOWN)
        self.assertIn(mod.WARNING_MISSING_MODEL_LABEL, nothing["warnings"])
        for payload in (resolved, default_lane, nothing):
            self.assertNotIn("()", payload["fields"]["model"])

    def test_empty_parentheses_are_named_not_refused(self) -> None:
        payload = mod.build_provenance(model="()", lane="main")
        self.assertEqual(payload["fields"]["model"], "main (())")
        self.assertIn(mod.WARNING_EMPTY_PARENTHESES, payload["warnings"])

    def test_fields_are_single_line_and_length_bounded(self) -> None:
        hostile = "\x1b[31m" + "a" * 400 + "\n\tsecond line\x00"
        payload = mod.build_provenance(
            model=hostile, lane=hostile, skill=hostile, status=hostile, task=hostile
        )
        for value in payload["fields"].values():
            self.assertNotIn("\x1b", value)
            self.assertNotIn("\n", value)
            self.assertNotIn("\x00", value)
            self.assertLessEqual(len(value), mod.MAX_FIELD_CHARS)
        self.assertEqual(mod.MAX_FIELD_CHARS, 96)

    def test_a_digest_without_the_prompt_keeps_the_text_out(self) -> None:
        payload = mod.build_provenance(
            model="gpt-x", lane="main", status="working", prompt_digest="a" * 64
        )
        self.assertEqual(payload["fields"]["prompt"], "sha256:" + "a" * 12)
        self.assertNotIn("prompt_preview", payload)

    def test_prompt_preview_is_bounded_and_marked(self) -> None:
        long_prompt = ("read the record\n" * 100).strip()
        payload = mod.build_provenance(model="gpt-x", lane="main", status="working", prompt=long_prompt)
        preview = payload["prompt_preview"]
        self.assertLessEqual(len(preview), mod.MAX_PROMPT_PREVIEW_CHARS)
        self.assertIn("[truncated,", preview)
        self.assertIn("\n", preview)
        self.assertEqual(mod.MAX_PROMPT_PREVIEW_CHARS, 320)

    def test_evidence_class_vocabulary_is_closed(self) -> None:
        for name in mod.EVIDENCE_CLASSES:
            payload = mod.build_provenance(model="gpt-x", lane="main", evidence_class=name)
            self.assertEqual(payload["fields"]["evidence_class"], name)
        for bogus in ("", "guessed", "provenance"):
            payload = mod.build_provenance(model="gpt-x", lane="main", evidence_class=bogus)
            self.assertEqual(payload["fields"]["evidence_class"], mod.UNKNOWN)
            self.assertIn(mod.WARNING_UNKNOWN_EVIDENCE_CLASS, payload["warnings"])
        matched = mod.build_provenance(
            model="gpt-x", lane="main", status="working", evidence_class=" Observed "
        )
        self.assertEqual(matched["fields"]["evidence_class"], "observed")
        self.assertNotIn(mod.WARNING_UNKNOWN_EVIDENCE_CLASS, matched["warnings"])

    def test_warnings_are_tolerant_of_a_foreign_payload(self) -> None:
        self.assertEqual(
            mod.provenance_warnings({}),
            sorted(
                [
                    mod.WARNING_MISSING_STATUS,
                    mod.WARNING_MISSING_PROMPT_REFERENCE,
                    mod.WARNING_UNKNOWN_EVIDENCE_CLASS,
                ]
            ),
        )
        self.assertEqual(mod.provenance_warnings({"fields": {"model": "main (gpt-x)", "status": "working", "prompt": "sha256:abc", "evidence_class": "observed"}}), [])
        self.assertEqual(mod.provenance_warnings({"fields": "not a mapping"}), mod.provenance_warnings({}))
        self.assertEqual(mod.provenance_warnings({"fields": {"brand_new": ""}}), mod.provenance_warnings({"fields": {}}))

    def test_render_plain_is_one_line_and_fenced_is_aligned(self) -> None:
        payload = mod.build_provenance(
            model="anthropic/sonnet",
            lane="main",
            skill="code-review",
            status="working",
            task="fix the parser",
            prompt="fix the parser, please",
            evidence_class="observed",
        )
        plain = mod.render_provenance(payload, plain=True)
        self.assertNotIn("\n", plain)
        self.assertIn("skill: code-review", plain)
        self.assertIn("evidence: observed", plain)
        self.assertIn(" — ", plain)

        block = mod.render_provenance(payload)
        lines = block.splitlines()
        self.assertEqual(lines[0], lines[-1])
        self.assertTrue(set(lines[0]) <= {"`"})
        self.assertGreaterEqual(len(lines[0]), 3)
        for label in ("SKILL", "MODEL", "STATUS", "TASK", "PROMPT", "EVIDENCE"):
            row = next(line for line in lines if line.startswith(label))
            self.assertEqual(row[8:10], "  ", label)
            self.assertNotEqual(row[10:11], " ", label)
            self.assertTrue(row[10:].strip(), label)
        self.assertIn("HANDOFF PROMPT", block)

    def test_render_fence_nests_a_prompt_that_contains_a_fence(self) -> None:
        payload = mod.build_provenance(
            model="gpt-x", lane="main", status="working", prompt="look:\n```\nvalue\n```"
        )
        block = mod.render_provenance(payload)
        fences = [line for line in block.splitlines() if line and set(line) <= {"`"}]
        self.assertIn(4, [len(fence) for fence in fences])
        self.assertIn("```\nvalue", block)

    def test_render_of_an_empty_payload_renders_nothing(self) -> None:
        self.assertEqual(mod.render_provenance({}), "")
        self.assertEqual(mod.render_provenance({"fields": {}}, plain=True), "")
        self.assertIn("main (gpt-x)", mod.render_provenance({"fields": {"model": "main (gpt-x)", "zeta": "v"}}))

    def test_unknown_field_keys_still_render_at_the_end(self) -> None:
        payload = {
            "fields": {"skill": "code-review", "model": "main (gpt-x)", "zeta": "known"},
            "field_order": ["skill", "model"],
        }
        block = mod.render_provenance(payload)
        self.assertIn("ZETA", block.upper())
        self.assertIn("known", block)

    def test_attach_provenance_places_the_gate_per_surface(self) -> None:
        payload = mod.build_provenance(
            model="gpt-x", lane="main", skill="s", status="working", evidence_class="observed"
        )
        plain = mod.attach_provenance("the answer\n\n", payload, plain=True)
        self.assertTrue(plain.startswith("the answer"))
        self.assertIn("model: main (gpt-x)", plain)
        fenced = mod.attach_provenance("the answer", payload)
        self.assertTrue(fenced.startswith("```"))
        self.assertTrue(fenced.rstrip().endswith("the answer"))
        self.assertEqual(mod.attach_provenance("the answer", {}, plain=True), "the answer")
        self.assertEqual(mod.attach_provenance("", payload, plain=True), mod.render_provenance(payload, plain=True))


class ArtifactTests(unittest.TestCase):
    def test_entry_rows_carry_no_body(self) -> None:
        artifact = mod.Artifact("goal", "Goal", "a body that must not leak", False)
        entry = artifact.entry()
        self.assertEqual(
            entry,
            {"artifact_id": "goal", "label": "Goal", "chars": len(artifact.text), "truncated": False},
        )
        self.assertNotIn("text", entry)
        with self.assertRaises(Exception):
            artifact.text = "other"  # type: ignore[misc]

    def test_listing_is_an_index_only(self) -> None:
        handoff = mod.build_handoff(
            state=_state(goal="A unique goal sentence nobody should get for free"),
            messages=MESSAGES,
            model="anthropic/sonnet",
            lane="main",
        )
        listing = handoff.listing()
        self.assertEqual([row["artifact_id"] for row in listing], list(mod.ARTIFACT_IDS))
        self.assertEqual([row["label"] for row in listing], [mod.ARTIFACT_LABELS[key] for key in mod.ARTIFACT_IDS])
        for row in listing:
            self.assertNotIn("text", row)
            self.assertGreaterEqual(row["chars"], 0)
        self.assertNotIn("unique goal sentence", json.dumps(listing))

    def test_build_handoff_projects_state_plan_and_claims(self) -> None:
        claims = [
            Claim("commands_ran", "observed", "3 commands", ("shell:1",)),
            Claim("tests_pass", "pending", "no test run recorded"),
        ]
        handoff = mod.build_handoff(
            state=_state(),
            messages=MESSAGES,
            plan=PLAN,
            claims=claims,
            model="",
            lane="",
        )
        texts = {artifact.artifact_id: artifact.text for artifact in handoff.artifacts}
        self.assertIn("Make the parser recover", texts["goal"])
        self.assertIn("status: working", texts["current_state"])
        self.assertIn("workspace: /tmp/project", texts["current_state"])
        self.assertIn("fix the parser, please", texts["current_state"])
        self.assertIn("Repaired the reader", texts["current_state"])
        self.assertNotIn("\x1b", texts["current_state"])
        self.assertIn("2. [pending] Run tests", texts["next_steps"])
        self.assertNotIn("Read the record", texts["next_steps"])
        self.assertIn("session next action: run tests", texts["next_steps"])
        self.assertIn("tests_pass: pending", texts["open_questions"])
        self.assertNotIn("commands_ran", texts["open_questions"])
        self.assertIn("commands_ran: observed refs: shell:1", texts["evidence"])
        self.assertIn("claims: observed=1, pending=1", texts["evidence"])
        self.assertTrue(texts["resume_prompt"].startswith("Continue OHAR session h1."))
        # State carries the route, so empty arguments still disclose it.
        self.assertEqual(handoff.provenance["fields"]["model"], "main (anthropic/sonnet)")
        self.assertEqual(handoff.provenance["fields"]["skill"], "code-review")
        self.assertEqual(handoff.provenance["fields"]["evidence_class"], "observed")
        self.assertEqual(handoff.provenance["fields"]["status"], "working")
        self.assertEqual(handoff.provenance["handoff_schema_version"], mod.HANDOFF_SCHEMA_VERSION)

    def test_explicit_route_and_skill_win(self) -> None:
        handoff = mod.build_handoff(
            state=_state(), messages=MESSAGES, model="gpt-x", lane="fast", skill="changelog"
        )
        fields = handoff.provenance["fields"]
        self.assertEqual(fields["model"], "fast (gpt-x)")
        self.assertEqual(fields["skill"], "changelog")

    def test_evidence_class_follows_the_claims_then_the_transcript(self) -> None:
        observed = mod.build_handoff(
            state=_state(),
            messages=MESSAGES,
            claims=[Claim("build_ok", "observed")],
            model="gpt-x",
            lane="main",
        )
        pending = mod.build_handoff(
            state=_state(), messages=MESSAGES, claims=[Claim("build_ok", "pending")], model="gpt-x", lane="main"
        )
        spoken = mod.build_handoff(state=_state(), messages=MESSAGES, model="gpt-x", lane="main")
        empty = mod.build_handoff(state=_state(), messages=[], model="gpt-x", lane="main")
        self.assertEqual(observed.provenance["fields"]["evidence_class"], "observed")
        self.assertEqual(pending.provenance["fields"]["evidence_class"], "planned")
        self.assertEqual(spoken.provenance["fields"]["evidence_class"], "model_reported")
        self.assertEqual(empty.provenance["fields"]["evidence_class"], mod.UNKNOWN)
        self.assertIn(mod.WARNING_UNKNOWN_EVIDENCE_CLASS, empty.provenance["warnings"])
        self.assertIn("(no recorded evidence)", mod.artifact_text(empty, "evidence", fence=False))

    def test_claims_and_messages_tolerate_foreign_shapes(self) -> None:
        class Odd:
            claim = "tests_pass"
            state = "refuted"
            reason = "the run failed"
            refs = None

        handoff = mod.build_handoff(
            state=_state(status="needs_review"),
            messages=[{"role": "user", "content": "carry on"}, "not a dict", {"no_role": 1}],
            plan={"steps": "not a list"},
            claims=[Odd(), {"claim": "work_verified", "state": "pending", "reason": "not reviewed", "refs": ["r"]}, None, "junk"],
            model="gpt-x",
            lane="main",
        )
        questions = mod.artifact_text(handoff, "open_questions", fence=False)
        self.assertIn("tests_pass: refuted — the run failed", questions)
        self.assertIn("work_verified: pending — not reviewed", questions)
        self.assertIn("needs_review", questions)
        self.assertIn("carry on", mod.artifact_text(handoff, "current_state", fence=False))

    def test_a_session_object_can_be_passed_as_the_transcript(self) -> None:
        session = Session(meta=SessionMeta(id="h1"), messages=list(MESSAGES))
        handoff = mod.build_handoff(state=_state(), messages=session, model="gpt-x", lane="main")
        self.assertIn("fix the parser, please", mod.artifact_text(handoff, "current_state", fence=False))

    def test_bodies_are_bounded_and_marked(self) -> None:
        huge = "line\n" * 4000
        handoff = mod.build_handoff(
            state=_state(goal=huge), messages=MESSAGES, model="gpt-x", lane="main"
        )
        goal = mod.select_artifact(handoff, "goal")
        self.assertTrue(goal.truncated)
        self.assertLessEqual(len(goal.text), mod.MAX_ARTIFACT_CHARS)
        self.assertIn("[truncated,", goal.text)
        self.assertEqual(goal.entry()["chars"], len(goal.text))
        self.assertEqual(mod.MAX_ARTIFACT_CHARS, 1600)

    def test_missing_pieces_still_produce_every_artifact(self) -> None:
        handoff = mod.build_handoff(
            state=SessionState(session_id="bare"), messages=[], model="", lane=""
        )
        self.assertEqual([a.artifact_id for a in handoff.artifacts], list(mod.ARTIFACT_IDS))
        self.assertIn(mod.UNKNOWN, mod.artifact_text(handoff, "goal", fence=False))
        self.assertIn("(none recorded)", mod.artifact_text(handoff, "open_questions", fence=False))
        self.assertIn("confirm the goal", mod.artifact_text(handoff, "next_steps", fence=False))
        self.assertEqual(handoff.provenance["warnings"], mod.provenance_warnings(handoff.provenance))

    def test_select_artifact_by_id_or_label(self) -> None:
        handoff = mod.build_handoff(state=_state(), messages=MESSAGES, model="gpt-x", lane="main")
        self.assertEqual(mod.select_artifact(handoff, "NEXT_STEPS").artifact_id, "next_steps")
        self.assertEqual(mod.select_artifact(handoff, "  evidence  ").artifact_id, "evidence")
        self.assertEqual(mod.select_artifact(handoff, "Open Questions").artifact_id, "open_questions")
        with self.assertRaises(mod.HandoffError) as raised:
            mod.select_artifact(handoff, "secrets")
        self.assertIn("secrets", str(raised.exception))
        self.assertIn("goal", str(raised.exception))
        with self.assertRaises(mod.HandoffError):
            mod.select_artifact(handoff, "")
        self.assertTrue(issubclass(mod.HandoffError, ValueError))

    def test_artifact_text_fences_from_its_own_content(self) -> None:
        handoff = mod.Handoff(
            session_id="h2",
            artifacts=[mod.Artifact("evidence", "Evidence", "a ``` fenced note", False)],
            provenance={},
        )
        raw = mod.artifact_text(handoff, "evidence", fence=False)
        self.assertEqual(raw, "a ``` fenced note")
        wrapped = mod.artifact_text(handoff, "evidence")
        self.assertIn("````", wrapped)
        self.assertIn("\n````\n", wrapped)
        self.assertTrue(wrapped.startswith("Evidence:"))

    def test_resume_prompt_is_copyable_and_bounded(self) -> None:
        handoff = mod.build_handoff(
            state=_state(), messages=MESSAGES, plan=PLAN, claims=[Claim("tests_pass", "pending")], model="gpt-x", lane="main"
        )
        prompt = mod.resume_prompt(handoff)
        self.assertIn("Continue OHAR session h1.", prompt)
        for label in ("Goal:", "Current state:", "Next steps:", "Open questions:", "Evidence:"):
            self.assertIn(label, prompt)
        self.assertIn("Pick up at the first next step", prompt)
        self.assertLessEqual(len(prompt), 4000)
        short = mod.resume_prompt(handoff, limit=60)
        self.assertLessEqual(len(short), 60)
        self.assertIn("[truncated,", short)
        self.assertEqual(mod.resume_prompt(handoff, limit=0), "")
        self.assertEqual(mod.resume_prompt(handoff, limit="bad"), "")  # type: ignore[arg-type]
        for ceiling in (1, 8, 20):
            self.assertLessEqual(len(mod.resume_prompt(handoff, limit=ceiling)), ceiling)
        self.assertIn("session unknown", mod.resume_prompt(mod.Handoff(session_id="   ")))

    def test_building_a_handoff_never_advances_session_state(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            home = Path(raw) / "home"
            with mock.patch.dict(os.environ, {"OHAR_HOME": str(home)}):
                open_state("h3", goal="Ship it", workspace="/tmp/project")
                before, _errors = read_events("h3")
                handoff = mod.build_handoff(
                    state=load_state("h3"),
                    messages=MESSAGES,
                    plan=PLAN,
                    claims=[Claim("commands_ran", "observed", "seen", ("shell:1",))],
                    model="anthropic/sonnet",
                    lane="main",
                )
                mod.artifact_text(handoff, "resume_prompt")
                after, _errors2 = read_events("h3")
                self.assertEqual([event["event"] for event in before], [event["event"] for event in after])
                self.assertEqual(load_state("h3").status, "open")
                self.assertEqual(load_state("h3").next_action, "draft_or_present_plan")
                self.assertEqual(handoff.session_id, "h3")


if __name__ == "__main__":
    unittest.main()
