from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from ohar.types import BrowserConfig, Config, McpServerConfig, MemoryConfig, ToolContext, WebConfig
from ohar.workflows.journal import ActivityJournal
from ohar.workflows.readiness import (
    CLAIM_BOUNDARY,
    EVIDENCE_TIERS,
    MAX_ACTION_CHARS,
    MAX_REASONS,
    NEXT_ACTIONS,
    READINESS_STATES,
    REPORTED_TIERS,
    STALE_AFTER_SECONDS,
    STALE_TIER,
    STATE_SOURCES,
    SURFACES,
    CapabilityReadiness,
    Evidence,
    ReadinessAnswer,
    SurfaceState,
    answer_readiness,
    build_evidence,
    evidence_errors,
    make_readiness_tool,
    surfaces_from_config,
    tier_level,
)
from ohar.workflows.store import WorkflowError

MOMENT = datetime(2026, 10, 25, 12, 0, 0, tzinfo=timezone.utc).timestamp()


def stamp(offset: float) -> str:
    return datetime.fromtimestamp(MOMENT + offset, timezone.utc).isoformat(timespec="microseconds")


def record(
    surface: str = "shell",
    *,
    action: str = "",
    tier: str = "usable_observed",
    result: str = "observed",
    at: str | None = None,
    detail: str = "journal tool_observed",
) -> dict[str, object]:
    return build_evidence(
        surface=surface,
        action=action,
        tier=tier,
        result=result,
        at=stamp(-60) if at is None else at,
        detail=detail,
    )


class VocabularyTests(unittest.TestCase):
    def test_tiers_are_ordered_and_the_derived_ones_rank_lowest(self) -> None:
        self.assertEqual(EVIDENCE_TIERS, ("installed", "host_observed", "usable_observed", "used"))
        self.assertLess(tier_level("installed"), tier_level("host_observed"))
        self.assertLess(tier_level("host_observed"), tier_level("usable_observed"))
        self.assertLess(tier_level("usable_observed"), tier_level("used"))
        self.assertEqual(tier_level(STALE_TIER), 0)
        self.assertEqual(tier_level("none"), 0)
        self.assertEqual(tier_level("magic"), 0)
        self.assertEqual(tier_level("USED"), 4)

    def test_every_state_names_its_smallest_next_action(self) -> None:
        self.assertEqual(set(NEXT_ACTIONS), set(READINESS_STATES))
        self.assertEqual(set(STATE_SOURCES), {"derived", "preserved_prior", "no_valid_evidence"})
        self.assertEqual(NEXT_ACTIONS["ready"], "perform_the_action")
        self.assertNotEqual(NEXT_ACTIONS["blocked"], NEXT_ACTIONS["stale"])
        self.assertIn("stale", REPORTED_TIERS)
        self.assertNotIn(STALE_TIER, EVIDENCE_TIERS)

    def test_surface_vocabulary_is_ohar_own(self) -> None:
        self.assertEqual(SURFACES, ("files", "shell", "web", "browser", "mcp", "subagents", "memory", "workflows"))
        self.assertEqual(STALE_AFTER_SECONDS, 21600)
        self.assertEqual(MAX_ACTION_CHARS, 160)
        self.assertEqual(MAX_REASONS, 6)


class EvidenceTests(unittest.TestCase):
    def test_build_returns_the_closed_key_set(self) -> None:
        data = record("shell", tier="used", action="write the report")
        self.assertEqual(set(data), {"surface", "action", "tier", "result", "at", "detail"})
        self.assertEqual(evidence_errors(data), [])
        self.assertEqual(evidence_errors(Evidence(**data)), [])
        self.assertEqual(
            Evidence(**data).to_dict(),
            {
                "surface": "shell",
                "action": "write the report",
                "tier": "used",
                "result": "observed",
                "at": stamp(-60),
                "detail": "journal tool_observed",
            },
        )

    def test_defaults_are_filled_and_text_is_bounded(self) -> None:
        data = build_evidence(surface="  MCP:FS\x1b[0m ", action="run", tier="installed")
        self.assertEqual(data["surface"], "mcp:fs")
        self.assertEqual(data["result"], "observed")
        self.assertTrue(str(data["at"]))
        long = build_evidence(surface="shell", action="x" * 500, tier="installed")
        self.assertLessEqual(len(str(long["action"])), MAX_ACTION_CHARS)

    def test_mint_refuses_unusable_records(self) -> None:
        with self.assertRaises(WorkflowError):
            build_evidence(surface="shell", action="", tier="stale")
        with self.assertRaises(WorkflowError):
            build_evidence(surface="shell", action="", tier="used")
        with self.assertRaises(WorkflowError):
            build_evidence(surface="teleport", action="", tier="installed")
        with self.assertRaises(WorkflowError):
            build_evidence(surface="shell", action="", tier="installed", result="guessed")
        with self.assertRaises(WorkflowError):
            build_evidence(surface="files:x", action="", tier="installed")
        blank = {**record(), "at": ""}
        self.assertIn("at is required", " ".join(evidence_errors(blank)))

    def test_errors_are_reported_not_raised_when_reading(self) -> None:
        self.assertEqual(evidence_errors("nope"), ["evidence must be an object"])
        self.assertEqual(evidence_errors(None), ["evidence must be an object"])
        self.assertIn(
            "evidence is missing keys",
            " ".join(evidence_errors({"surface": "shell"})),
        )
        self.assertIn(
            "unsupported keys",
            " ".join(evidence_errors({**record(), "extra": 1})),
        )
        secret = {**record(), "api_key": "sk-1"}
        self.assertIn("raw or secret keys", " ".join(evidence_errors(secret)))
        multiline = build_evidence(surface="shell", action="", tier="installed", detail="a\nb\x1b[31m")
        self.assertNotIn("\n", str(multiline["detail"]))
        self.assertNotIn("\x1b", str(multiline["detail"]))
        invented = {**record(), "surface": "files:x"}
        self.assertIn("not an OHAR capability surface", " ".join(evidence_errors(invented)))

    def test_a_used_record_must_name_the_action_it_observed(self) -> None:
        bad = {
            "surface": "shell",
            "action": "",
            "tier": "used",
            "result": "observed",
            "at": stamp(-1),
            "detail": "",
        }
        self.assertIn("must name the action", " ".join(evidence_errors(bad)))


class AnswerReadinessTests(unittest.TestCase):
    def test_configuration_alone_can_never_answer_ready(self) -> None:
        answer = answer_readiness(surface="shell", action="", configured=True, evidence=[], now=MOMENT)
        self.assertEqual(answer.state, "not_observed")
        self.assertEqual(answer.tier, "installed")
        self.assertEqual(answer.next_action, "observe_the_surface_before_acting")
        self.assertEqual(answer.age_seconds, -1)
        self.assertEqual(answer.state_source, "derived")
        self.assertEqual(answer.evidence_count, 0)
        self.assertIn("configured locally and nothing has been observed", " ".join(answer.reasons))

    def test_presence_tiers_stop_below_ready(self) -> None:
        for tier in ("installed", "host_observed"):
            answer = answer_readiness(
                surface="web",
                action="",
                configured=True,
                evidence=[record("web", tier=tier)],
                now=MOMENT,
            )
            self.assertEqual(answer.state, "not_observed", tier)
            self.assertEqual(answer.tier, tier)
            self.assertEqual(answer.evidence_count, 1)

    def test_observed_use_answers_ready_and_names_the_age(self) -> None:
        answer = answer_readiness(surface="shell", action="", configured=True, evidence=[record()], now=MOMENT)
        self.assertEqual(answer.state, "ready")
        self.assertEqual(answer.tier, "usable_observed")
        self.assertEqual(answer.next_action, "perform_the_action")
        self.assertEqual(answer.age_seconds, 60)
        self.assertEqual(answer.state, "ready")

        used = answer_readiness(
            surface="mcp:fs",
            action="list the directory",
            configured=False,
            evidence=[record("mcp:fs", action="list the directory", tier="used")],
            now=MOMENT,
        )
        self.assertEqual(used.state, "ready")
        self.assertEqual(used.tier, "used")
        self.assertEqual(used.surface, "mcp:fs")

    def test_freshness_is_derived_at_read_time_not_stored(self) -> None:
        evidence = [record(at=stamp(-30_000))]
        wide = answer_readiness(surface="shell", action="", configured=True, evidence=evidence, now=MOMENT, stale_after_seconds=40_000)
        self.assertEqual(wide.state, "ready")
        short = answer_readiness(surface="shell", action="", configured=True, evidence=evidence, now=MOMENT, stale_after_seconds=60)
        self.assertEqual(short.state, "stale")
        self.assertEqual(short.tier, STALE_TIER)
        self.assertEqual(short.next_action, "re_observe_the_surface_before_acting")
        self.assertGreater(short.age_seconds, 30_000 - 60)

    def test_an_unreadable_or_future_stamp_cannot_read_as_fresh(self) -> None:
        for at in ("not a timestamp", stamp(+7_200)):
            answer = answer_readiness(surface="shell", action="", configured=True, evidence=[record(at=at)], now=MOMENT)
            self.assertEqual(answer.state, "stale", at)
            self.assertEqual(answer.tier, STALE_TIER)

    def test_blocked_and_failed_are_their_own_answers(self) -> None:
        blocked = answer_readiness(
            surface="browser",
            action="click submit",
            configured=True,
            evidence=[record("browser", result="blocked", detail="no chromium executable")],
            now=MOMENT,
        )
        self.assertEqual(blocked.state, "blocked")
        self.assertEqual(blocked.next_action, "clear_the_recorded_blocker")
        self.assertIn("no chromium executable", " ".join(blocked.reasons))

        failed = answer_readiness(
            surface="shell",
            action="",
            configured=True,
            evidence=[record("shell", result="failed", detail="exited 127")],
            now=MOMENT,
        )
        self.assertEqual(failed.state, "failed")
        self.assertEqual(failed.next_action, "read_the_recorded_failure_before_retrying")

    def test_a_negative_only_stands_until_something_at_least_as_new_replaces_it(self) -> None:
        older = answer_readiness(
            surface="shell",
            action="",
            configured=True,
            evidence=[
                record("shell", result="failed", at=stamp(-300), detail="exited 127"),
                record("shell", at=stamp(-60), detail="ran green"),
            ],
            now=MOMENT,
        )
        self.assertEqual(older.state, "ready")
        newer = answer_readiness(
            surface="shell",
            action="",
            configured=True,
            evidence=[
                record("shell", at=stamp(-300), detail="ran green"),
                record("shell", result="failed", at=stamp(-10), detail="exited 127"),
            ],
            now=MOMENT,
        )
        self.assertEqual(newer.state, "failed")
        self.assertEqual(newer.tier, "usable_observed")

    def test_evidence_for_another_action_is_out_of_scope_not_rejected(self) -> None:
        answer = answer_readiness(
            surface="shell",
            action="write the report",
            configured=True,
            evidence=[record("shell", action="list files", tier="used")],
            now=MOMENT,
        )
        self.assertEqual(answer.state, "not_observed")
        self.assertEqual(answer.evidence_count, 0)
        self.assertEqual(answer.rejected_count, 0)

    def test_a_sibling_server_never_substitutes_for_the_one_asked_about(self) -> None:
        answer = answer_readiness(
            surface="mcp:fs",
            action="",
            configured=True,
            evidence=[record("mcp:git", tier="used", action="commit")],
            now=MOMENT,
        )
        self.assertEqual(answer.evidence_count, 0)
        self.assertEqual(answer.state, "not_observed")
        family = answer_readiness(surface="mcp", action="", configured=True, evidence=[record("mcp:fs")], now=MOMENT)
        self.assertEqual(family.state, "ready")

    def test_invalid_evidence_is_rejected_and_cannot_raise_the_answer(self) -> None:
        bad = {"surface": "shell", "action": "", "tier": "used", "result": "observed", "at": stamp(-1), "detail": ""}
        answer = answer_readiness(
            surface="shell",
            action="",
            configured=True,
            evidence=[bad, bad],
            now=MOMENT,
        )
        self.assertEqual(answer.state, "not_observed")
        self.assertEqual(answer.rejected_count, 2)
        self.assertEqual(answer.evidence_count, 0)
        self.assertEqual(answer.state_source, "no_valid_evidence")
        self.assertIn("rejected evidence cannot raise this answer", " ".join(answer.reasons))
        mixed = answer_readiness(
            surface="shell",
            action="",
            configured=True,
            evidence=[bad, record()],
            now=MOMENT,
        )
        self.assertEqual(mixed.state, "ready")
        self.assertEqual(mixed.rejected_count, 1)
        self.assertEqual(mixed.state_source, "derived")

    def test_a_prior_is_preserved_only_when_nothing_in_scope_survived(self) -> None:
        prior = {"surface": "shell", "action": "", "state": "stale", "tier": STALE_TIER}
        kept = answer_readiness(
            surface="shell",
            action="",
            configured=True,
            evidence=[{"tier": "used"}],
            prior=prior,
            now=MOMENT,
        )
        self.assertEqual(kept.state_source, "preserved_prior")
        self.assertEqual(kept.state, "stale")
        self.assertEqual(kept.age_seconds, -1)
        self.assertIn("last valid answer is kept", " ".join(kept.reasons))

        ignored = answer_readiness(
            surface="shell",
            action="",
            configured=True,
            evidence=[{"tier": "used"}, record()],
            prior=prior,
            now=MOMENT,
        )
        self.assertEqual(ignored.state_source, "derived")
        self.assertEqual(ignored.state, "ready")

        wrong_surface = answer_readiness(
            surface="shell",
            action="",
            configured=True,
            evidence=[{"tier": "used"}],
            prior={"surface": "web", "action": "", "state": "stale", "tier": STALE_TIER},
            now=MOMENT,
        )
        self.assertEqual(wrong_surface.state_source, "no_valid_evidence")

        junk_prior = answer_readiness(
            surface="shell",
            action="",
            configured=True,
            evidence=[{"tier": "used"}],
            prior={"surface": "shell", "action": "", "state": "probably", "tier": "used"},
            now=MOMENT,
        )
        self.assertEqual(junk_prior.state_source, "no_valid_evidence")
        self.assertEqual(junk_prior.state, "not_observed")

    def test_unknown_surfaces_are_refused_and_text_stays_bounded(self) -> None:
        with self.assertRaises(WorkflowError):
            answer_readiness(surface="teleport", action="", configured=False)
        answer = answer_readiness(
            surface="  Shell\x1b[0m ",
            action="run the " + "x" * 400,
            configured=True,
            evidence=[record("shell", detail="\x1b[31mcolour\n")],
            now=MOMENT,
        )
        self.assertEqual(answer.surface, "shell")
        self.assertLessEqual(len(answer.action), MAX_ACTION_CHARS)
        self.assertLessEqual(len(answer.reasons), MAX_REASONS)
        self.assertTrue(all("\x1b" not in line and "\n" not in line for line in answer.reasons))
        payload = answer.to_dict()
        self.assertEqual(
            set(payload),
            {
                "surface",
                "action",
                "state",
                "tier",
                "age_seconds",
                "next_action",
                "reasons",
                "evidence_count",
                "rejected_count",
                "state_source",
            },
        )
        self.assertIsInstance(ReadinessAnswer(**payload), ReadinessAnswer)


class _Journal(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name)
        self.workspace = base / "workspace"
        self.workspace.mkdir()
        self.now = MOMENT
        self.journal = ActivityJournal(base / "home", self.workspace, clock=lambda: self.now)
        self.surfaces = [
            SurfaceState("shell", True, True, True, "", "bash"),
            SurfaceState("files", True, True, True, "", str(self.workspace)),
            SurfaceState("browser", True, True, False, "no chromium executable was found", ""),
            SurfaceState("web", True, False, True, "web is disabled in configuration", ""),
            SurfaceState("mcp:fs", True, True, True, "", "declared server"),
        ]

    def reader(self, **kwargs: object) -> CapabilityReadiness:
        options: dict[str, object] = {
            "approval_mode": "ask",
            "surfaces": self.surfaces,
            "journal": self.journal,
            "tool_names": ["shell", "read_file", "write_file", "mcp__fs__list"],
            "now": lambda: self.now,
        }
        options.update(kwargs)  # type: ignore[arg-type]
        return CapabilityReadiness(**options)  # type: ignore[arg-type]


class CapabilityReadinessTests(_Journal):
    def test_configuration_without_observation_is_never_ready(self) -> None:
        answer = self.reader().answer("shell")
        self.assertEqual(answer.state, "not_observed")
        self.assertEqual(answer.tier, "installed")
        self.assertEqual(answer.evidence_count, 0)

    def test_a_journal_observation_of_a_bound_tool_carries_the_surface(self) -> None:
        reader = self.reader()
        self.journal.record("tool_observed", subject="shell", ok=True, session_id="s1")
        answer = reader.answer("shell")
        self.assertEqual(answer.state, "ready")
        self.assertEqual(answer.tier, "usable_observed")
        self.assertEqual(answer.age_seconds, 0)
        self.assertIn("session s1", " ".join(answer.reasons))

    def test_an_observation_with_no_outcome_stays_below_ready(self) -> None:
        self.journal.record("session_started", subject="shell", ok=None)
        answer = self.reader().answer("shell")
        self.assertEqual(answer.state, "not_observed")
        self.assertEqual(answer.tier, "host_observed")

    def test_a_failed_observation_answers_failed_not_ready(self) -> None:
        self.journal.record("tool_observed", subject="shell", ok=False)
        answer = self.reader().answer("shell")
        self.assertEqual(answer.state, "failed")
        self.assertEqual(answer.next_action, "read_the_recorded_failure_before_retrying")

    def test_observed_ageing_out_is_derived_at_read_time(self) -> None:
        reader = self.reader()
        self.journal.record("tool_observed", subject="shell", ok=True)
        self.assertEqual(reader.answer("shell").state, "ready")
        self.now = MOMENT + STALE_AFTER_SECONDS + 60
        self.assertEqual(reader.answer("shell").state, "stale")

    def test_disabled_and_host_absent_surfaces_are_blocked_with_a_reason(self) -> None:
        reader = self.reader()
        browser = reader.answer("browser")
        self.assertEqual(browser.state, "blocked")
        self.assertIn("no chromium executable", " ".join(browser.reasons))
        web = reader.answer("web")
        self.assertEqual(web.state, "blocked")
        self.assertIn("web is disabled", " ".join(web.reasons))

    def test_read_only_mode_blocks_a_mutating_action_only(self) -> None:
        reader = self.reader(approval_mode="read-only")
        mutating = reader.answer("shell", "write the report")
        self.assertEqual(mutating.state, "blocked")
        self.assertIn("read-only", " ".join(mutating.reasons))
        reading = reader.answer("files", "read the file")
        self.assertNotEqual(reading.state, "blocked")

    def test_a_surface_no_bound_tool_can_reach_is_not_capability(self) -> None:
        unbound = self.reader(tool_names=["shell"])
        answer = unbound.answer("mcp:fs")
        self.assertEqual(answer.state, "blocked")
        self.assertIn("no bound tool", " ".join(answer.reasons))
        reader = self.reader()
        self.assertEqual(reader.bound_tools("shell"), ["shell"])
        self.assertEqual(reader.bound_tools("mcp"), ["mcp__fs__list"])
        self.assertEqual(reader.bound_tools("files"), ["read_file", "write_file"])

    def test_subagent_and_memory_surfaces_answer_without_being_observed(self) -> None:
        reader = self.reader()
        for surface in ("memory", "subagents", "workflows"):
            answer = reader.answer(surface)
            self.assertIn(answer.state, ("not_observed", "blocked"), surface)
            self.assertNotEqual(answer.state, "ready", surface)

    def test_survey_covers_every_surface_once_plus_the_requested_actions(self) -> None:
        reader = self.reader()
        answers = reader.survey(actions={"shell": ["write the report"], "files": ["read a file"]})
        bare = [answer for answer in answers if not answer.action]
        self.assertEqual([answer.surface for answer in bare], list(SURFACES) + ["mcp:fs"])
        self.assertEqual(len(answers), len(bare) + 2)
        self.assertTrue(all(isinstance(answer, ReadinessAnswer) for answer in answers))

    def test_rows_and_headers_line_up(self) -> None:
        reader = self.reader()
        answers = reader.survey()
        table_rows = reader.rows(answers)
        self.assertEqual(len(reader.headers()), 7)
        self.assertEqual({len(row) for row in table_rows}, {len(reader.headers())})
        self.assertEqual(table_rows[1][0], "shell")
        self.assertIn("any action", table_rows[1][1])

    def test_answer_for_an_unknown_surface_is_refused(self) -> None:
        with self.assertRaises(WorkflowError):
            self.reader().answer("teleport")


class SurfacesFromConfigTests(unittest.TestCase):
    def config(self, **overrides: object) -> Config:
        options: dict[str, object] = {
            "approval": "ask",
            "workspace": "/srv/project",
            "web": WebConfig(enabled=True),
            "browser": BrowserConfig(enabled=True),
            "memory": MemoryConfig(enabled=True),
            "subagents_max": 4,
            "mcp_servers": [McpServerConfig(name="fs", command="mcp-fs"), McpServerConfig(name="git", command="mcp-git")],
        }
        options.update(overrides)
        return Config(**options)  # type: ignore[arg-type]

    def by_name(self, states: list[SurfaceState]) -> dict[str, SurfaceState]:
        return {state.name: state for state in states}

    def test_default_config_facts_map_onto_surfaces(self) -> None:
        states = self.by_name(surfaces_from_config(self.config(), mcp_status={"fs": "ready"}, browser_available=True))
        self.assertEqual(sorted(states), sorted([*SURFACES, "mcp:fs", "mcp:git"]))
        self.assertTrue(states["files"].configured and states["files"].host_ready)
        self.assertTrue(states["shell"].host_ready)
        self.assertTrue(states["browser"].host_ready)
        self.assertTrue(states["mcp:fs"].host_ready)
        self.assertFalse(states["mcp:git"].host_ready)
        self.assertIn("not_observed", states["mcp:git"].reason)
        self.assertTrue(states["subagents"].configured)

    def test_absent_facts_are_declared_not_invented(self) -> None:
        states = self.by_name(
            surfaces_from_config(
                self.config(approval="read-only", web=WebConfig(enabled=False), subagents_max=0),
                mcp_status=None,
                browser_available=False,
                memory_enabled=False,
            )
        )
        self.assertFalse(states["shell"].host_ready)
        self.assertIn("read-only", states["shell"].reason)
        self.assertFalse(states["web"].enabled)
        self.assertFalse(states["browser"].host_ready)
        self.assertFalse(states["memory"].enabled)
        self.assertFalse(states["subagents"].configured)
        # Servers are declared, but no host reported any of them ready: that is a
        # declared fact, not a capability.
        self.assertTrue(states["mcp"].configured)
        self.assertFalse(states["mcp"].host_ready)
        self.assertIn("no mcp server reported ready", states["mcp"].reason)
        self.assertFalse(states["mcp:fs"].host_ready)

    def test_no_servers_still_yields_one_mcp_surface(self) -> None:
        states = self.by_name(surfaces_from_config(self.config(mcp_servers=[])))
        self.assertEqual(sorted(states), sorted(list(SURFACES)))
        self.assertFalse(states["mcp"].configured)
        self.assertIn("no mcp servers", states["mcp"].reason)


class ReadinessToolTests(_Journal):
    def setUp(self) -> None:
        super().setUp()
        self.reader = self.reader()
        self.tool = make_readiness_tool(self.reader)
        self.ctx = ToolContext(workspace=self.workspace, approval_mode="ask")

    def test_the_tool_is_read_only_and_asks_for_a_surface(self) -> None:
        self.assertTrue(self.tool.read_only)
        self.assertFalse(self.tool.destructive)
        self.assertEqual(self.tool.name, "capabilities")
        spec = self.tool.api_spec()["function"]["parameters"]
        self.assertEqual(spec["required"], ["surface"])
        self.assertFalse(spec["additionalProperties"])
        missing = self.tool.handler({}, self.ctx)
        self.assertFalse(missing.ok)
        self.assertEqual(missing.meta["reason"], "surface_required")
        self.assertIn("files", missing.content)

    def test_one_surface_answers_with_state_tier_and_next_action(self) -> None:
        self.journal.record("tool_observed", subject="shell", ok=True)
        result = self.tool.handler({"surface": "shell"}, self.ctx)
        self.assertTrue(result.ok)
        self.assertEqual(result.meta["state"], "ready")
        self.assertEqual(result.meta["next_action"], "perform_the_action")
        self.assertNotIn("\x1b", result.content)
        self.assertIn("shell", result.content)
        self.assertIn(CLAIM_BOUNDARY.split()[0], result.content)

    def test_an_action_scopes_the_answer(self) -> None:
        result = self.tool.handler({"surface": "browser", "action": "click submit"}, self.ctx)
        self.assertTrue(result.ok)
        self.assertEqual(result.meta["state"], "blocked")

    def test_all_surveys_every_surface(self) -> None:
        result = self.tool.handler({"surface": "all"}, self.ctx)
        self.assertTrue(result.ok)
        self.assertEqual(result.meta["count"], len(SURFACES) + 1)
        for surface in SURFACES:
            self.assertIn(surface, result.content)

    def test_unknown_surface_is_a_refusal_not_a_crash(self) -> None:
        result = self.tool.handler({"surface": "teleport"}, self.ctx)
        self.assertFalse(result.ok)
        self.assertEqual(result.meta["reason"], "surface_unknown")
        self.assertIn("Known surfaces", result.content)


if __name__ == "__main__":
    unittest.main()
