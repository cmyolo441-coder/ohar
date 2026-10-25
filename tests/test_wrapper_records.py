from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ohar.wrapper import records

SESSION = "sess-1"


def mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


class WrapperHomeMixin(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.previous_home = os.environ.get("OHAR_HOME")
        os.environ["OHAR_HOME"] = str(self.base / "home")

    def tearDown(self) -> None:
        if self.previous_home is None:
            os.environ.pop("OHAR_HOME", None)
        else:
            os.environ["OHAR_HOME"] = self.previous_home
        self.tmp.cleanup()

    def record_path(self, name: str = "state.json", session_id: str = SESSION) -> Path:
        return records.session_dir(session_id) / name


class RootTests(WrapperHomeMixin):
    def test_root_and_session_dirs_are_private(self) -> None:
        root = records.wrapper_root()
        directory = records.session_dir(SESSION)
        self.assertEqual(root, self.base / "home" / "wrapper")
        self.assertEqual(directory, root / SESSION)
        self.assertEqual(mode(root), 0o700)
        self.assertEqual(mode(directory), 0o700)

    def test_create_false_writes_nothing(self) -> None:
        root = records.wrapper_root(create=False)
        self.assertEqual(records.session_dir(SESSION, create=False), root / SESSION)
        self.assertFalse(root.exists())

    def test_session_id_is_revalidated(self) -> None:
        for bad in ("", "../escape", "a/b", ".", "..", "sp ace", "x\x00y", None, 7):
            with self.subTest(bad=bad):
                with self.assertRaises(records.RecordError):
                    records.session_dir(bad)  # type: ignore[arg-type]

    def test_record_name_is_confined_to_one_component(self) -> None:
        for bad in ("", "../state", "a/b.json", ".hidden", "..json.json"):
            with self.subTest(bad=bad):
                with self.assertRaises(records.RecordError):
                    records.write_record(bad, SESSION, {})
        self.assertTrue(records.read_record("state", SESSION) == {})
        records.write_record("state", SESSION, {"marker": 1})
        self.assertTrue((records.session_dir(SESSION) / "state.json").is_file())


class ReadWriteTests(WrapperHomeMixin):
    def test_read_record_never_raises_on_missing_or_corrupt(self) -> None:
        self.assertEqual(records.read_record("state.json", SESSION), {})
        path = self.record_path()
        path.write_text("{not json", encoding="utf-8")
        self.assertEqual(records.read_record("state.json", SESSION), {})
        path.write_text("[1, 2]", encoding="utf-8")
        self.assertEqual(records.read_record("state.json", SESSION), {})
        path.write_bytes(b'{"a": "\xff\xfe"}')
        self.assertEqual(records.read_record("state.json", SESSION), {})

    def test_write_record_is_atomic_private_and_bound(self) -> None:
        written = records.write_record("state.json", SESSION, {"status": "open", "n": 2})
        self.assertEqual(written["schema_version"], records.SCHEMA_VERSION)
        self.assertEqual(written["session_id"], SESSION)
        self.assertEqual(written["status"], "open")
        path = self.record_path()
        self.assertEqual(mode(path), 0o600)
        self.assertEqual(records.read_record("state.json", SESSION), written)
        self.assertEqual(list(path.parent.glob("*.tmp")), [])
        self.assertEqual(records.validate_record(written), [])

    def test_write_record_rejects_unusable_content(self) -> None:
        with self.assertRaises(records.RecordError):
            records.write_record("state.json", SESSION, {"bad": {1, 2}})
        with self.assertRaises(records.RecordError):
            records.write_record("state.json", SESSION, {"nested": {object(): 1}})  # type: ignore[dict-item]

    def test_write_record_replaces_previous_content(self) -> None:
        records.write_record("state.json", SESSION, {"a": 1, "b": 2})
        records.write_record("state.json", SESSION, {"a": 3})
        self.assertEqual(records.read_record("state.json", SESSION)["a"], 3)
        self.assertNotIn("b", records.read_record("state.json", SESSION))


class RevisionAndDigestTests(WrapperHomeMixin):
    def test_revision_of_is_fail_closed(self) -> None:
        self.assertEqual(records.revision_of({}), 0)
        self.assertEqual(records.revision_of({"record_revision": 9}), 9)
        for bad in ("3", 1.5, True, -1, None, []):
            with self.subTest(bad=bad):
                self.assertEqual(records.revision_of({"record_revision": bad}), 0)

    def test_digest_of_is_stable_and_order_sensitive(self) -> None:
        first = records.digest_of("op", {"b": 1, "a": 2})
        self.assertEqual(first, records.digest_of("op", {"a": 2, "b": 1}))
        self.assertNotEqual(first, records.digest_of("op", {"a": 2}))
        self.assertEqual(len(first), 64)
        self.assertRegex(first, r"^[0-9a-f]{64}$")
        self.assertEqual(records.digest_of(1, 2), records.digest_of(1, 2))
        self.assertNotEqual(records.digest_of(1, 2), records.digest_of(2, 1))
        self.assertEqual(records.digest_of(object()), records.digest_of(object()))


class GuardedUpdateTests(WrapperHomeMixin):
    def test_bumps_revision_and_stamps_fields(self) -> None:
        first = records.guarded_update(
            "state.json", SESSION, lambda current: {**current, "status": "open"}, operation="open_state"
        )
        second = records.guarded_update(
            "state.json", SESSION, lambda current: {**current, "status": "working"}, operation="transition"
        )
        self.assertFalse(first.replayed)
        self.assertEqual((first.revision, second.revision), (1, 2))
        self.assertEqual(second.record["status"], "working")
        self.assertEqual(second.record["record_revision"], 2)
        self.assertEqual(records.validate_record(second.record), [])
        self.assertEqual(mode(self.record_path()), 0o600)

    def test_expected_revision_refuses_stale_writes(self) -> None:
        records.guarded_update("state.json", SESSION, lambda current: dict(current), operation="op")
        with self.assertRaises(records.RecordError):
            records.guarded_update(
                "state.json", SESSION, lambda current: dict(current), operation="op", expected_revision=7
            )
        with self.assertRaises(records.RecordError):
            records.guarded_update(
                "state.json", SESSION, lambda current: dict(current), operation="op", expected_revision="1"  # type: ignore[arg-type]
            )
        ok = records.guarded_update(
            "state.json", SESSION, lambda current: dict(current), operation="op", expected_revision=1
        )
        self.assertEqual(ok.revision, 2)

    def test_replay_is_idempotent_per_operation(self) -> None:
        digest = records.digest_of("transition", "working")
        first = records.guarded_update(
            "state.json",
            SESSION,
            lambda current: {**current, "status": "working"},
            operation="transition",
            mutation_id="m1",
            mutation_digest=digest,
        )
        replay = records.guarded_update(
            "state.json",
            SESSION,
            lambda current: {**current, "status": "cancelled"},
            operation="transition",
            mutation_id="m1",
            mutation_digest=digest,
        )
        self.assertEqual(first.revision, 1)
        self.assertTrue(replay.replayed)
        self.assertEqual(replay.revision, 1)
        self.assertEqual(replay.record["status"], "working")
        other = records.guarded_update(
            "state.json",
            SESSION,
            lambda current: {**current, "status": "done"},
            operation="close",
            mutation_id="m1",
            mutation_digest=records.digest_of("close", "done"),
        )
        self.assertFalse(other.replayed)
        self.assertEqual(other.revision, 2)

    def test_replay_with_different_digest_is_refused(self) -> None:
        records.guarded_update(
            "state.json",
            SESSION,
            lambda current: dict(current),
            operation="transition",
            mutation_id="m1",
            mutation_digest=records.digest_of("transition", "working"),
        )
        with self.assertRaises(records.RecordError) as caught:
            records.guarded_update(
                "state.json",
                SESSION,
                lambda current: dict(current),
                operation="transition",
                mutation_id="m1",
                mutation_digest=records.digest_of("transition", "cancelled"),
            )
        self.assertIn("refusing to replay", str(caught.exception))
        self.assertEqual(records.revision_of(records.read_record("state.json", SESSION)), 1)

    def test_applied_mutations_are_bounded(self) -> None:
        def write(current: dict) -> dict:
            return dict(current)

        for index in range(records.MAX_APPLIED_MUTATIONS + 12):
            records.guarded_update(
                "state.json",
                SESSION,
                write,
                operation="tick",
                mutation_id=f"m{index}",
                mutation_digest=records.digest_of("tick", index),
            )
        record = records.read_record("state.json", SESSION)
        entries = record["applied_mutations"]
        self.assertEqual(len(entries), records.MAX_APPLIED_MUTATIONS)
        self.assertEqual(entries[-1]["mutation_id"], f"m{records.MAX_APPLIED_MUTATIONS + 11}")

    def test_mutate_contract_and_validate_hook(self) -> None:
        with self.assertRaises(records.RecordError):
            records.guarded_update("state.json", SESSION, lambda current: [], operation="op")  # type: ignore[return-value]
        with self.assertRaises(records.RecordError):
            records.guarded_update("state.json", SESSION, lambda current: dict(current), operation="   ")
        with self.assertRaises(records.RecordError):
            records.guarded_update("state.json", SESSION, "not-callable", operation="op")  # type: ignore[arg-type]

        def reject(record: dict) -> None:
            raise records.RecordError("business rule refused")

        with self.assertRaises(records.RecordError) as caught:
            records.guarded_update(
                "state.json", SESSION, lambda current: dict(current), operation="op", validate=reject
            )
        self.assertIn("business rule refused", str(caught.exception))
        self.assertEqual(records.read_record("state.json", SESSION), {})

    def test_mutator_exceptions_propagate_unwritten(self) -> None:
        records.guarded_update("state.json", SESSION, lambda current: {**current, "a": 1}, operation="op")
        before = self.record_path().read_bytes()

        def boom(_current: dict) -> dict:
            raise ValueError("mutate blew up")

        with self.assertRaises(ValueError):
            records.guarded_update("state.json", SESSION, boom, operation="op")
        self.assertEqual(self.record_path().read_bytes(), before)

    def test_corrupt_record_self_heals(self) -> None:
        self.record_path().write_text("{ torn", encoding="utf-8")
        result = records.guarded_update(
            "state.json", SESSION, lambda current: {**current, "status": "open"}, operation="op"
        )
        self.assertEqual(result.revision, 1)
        self.assertEqual(result.record["status"], "open")

    def test_invalid_candidate_is_refused_before_write(self) -> None:
        with self.assertRaises(records.RecordError):
            records.guarded_update(
                "state.json", SESSION, lambda current: {**current, "session_id": "../evil"}, operation="op"
            )
        with self.assertRaises(records.RecordError):
            records.guarded_update(
                "state.json", SESSION, lambda current: {**current, "blob": object()}, operation="op"
            )
        self.assertEqual(records.read_record("state.json", SESSION), {})


class EventTests(WrapperHomeMixin):
    def test_append_and_read_events(self) -> None:
        first = records.append_event(SESSION, "session_started", {"status": "open"})
        second = records.append_event(SESSION, "status_working", {"from": "open", "to": "working"})
        self.assertEqual((first["seq"], second["seq"]), (1, 2))
        self.assertEqual(first["event"], "session_started")
        self.assertEqual(first["schema_version"], records.SCHEMA_VERSION)
        events, errors = records.read_events(SESSION)
        self.assertEqual(errors, [])
        self.assertEqual([item["event"] for item in events], ["session_started", "status_working"])
        self.assertEqual(events[1]["data"], {"from": "open", "to": "working"})
        self.assertEqual(mode(records.session_dir(SESSION) / records.EVENTS_FILE), 0o600)

    def test_events_are_one_clean_line(self) -> None:
        item = records.append_event(
            SESSION,
            "  notice\x1b[31m red\x07\n  second line  ",
            {"detail": "\x1b[0mmulti\nline\x00 value", "code": 3, "ratio": 1.5, "flag": True, "none": None},
        )
        self.assertEqual(item["event"], "notice red second line")
        self.assertEqual(item["data"]["detail"], "multi line value")
        self.assertEqual(item["data"]["code"], 3)
        raw = (records.session_dir(SESSION) / records.EVENTS_FILE).read_text(encoding="utf-8")
        self.assertEqual(len(raw.strip().splitlines()), 1)

    def test_empty_event_name_is_refused(self) -> None:
        for bad in ("", "   ", "\n", "\x1b[0m", None):
            with self.subTest(bad=bad):
                with self.assertRaises(records.RecordError):
                    records.append_event(SESSION, bad, {})  # type: ignore[arg-type]

    def test_oversized_payload_collapses_instead_of_growing(self) -> None:
        item = records.append_event(SESSION, "big", {f"k{i}": "x" * 500 for i in range(40)})
        self.assertTrue(item["data"]["truncated"])
        self.assertEqual(item["data"]["detail"].count("k0"), 1)
        events, errors = records.read_events(SESSION)
        self.assertEqual(errors, [])
        self.assertEqual(events[0]["event"], "big")

    def test_rotation_keeps_the_newest_events(self) -> None:
        with mock.patch.object(records, "MAX_EVENTS", 6):
            for index in range(10):
                records.append_event(SESSION, f"event-{index}", {})
            events, errors = records.read_events(SESSION, limit=6)
        self.assertEqual(errors, [])
        self.assertEqual([item["event"] for item in events], [f"event-{index}" for index in range(4, 10)])
        self.assertEqual(events[-1]["seq"], 10)

    def test_read_events_reports_parse_errors_and_bounds_limit(self) -> None:
        records.append_event(SESSION, "ok-1", {})
        path = records.session_dir(SESSION) / records.EVENTS_FILE
        with path.open("a", encoding="utf-8") as handle:
            handle.write("{ corrupt\n")
            handle.write('"not an object"\n')
        records.append_event(SESSION, "ok-2", {})
        events, errors = records.read_events(SESSION, limit=50)
        self.assertEqual([item["event"] for item in events], ["ok-1", "ok-2"])
        self.assertEqual(len(errors), 2)
        self.assertIn(":2:", errors[0])
        self.assertIn("not a JSON object", errors[1])
        recent, _ = records.read_events(SESSION, limit=1)
        self.assertEqual([item["event"] for item in recent], ["ok-2"])
        for bad_limit in (0, -5, "many", None, True):
            clipped, _ = records.read_events(SESSION, limit=bad_limit)  # type: ignore[arg-type]
            self.assertLessEqual(len(clipped), records.MAX_EVENTS)
        missing_events, missing_errors = records.read_events("never-used")
        self.assertEqual((missing_events, missing_errors), ([], []))


class CleanTextTests(WrapperHomeMixin):
    def test_single_line_and_bounds(self) -> None:
        self.assertEqual(records.clean_text(None), "")
        self.assertEqual(records.clean_text(""), "")
        self.assertEqual(records.clean_text("a\r\nb\rc\nd"), "a b c d")
        self.assertEqual(records.clean_text("\x1b[31mred\x1b[0m text"), "red text")
        self.assertEqual(records.clean_text("tab\tseparated"), "tab separated")
        self.assertEqual(records.clean_text("x" * 10, limit=4), "xxx…")
        self.assertEqual(records.clean_text("日本語です", limit=5), "日本…")
        self.assertEqual(records.clean_text("anything", limit=0), "")
        self.assertEqual(records.clean_text("anything", limit=-1), "")
        self.assertEqual(records.clean_text(12), "12")
        self.assertEqual(len(records.clean_text("y" * 5000)), records.MAX_EVENT_CHARS)

    def test_multiline_mode_keeps_newlines(self) -> None:
        kept = records.clean_text("first\nsecond\r\nthird\x07", single_line=False)
        self.assertEqual(kept, "first\nsecond\nthird")
        self.assertEqual(records.clean_text("\x1b[31ma\nb"), "a b")


class ValidateRecordTests(WrapperHomeMixin):
    def test_accepts_a_written_record(self) -> None:
        written = records.write_record("state.json", SESSION, {"record_revision": 0})
        self.assertEqual(records.validate_record(written), [])

    def test_reports_every_problem(self) -> None:
        errors = records.validate_record(
            {
                "schema_version": 2,
                "session_id": "../etc",
                "record_revision": -3,
                "updated_at": "yesterday",
                "applied_mutations": {"operation": "x"},
            }
        )
        joined = "\n".join(errors)
        self.assertIn("schema_version", joined)
        self.assertIn("session_id", joined)
        self.assertIn("record_revision", joined)
        self.assertIn("updated_at", joined)
        self.assertIn("applied_mutations must be a list", joined)

    def test_entry_shape_and_size(self) -> None:
        too_many = [
            {"operation": "op", "mutation_id": f"m{i}", "digest": "d"}
            for i in range(records.MAX_APPLIED_MUTATIONS + 1)
        ]
        self.assertIn(
            "exceeds",
            "\n".join(records.validate_record({"schema_version": 1, "applied_mutations": too_many})),
        )
        bad_entry = records.validate_record({"schema_version": 1, "applied_mutations": [{"operation": "op"}]})
        self.assertEqual(len(bad_entry), 2)
        self.assertIn("mutation_id", bad_entry[0])
        self.assertIn("digest", bad_entry[1])
        self.assertEqual(
            records.validate_record({"schema_version": 1, "applied_mutations": ["nope"]}),
            ["record applied_mutations[0] must be an object."],
        )

    def test_non_mapping_and_non_serializable(self) -> None:
        self.assertEqual(records.validate_record([]), ["Wrapper record must be a JSON object."])
        self.assertEqual(records.validate_record(None), ["Wrapper record must be a JSON object."])
        errors = records.validate_record({"schema_version": 1, "float": float("nan"), "obj": object()})
        self.assertEqual(len(errors), 1)
        self.assertIn("not JSON-serializable", errors[0])


if __name__ == "__main__":
    unittest.main()
