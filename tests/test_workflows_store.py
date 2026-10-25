from __future__ import annotations

import json
import os
import stat
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from ohar.workflows.store import (
    MAX_JSON_DEPTH,
    MAX_LINE_BYTES,
    MAX_STORE_BYTES,
    WORKFLOWS_DIRNAME,
    RecordStore,
    StaleRevisionError,
    WorkflowError,
    canonical_digest,
    clean_text,
    new_record_id,
    now_iso,
    opaque_ref,
    read_json_bounded,
    short_digest,
    stamp_to_epoch,
    validate_vocabularies,
    workspace_key,
    workflows_root,
    write_json_atomic,
)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.ws = self.base / "ws"
        self.ws.mkdir()
        self.root = self.base / "workflows"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def store(self, name: str = "gates", workspace: Path | None = None) -> RecordStore:
        return RecordStore(self.root, name, workspace=workspace if workspace is not None else self.ws)

    def test_path_is_workspace_scoped_and_private(self) -> None:
        store = self.store()
        self.assertEqual(store.path(), self.root / workspace_key(self.ws) / "gates.jsonl")
        other = RecordStore(self.root, "gates", workspace=self.base / "other")
        self.assertNotEqual(store.path().parent, other.path().parent)
        store.append({"gate_id": "g1"})
        self.assertEqual(stat.S_IMODE(store.path().stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(store.path().parent.stat().st_mode), 0o700)

    def test_name_cannot_become_a_path(self) -> None:
        sneaky = RecordStore(self.root, "../../escape", workspace=self.ws)
        self.assertEqual(sneaky.path().name, "escape.jsonl")
        self.assertRegex(sneaky.path().name, r"^[a-z0-9_-]+\.jsonl$")
        self.assertTrue(sneaky.path().is_relative_to(self.root / workspace_key(self.ws)))
        self.assertEqual(RecordStore(self.root, "///", workspace=self.ws).path().name, "records.jsonl")

    def test_append_fills_defaults_and_records_are_ordered(self) -> None:
        store = self.store()
        stored = store.append({"gate_id": "g1", "state": "open"})
        self.assertTrue({"gate_id", "state", "record_id", "at", "schema_version"} <= set(stored))
        self.assertRegex(stored["record_id"], r"^gates-\d{14}-[0-9a-f]{6}$")
        self.assertIsNotNone(stamp_to_epoch(stored["at"]))
        store.append({"gate_id": "g2"})
        store.append({"gate_id": "g3"})
        records = store.records()
        self.assertEqual([item["gate_id"] for item in records], ["g1", "g2", "g3"])
        self.assertEqual([item["gate_id"] for item in store.records(limit=2)], ["g2", "g3"])
        self.assertEqual(store.records(limit=0), [])

    def test_records_skips_damaged_lines(self) -> None:
        store = self.store("journal")
        store.append({"event": "tool_observed"})
        with open(store.path(), "a", encoding="utf-8") as handle:
            handle.write("{not json}\n\n[]\n")
        store.append({"event": "turn_completed"})
        self.assertEqual(len(store.records()), 2)

    def test_append_refuses_an_oversized_line(self) -> None:
        store = self.store()
        with self.assertRaises(WorkflowError):
            store.append({"gate_id": "g1", "blob": "x" * (MAX_LINE_BYTES + 10)})
        self.assertFalse(store.path().exists())

    def test_answers_supersede_questions_without_editing_disk(self) -> None:
        store = self.store()
        store.append({"gate_id": "g1", "state": "open", "question": "allow_write"})
        store.append({"gate_id": "g1", "state": "answered", "choice": "approve"})
        self.assertEqual(store.latest("gate_id", "g1")["state"], "answered")
        heads = store.heads("gate_id")
        self.assertEqual(list(heads), ["g1"])
        self.assertEqual(len(store.records()), 2)
        self.assertIsNone(store.latest("gate_id", "missing"))

    def test_heads_ignores_records_without_the_identity_key(self) -> None:
        store = self.store()
        store.append({"gate_id": "g1"})
        store.append({"note": "no identity"})
        store.append({"gate_id": ""})
        self.assertEqual(list(store.heads("gate_id")), ["g1"])

    def test_transact_bumps_revision_and_refuses_stale_reads(self) -> None:
        store = self.store("goals")
        first, replayed = store.transact("goal_id", "goal-1", lambda current: {"objective": "ship"})
        self.assertFalse(replayed)
        self.assertEqual(first["revision"], 1)
        second, _ = store.transact(
            "goal_id", "goal-1", lambda current: {"objective": current["objective"], "n": 1}, expected_revision=1
        )
        self.assertEqual(second["revision"], 2)
        with self.assertRaises(StaleRevisionError):
            store.transact("goal_id", "goal-1", lambda current: {"objective": "late"}, expected_revision=1)
        self.assertEqual(store.latest("goal_id", "goal-1")["revision"], 2)

    def test_transact_mutation_id_replays_without_writing(self) -> None:
        store = self.store("goals")

        def mutate(current: dict | None) -> dict:
            return {"objective": "ship"}

        record, replayed = store.transact("goal_id", "g", mutate, mutation_id="m-1", mutation_digest="d-1")
        self.assertFalse(replayed)
        again, replayed = store.transact("goal_id", "g", mutate, mutation_id="m-1", mutation_digest="d-1")
        self.assertTrue(replayed)
        self.assertEqual(again, record)
        self.assertEqual(len(store.records()), 1)
        with self.assertRaises(WorkflowError):
            store.transact("goal_id", "g", mutate, mutation_id="m-1", mutation_digest="other")

    def test_transact_identity_is_stable_and_a_noop_mutate_writes_nothing(self) -> None:
        store = self.store("units")
        self.assertEqual(store.transact("unit_id", "u1", lambda current: None), ({}, False))
        self.assertEqual(store.records(), [])
        record, _ = store.transact("unit_id", "u1", lambda current: {"statement": "tests pass"})
        self.assertEqual(record["unit_id"], "u1")
        with self.assertRaises(WorkflowError):
            store.transact("unit_id", "u1", lambda current: {"unit_id": "other"})
        with self.assertRaises(WorkflowError):
            store.transact("unit_id", "", lambda current: {})

    def test_only_one_surface_wins_a_contended_transaction(self) -> None:
        store = self.store("goals")
        barrier = threading.Barrier(3)
        results: list[str] = []

        def worker() -> None:
            barrier.wait()
            try:
                store.transact("goal_id", "g1", lambda current: {"who": threading.current_thread().name}, expected_revision=0)
                results.append("won")
            except StaleRevisionError:
                results.append("stale")

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join()
        self.assertEqual(sorted(results), ["stale", "won"])
        self.assertEqual(len(store.records()), 1)

    def test_concurrent_appends_keep_every_line_intact(self) -> None:
        store = self.store("activity")
        barrier = threading.Barrier(4)

        def worker(index: int) -> None:
            barrier.wait()
            for step in range(10):
                store.append({"worker": index, "step": step, "pad": "y" * 200})

        threads = [threading.Thread(target=worker, args=(index,)) for index in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        records = store.records()
        self.assertEqual(len(records), 40)
        self.assertEqual({item["worker"] for item in records}, {0, 1, 2, 3})
        lines = store.path().read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 40)
        self.assertTrue(all(isinstance(json.loads(line), dict) for line in lines))

    def test_workflows_root_is_the_private_home_directory(self) -> None:
        with mock.patch.dict(os.environ, {"OHAR_HOME": str(self.base / "home")}):
            root = workflows_root()
            self.assertEqual(root, self.base / "home" / WORKFLOWS_DIRNAME)
            self.assertEqual(workflows_root(self.ws), root / workspace_key(self.ws))
            self.assertTrue(root.is_dir())
            self.assertEqual(stat.S_IMODE(root.stat().st_mode), 0o700)

    def test_json_head_record_round_trip(self) -> None:
        path = self.root / workspace_key(self.ws) / "goal.json"
        write_json_atomic(path, {"goal_id": "g1", "criteria": [{"id": "AC001"}]})
        self.assertEqual(read_json_bounded(path)["criteria"], [{"id": "AC001"}])
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        write_json_atomic(path, {"goal_id": "g1", "revision": 2})
        self.assertEqual(read_json_bounded(path)["revision"], 2)
        self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_json_reads_are_bounded(self) -> None:
        path = self.root / "plain.json"
        write_json_atomic(path, {"a": 1})
        with self.assertRaises(WorkflowError):
            read_json_bounded(path, max_bytes=4)
        with self.assertRaises(WorkflowError):
            read_json_bounded(self.root / "absent.json")
        hostile = self.root / "hostile.json"
        hostile.write_text("[" * (MAX_JSON_DEPTH + 8) + "]" * (MAX_JSON_DEPTH + 8))
        with self.assertRaises(WorkflowError):
            read_json_bounded(hostile)
        broken = self.root / "broken.json"
        broken.write_text("{oops")
        with self.assertRaises(WorkflowError):
            read_json_bounded(broken)
        array = self.root / "array.json"
        array.write_text("[1, 2]")
        with self.assertRaises(WorkflowError):
            read_json_bounded(array)

    def test_write_json_atomic_rejects_unusable_payloads(self) -> None:
        path = self.root / "head.json"
        with self.assertRaises(WorkflowError):
            write_json_atomic(path, ["not", "a", "mapping"])  # type: ignore[arg-type]
        with self.assertRaises(WorkflowError):
            write_json_atomic(path, {"blob": "x" * (MAX_STORE_BYTES + 1)})

    def test_clean_text_strips_ansi_and_bounds(self) -> None:
        self.assertEqual(clean_text("\x1b[31mred\x1b[0m line"), "red line")
        self.assertEqual(clean_text("a\x00b\x1fc"), "abc")
        self.assertEqual(clean_text("multi\n\nline", single_line=False), "multi\n\nline")
        self.assertEqual(clean_text(None), "")
        self.assertEqual(clean_text(7), "7")
        clipped = clean_text("word " * 100, limit=24)
        self.assertLessEqual(len(clipped), 24)
        self.assertTrue(clipped.endswith("…"))
        with self.assertRaises(WorkflowError):
            clean_text("x", limit=True)

    def test_opaque_refs_are_non_navigable(self) -> None:
        self.assertEqual(opaque_ref("unit:ab12cd34"), "unit:ab12cd34")
        self.assertTrue(opaque_ref("/etc/passwd").startswith("ref:"))
        self.assertTrue(opaque_ref("a/../b").startswith("ref:"))
        self.assertTrue(opaque_ref("token=super-secret").startswith("ref:"))
        self.assertTrue(opaque_ref("secret: hunter2").startswith("ref:"))
        self.assertNotIn("hunter2", opaque_ref("secret: hunter2"))
        self.assertEqual(opaque_ref(""), "")
        self.assertLessEqual(len(opaque_ref("plain " * 40, limit=32)), 32)

    def test_identifiers_and_digests(self) -> None:
        self.assertEqual(canonical_digest({"b": 1, "a": 2}), canonical_digest({"a": 2, "b": 1}))
        self.assertEqual(len(canonical_digest("x")), 64)
        self.assertEqual(short_digest("x"), canonical_digest("x")[:12])
        self.assertEqual(len(short_digest("x", 8)), 8)
        with self.assertRaises(WorkflowError):
            short_digest("x", length="8")
        ids = {new_record_id("goal ledger!") for _ in range(5)}
        self.assertEqual(len(ids), 5)
        for identifier in ids:
            self.assertRegex(identifier, r"^goal-ledger-\d{14}-[0-9a-f]{6}$")
        self.assertRegex(new_record_id(""), r"^rec-\d{14}-[0-9a-f]{6}$")
        self.assertTrue(new_record_id("../x").startswith("x-"))
        self.assertEqual(len(workspace_key(self.ws)), 16)
        self.assertEqual(workspace_key(self.ws), workspace_key(Path(f"{self.ws}/")))

    def test_timestamp_helpers(self) -> None:
        stamp = now_iso()
        self.assertIsNotNone(stamp_to_epoch(stamp))
        self.assertIsNone(stamp_to_epoch("yesterday"))
        self.assertIsNone(stamp_to_epoch(""))
        self.assertIsNotNone(stamp_to_epoch(stamp.replace("+00:00", "Z")))

    def test_validate_vocabularies_closes_the_set(self) -> None:
        record = {"state": "answered", "choices": ["approve", "approve"]}
        self.assertEqual(
            validate_vocabularies(record, [("state", ("open", "answered")), ("choices", ("approve", "decline"))]),
            [],
        )
        errors = validate_vocabularies({"state": "weird"}, [("state", ("open", "answered"))])
        self.assertEqual(len(errors), 1)
        self.assertIn("weird", errors[0])
        self.assertEqual(len(validate_vocabularies({}, [("risk", ("low",))])), 1)
        self.assertEqual(len(validate_vocabularies({"choices": ["nope"]}, [("choices", ("approve",))])), 1)
        self.assertEqual(len(validate_vocabularies({"state": 3}, [("state", ("open",))])), 1)


if __name__ == "__main__":
    unittest.main()
