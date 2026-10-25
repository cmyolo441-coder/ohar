from __future__ import annotations

import hashlib
import json
import stat
import tempfile
import unittest
from pathlib import Path

from ohar.memory import MemoryStore, make_memory_tools
from ohar.types import ToolContext


class MemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.root = self.base / "memory"
        self.ws = self.base / "ws"
        self.ws.mkdir()
        self.store = MemoryStore(self.root, self.ws)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def ctx(self, **kwargs: object) -> ToolContext:
        return ToolContext(workspace=self.ws, **kwargs)  # type: ignore[arg-type]

    def tools(self, *, require_approval: bool = True) -> dict:
        return {t.name: t for t in make_memory_tools(self.store, require_approval=require_approval)}

    def test_file_layout_and_ranking(self) -> None:
        digest = hashlib.sha256(str(self.ws.resolve()).encode()).hexdigest()[:16]
        self.assertEqual(self.store.path, self.root / f"{digest}.jsonl")
        a = self.store.remember("Run pytest before deploys", ["ci", "deploy"])
        b = self.store.remember("Python typing notes for the parser", ["python"])
        self.store.remember("Unrelated grocery list", [])
        self.assertEqual(set(a), {"schema_version", "id", "text", "tags", "created_at", "workspace"})
        self.assertRegex(a["id"], r"^m-[0-9a-f]{8}$")
        found = self.store.recall("python typing")
        self.assertEqual([r["id"] for r in found], [b["id"]])
        self.assertEqual(self.store.recall("deploy")[0]["id"], a["id"])
        self.assertEqual(self.store.recall("", 1)[0]["text"], "Unrelated grocery list")
        self.assertEqual(stat.S_IMODE(self.store.path.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(self.root.stat().st_mode), 0o700)

    def test_tags_trimming_limits_and_copies(self) -> None:
        tags = [f"  TAG-{i}-{'x' * 60}  " for i in range(15)]
        record = self.store.remember("\x1b[31malpha\x00 " + "z" * 5_000, tags)
        self.assertEqual(len(record["text"]), 4_000)
        self.assertNotIn("\x1b", record["text"])
        self.assertNotIn("\x00", record["text"])
        self.assertEqual(len(record["tags"]), 12)
        self.assertTrue(all(len(tag) <= 40 for tag in record["tags"]))
        self.assertEqual(self.store.recall("tag-0")[0]["id"], record["id"])
        record["tags"].append("mutated")
        record["text"] = "mutated"
        again = self.store.recent(1)[0]
        self.assertNotIn("mutated", again["tags"])
        self.assertIn("alpha", again["text"])
        self.assertEqual(self.store.recall("alpha")[0]["id"], record["id"])

    def test_max_records_drops_oldest(self) -> None:
        store = MemoryStore(self.root, self.ws, max_records=2)
        first = store.remember("first")
        second = store.remember("second")
        third = store.remember("third")
        self.assertEqual([r["id"] for r in store.recent(10)], [third["id"], second["id"]])
        self.assertEqual(store.forget(first["id"]), 0)
        lines = store.path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual([json.loads(line)["id"] for line in lines], [second["id"], third["id"]])

    def test_forget_prefix_ambiguity_and_full_id(self) -> None:
        a = self.store.remember("alpha")
        b = self.store.remember("beta")
        with self.assertRaises(ValueError):
            self.store.forget("m-")
        prefix = next(a["id"][:n] for n in range(3, len(a["id"]) + 1) if not b["id"].startswith(a["id"][:n]))
        self.assertEqual(self.store.forget(prefix), 1)
        self.assertEqual(len(self.store.recent()), 1)
        self.assertEqual(self.store.forget(b["id"]), 1)
        self.assertEqual(self.store.forget(b["id"]), 0)
        self.assertEqual(self.store.forget("nope"), 0)

    def test_clear_and_corrupt_lines(self) -> None:
        self.store.remember("keep")
        self.store.remember("also keep")
        with self.store.path.open("a", encoding="utf-8") as handle:
            handle.write('not json\n{"schema_version": 1}\n')
            handle.write('{"schema_version": 1, "id": "bad"}\n')
        self.assertEqual(len(self.store.recent()), 2)
        self.store.remember("survivor")
        self.assertEqual(self.store.clear(), 3)
        self.assertEqual(self.store.path.read_bytes(), b"")
        self.assertEqual(stat.S_IMODE(self.store.path.stat().st_mode), 0o600)
        self.assertEqual(self.store.recent(), [])
        self.assertEqual(self.store.clear(), 0)

    def test_workspace_isolation(self) -> None:
        other_ws = self.base / "other"
        other_ws.mkdir()
        other = MemoryStore(self.root, other_ws)
        mine = self.store.remember("shared token mine", ["mine"])
        theirs = other.remember("shared token theirs", ["theirs"])
        self.assertNotEqual(self.store.path, other.path)
        self.assertEqual([r["id"] for r in self.store.recall("shared")], [mine["id"]])
        self.assertEqual([r["id"] for r in other.recall("shared")], [theirs["id"]])
        self.assertEqual(self.store.forget(theirs["id"]), 0)
        self.assertEqual(len(other.recent()), 1)

    def test_tools_approval_denial_and_read_only(self) -> None:
        tools = self.tools()
        self.assertEqual(list(tools), ["remember", "recall", "forget"])
        self.assertTrue(tools["recall"].read_only)
        self.assertFalse(tools["remember"].read_only)
        calls = []
        ctx = self.ctx(approval_mode="ask", confirm=lambda request: calls.append(request) or False)
        denied = tools["remember"].handler({"text": "do not store"}, ctx)
        self.assertFalse(denied.ok)
        self.assertTrue(denied.meta.get("denied"))
        self.assertEqual(self.store.recent(), [])
        ctx.confirm = lambda request: calls.append(request) or True
        saved = tools["remember"].handler({"text": "store this", "tags": "a, b"}, ctx)
        self.assertTrue(saved.ok)
        self.assertIn(saved.meta["id"], saved.content)
        self.assertEqual(calls[1].action, "remember")
        self.assertEqual(calls[1].risk, "low")
        self.assertIn("store this", calls[1].summary)
        self.assertEqual(calls[1].detail, "a, b")
        memory_id = saved.meta["id"]
        deny_ctx = self.ctx(approval_mode="ask", confirm=lambda request: False)
        self.assertFalse(tools["forget"].handler({"id": memory_id}, deny_ctx).ok)
        self.assertEqual(len(self.store.recent()), 1)
        self.assertFalse(tools["remember"].handler(
            {"text": "read only"}, self.ctx(approval_mode="auto", read_only=True)
        ).ok)
        result = tools["recall"].handler({"query": "store", "limit": 100}, self.ctx())
        self.assertTrue(result.ok)
        self.assertIn(memory_id, result.content)
        self.assertIn("tags: a, b", result.content)
        self.assertEqual(tools["recall"].handler({"query": "zzz"}, self.ctx()).content, "No memories.")
        auto = self.tools(require_approval=False)
        self.assertTrue(auto["remember"].handler({"text": "no prompt"}, self.ctx(approval_mode="ask")).ok)

    def test_recent_and_recall_limits_and_no_mutation(self) -> None:
        records = [self.store.remember(f"note {i}") for i in range(25)]
        self.assertEqual(len(self.store.recall("note", 100)), 20)
        self.assertEqual(len(self.store.recall("note", 0)), 1)
        self.assertEqual(len(self.store.recent(3)), 3)
        before = self.store.path.read_bytes()
        self.store.recall("note", 20)
        self.assertEqual(self.store.path.read_bytes(), before)
        self.assertEqual(self.store.recent(1)[0]["id"], records[-1]["id"])


if __name__ == "__main__":
    unittest.main()
