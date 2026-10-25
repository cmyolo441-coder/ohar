from __future__ import annotations

import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ohar.config import new_config
from ohar.skills import SkillLibrary, make_skill_tool
from ohar.terminal import MarkdownStream, banner_lines, make_theme, strip_ansi, table
from ohar.tools import Toolbox
from ohar.types import ApprovalRequest, Config, ToolContext, UIConfig
from ohar.ui import UI

PROJECT = Path(__file__).resolve().parent.parent


class TerminalTests(unittest.TestCase):
    def test_theme_disabled_is_plain(self) -> None:
        theme = make_theme("aurora", color=False)
        self.assertEqual(theme.paint("hello", fg="accent", bold=True), "hello")
        self.assertEqual(theme.gradient("OHAR"), "OHAR")

    def test_markdown_lines_and_ansi_cleanup(self) -> None:
        theme = make_theme("aurora", color=False)
        stream = MarkdownStream(theme, enabled=True, width=80)
        lines = stream.feed("# Title\n\n- item **bold**\n\n```py\nprint(1)\n```\n")
        self.assertEqual(lines[0], "  ▌ Title")
        self.assertEqual(lines[1], "")
        self.assertEqual(lines[2], "  • item bold")
        self.assertIn("    │ py", lines)
        self.assertIn("    │ print(1)", lines)
        self.assertEqual(stream.flush(), [])
        stream.feed("\x1b[31mred\x1b[0m")
        flushed = stream.flush()
        self.assertEqual(flushed, ["  red"])

    def test_syntax_highlighting_colors_without_changing_text(self) -> None:
        theme = make_theme("aurora", color=True, depth="truecolor")
        source = "```python\ndef hello(name: int):  # note\n    return name\n```\n"
        highlighted = MarkdownStream(theme, enabled=True, width=80, syntax=True).feed(source)
        plain = MarkdownStream(theme, enabled=True, width=80, syntax=False).feed(source)
        highlighted_code = [line for line in highlighted if "def hello" in strip_ansi(line)]
        plain_code = [line for line in plain if "def hello" in strip_ansi(line)]
        self.assertEqual(len(highlighted_code), 1)
        self.assertEqual(strip_ansi(highlighted_code[0]), strip_ansi(plain_code[0]))
        self.assertIn("\x1b[", highlighted_code[0])
        self.assertNotEqual(highlighted_code[0], plain_code[0])

    def test_banner_and_table_are_aligned(self) -> None:
        theme = make_theme("mono", color=False)
        lines = banner_lines(
            theme,
            version="0.2.0",
            workspace=str(PROJECT),
            model="openai/test",
            lane="main",
            width=90,
        )
        plain = "\n".join(lines)
        self.assertIn("think", plain)
        self.assertIn("openai/test", plain)
        self.assertNotIn("\x1b", plain)
        rows = table(theme, ["A", "B"], [["one", "two"], ["longer", "x"]])
        self.assertEqual(len(rows), 4)
        self.assertEqual(len({len(strip_ansi(row)) for row in rows}), 1)


class PlainUITests(unittest.TestCase):
    def make_ui(self) -> tuple[UI, io.StringIO, io.StringIO]:
        config = Config(ui=UIConfig(theme="aurora", markdown=True))
        out, inp = io.StringIO(), io.StringIO("hello\n")
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            ui = UI(config, workspace, plain=True, stdout=out, stdin=inp)
        ui.stderr = io.StringIO()
        return ui, out, inp

    def test_plain_stream_suppresses_cards(self) -> None:
        ui, out, _ = self.make_ui()
        ui.startup(model="m", lane="main")
        ui.stream_start()
        ui.text("Hello")
        ui.text(" world\nsecond")
        ui.tool_start("read_file", {"path": "x"})
        ui.tool_end("read_file", type("R", (), {"ok": True, "content": "x", "meta": {}})(), 0.1)
        ui.stream_end()
        self.assertEqual(out.getvalue(), "Hello world\nsecond\n")
        self.assertNotIn("\x1b", out.getvalue())

    def test_prompt_and_notice_paths(self) -> None:
        ui, out, _ = self.make_ui()
        self.assertEqual(ui.read_prompt(), "hello")
        self.assertIsNone(ui.read_prompt())
        ui.notice("visible in plain?", "info")
        ui.notice("warning text", "warning")
        self.assertNotIn("visible", out.getvalue())
        self.assertIn("warning text", ui.stderr.getvalue())
        request = ApprovalRequest(action="shell", summary="run", detail="", risk="high")
        self.assertFalse(ui.confirm(request))
        self.assertIn("denied", ui.stderr.getvalue())


class RawInteractiveUITests(unittest.TestCase):
    def test_markdown_off_still_streams_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = io.StringIO()
            with mock.patch.dict(
                os.environ,
                {"TERM": "dumb", "OHAR_HOME": str(Path(directory) / "home")},
            ):
                config = Config(ui=UIConfig(theme="mono", markdown=False))
                ui = UI(
                    config,
                    Path(directory),
                    plain=False,
                    stdout=out,
                    stdin=io.StringIO(),
                )
                ui.stream_start()
                ui.text("**raw** text\nline2")
                ui.stream_end()
            self.assertIn("**raw** text\nline2\n", out.getvalue())
            self.assertNotIn("\x1b[", out.getvalue())


class AdvancedUITests(unittest.TestCase):
    def test_theme_switch_and_plan_and_table(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = io.StringIO()
            with mock.patch.dict(
                os.environ,
                {"TERM": "dumb", "OHAR_HOME": str(Path(directory) / "home")},
            ):
                config = Config(ui=UIConfig(theme="aurora", markdown=False))
                ui = UI(
                    config,
                    Path(directory),
                    plain=False,
                    stdout=out,
                    stdin=io.StringIO(),
                )
                self.assertTrue(ui.set_theme("ember"))
                self.assertEqual(config.ui.theme, "ember")
                self.assertFalse(ui.set_theme("does-not-exist"))
                ui.plan(
                    {
                        "steps": [
                            {"step": "inspect", "status": "completed"},
                            {"step": "build", "status": "in_progress"},
                            {"step": "verify", "status": "pending"},
                        ],
                        "counts": {"completed": 1},
                    }
                )
            rendered = out.getvalue()
            self.assertIn("1/3 done", rendered)
            self.assertIn("inspect", rendered)
            self.assertIn("build", rendered)

    def test_data_table_plain(self) -> None:
        config = Config(ui=UIConfig(theme="aurora"))
        out = io.StringIO()
        ui = UI(config, Path(tempfile.gettempdir()), plain=True, stdout=out, stdin=io.StringIO())
        ui.data_table("Models", ["Provider", "Model"], [["gwarden", "gemini-4-argon"]])
        text = out.getvalue()
        self.assertIn("Models", text)
        self.assertIn("gemini-4-argon", text)
        self.assertNotIn("\x1b", text)


class SkillTests(unittest.TestCase):
    def test_discovery_and_tool(self) -> None:
        library = SkillLibrary.discover(PROJECT / "skills")
        expected = {
            "accessibility",
            "api",
            "data",
            "debug",
            "docs",
            "frontend",
            "git",
            "onboard",
            "performance",
            "plan",
            "refactor",
            "review",
            "security",
            "test",
            "verify",
            "web",
        }
        self.assertEqual(set(library.names), expected)
        self.assertEqual(library.names, sorted(expected))
        self.assertIn("plan", library.index_text())
        self.assertIsNotNone(library.get("plan"))
        self.assertEqual(library.get("planning").name, "plan")
        tool = make_skill_tool(library)
        with tempfile.TemporaryDirectory() as directory:
            ctx = ToolContext(workspace=Path(directory), approval_mode="auto")
            toolbox = Toolbox(ctx, [tool])
            ok = toolbox.execute("load_skill", {"name": "verify"})
            self.assertTrue(ok.ok)
            self.assertIn("Verify", ok.content)
            missing = toolbox.execute("load_skill", {"name": "nope"})
            self.assertFalse(missing.ok)
            self.assertIn("debug", missing.content)


if __name__ == "__main__":
    unittest.main()
