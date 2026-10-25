from __future__ import annotations

import unittest

from ohar.syntax import TOKEN_KINDS, highlight_line, normalize_language


class NormalizeLanguageTests(unittest.TestCase):
    def test_aliases_and_unknown(self) -> None:
        self.assertEqual(normalize_language("PY"), "python")
        self.assertEqual(normalize_language(".js"), "javascript")
        self.assertEqual(normalize_language("c++"), "cpp")
        self.assertEqual(normalize_language("sh"), "bash")
        self.assertEqual(normalize_language("tsx"), "typescript")
        self.assertIsNone(normalize_language("text"))
        self.assertIsNone(normalize_language(""))
        self.assertIsNone(normalize_language(None))
        self.assertIsNone(normalize_language(123))  # type: ignore[arg-type]


class HighlightLineTests(unittest.TestCase):
    def kinds(self, line: str, language: str) -> dict[str, str]:
        return dict(highlight_line(line, language))

    def assert_roundtrip(self, line: str, language: str | None) -> None:
        tokens = highlight_line(line, language)
        self.assertEqual("".join(text for text, _ in tokens), line)
        self.assertTrue(all(kind in TOKEN_KINDS for _, kind in tokens))
        self.assertNotIn("\x1b", "".join(text for text, _ in tokens))

    def test_python_tokens(self) -> None:
        line = 'def hello(x: int = 42):  # note "quoted"'
        tokens = self.kinds(line, "python")
        self.assertEqual(tokens["def"], "keyword")
        self.assertEqual(tokens["hello"], "function")
        self.assertEqual(tokens["int"], "builtin")
        self.assertEqual(tokens["42"], "number")
        self.assertEqual(tokens['# note "quoted"'], "comment")
        self.assert_roundtrip(line, "python")
        self.assertEqual(self.kinds('name = f"{value!r}"', "py")["f\"{value!r}\""], "string")
        self.assertEqual(self.kinds('"""one line"""', "py")['"""one line"""'], "string")

    def test_javascript_and_typescript_tokens(self) -> None:
        line = 'const data = await fetch("/api"); // call'
        tokens = self.kinds(line, "js")
        self.assertEqual(tokens["const"], "keyword")
        self.assertEqual(tokens["await"], "keyword")
        self.assertEqual(tokens["fetch"], "builtin")
        self.assertEqual(tokens['"/api"'], "string")
        self.assertEqual(tokens["// call"], "comment")
        self.assert_roundtrip(line, "js")
        typescript = self.kinds("interface User { name: string }", "ts")
        self.assertEqual(typescript["interface"], "keyword")
        self.assertEqual(typescript["name"], "property")
        self.assertEqual(typescript["string"], "type")

    def test_json_yaml_toml_and_css(self) -> None:
        json_tokens = self.kinds('{"name": true, "value": null}', "json")
        self.assertEqual(json_tokens['"name"'], "property")
        self.assertEqual(json_tokens["true"], "keyword")
        self.assertEqual(json_tokens["null"], "keyword")
        self.assert_roundtrip('{"name": true, "value": null}', "json")
        self.assertEqual(self.kinds("name: value # note", "yaml")["name"], "property")
        self.assertEqual(self.kinds("name: value # note", "yaml")["# note"], "comment")
        self.assertEqual(self.kinds("port = 8080", "toml")["port"], "property")
        self.assertEqual(self.kinds("color: #fff;", "css")["color"], "property")

    def test_bash_sql_and_diff(self) -> None:
        bash = self.kinds('echo "$HOME" $USER # note', "shell")
        self.assertEqual(bash["echo"], "builtin")
        self.assertEqual(bash['"$HOME"'], "string")
        self.assertEqual(bash["$USER"], "variable")
        self.assertEqual(bash["# note"], "comment")
        self.assert_roundtrip('echo "$HOME" $USER # note', "shell")
        sql = self.kinds("SELECT id FROM users WHERE id = 1; -- note", "sql")
        self.assertEqual(sql["SELECT"], "keyword")
        self.assertEqual(sql["FROM"], "keyword")
        self.assertEqual(sql["1"], "number")
        self.assertEqual(sql["-- note"], "comment")
        self.assertEqual(highlight_line("+added", "diff"), [("+added", "string")])
        self.assertEqual(highlight_line("-removed", "diff"), [("-removed", "comment")])
        self.assertEqual(highlight_line("@@ -1 +1 @@", "diff"), [("@@ -1 +1 @@", "keyword")])

    def test_html_tokens(self) -> None:
        line = '<a class="nav" href="/x">Hi &amp; bye</a><!-- note -->'
        tokens = self.kinds(line, "html")
        self.assertEqual(tokens["a"], "tag")
        self.assertEqual(tokens["class"], "attribute")
        self.assertEqual(tokens['"nav"'], "string")
        self.assertEqual(tokens["<!-- note -->"], "comment")
        self.assert_roundtrip(line, "html")

    def test_unknown_empty_and_long_lines(self) -> None:
        self.assertEqual(highlight_line("hello", None), [("hello", "plain")])
        self.assertEqual(highlight_line("hello", "brainfuck"), [("hello", "plain")])
        self.assertEqual(highlight_line("", "python"), [("", "plain")])
        long_line = "x = 1  # " + "a" * 4_100
        self.assertEqual(highlight_line(long_line, "python"), [(long_line, "plain")])
        self.assertEqual(highlight_line("plain text", "this-language-does-not-exist"), [("plain text", "plain")])

    def test_roundtrip_tricky_inputs(self) -> None:
        lines = [
            "",
            "   ",
            "\tvalue",
            'say "hi" \\ there',
            "emoji 🚀 and ünïcode",
            "/* unterminated",
            "// only a comment",
            "-- sql comment",
            "x = 'unclosed",
            "<div data-x='1' broken",
            "+ new line",
            "???",
        ]
        for language in (
            "python",
            "javascript",
            "typescript",
            "json",
            "jsonc",
            "bash",
            "html",
            "css",
            "toml",
            "yaml",
            "sql",
            "rust",
            "go",
            "java",
            "c",
            "cpp",
            "csharp",
            "diff",
        ):
            for line in lines:
                with self.subTest(language=language, line=line):
                    self.assert_roundtrip(line, language)

    def test_all_supported_languages_emit_known_kinds(self) -> None:
        emitted: set[str] = set()
        samples = {
            "python": 'def f(x: int): return "s" # c',
            "javascript": 'const f = (x) => { return "s"; } // c',
            "typescript": 'interface T { x: string }',
            "json": '{"k": 1, "b": true}',
            "jsonc": '{"k": 1} // c',
            "bash": 'echo "$HOME" # c',
            "html": '<p class="x">text</p>',
            "css": 'a { color: #fff; }',
            "toml": 'key = "value" # c',
            "yaml": 'key: value # c',
            "sql": 'SELECT 1; -- c',
            "rust": 'fn main() { let x: i32 = 1; }',
            "go": 'func main() { var x int = 1 }',
            "java": 'class A { int x = 1; }',
            "c": 'int main(void) { return 0; }',
            "cpp": 'int main() { return 0; }',
            "csharp": 'class A { int x = 1; }',
            "diff": '@@ -1 +1 @@',
        }
        for language, line in samples.items():
            for _, kind in highlight_line(line, language):
                emitted.add(kind)
        self.assertLessEqual(emitted, TOKEN_KINDS)
        for kind in ("keyword", "string", "comment", "number", "function", "builtin", "punctuation", "property", "tag", "attribute"):
            self.assertIn(kind, emitted)


if __name__ == "__main__":
    unittest.main()
