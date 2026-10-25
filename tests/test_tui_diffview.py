from __future__ import annotations

import unittest

from ohar.terminal import display_width, make_theme, strip_ansi
from ohar.tui import diffview
from ohar.tui.diffview import (
    DIFF_ADDED,
    DIFF_CONTEXT,
    DIFF_FILE,
    DIFF_HUNK,
    DIFF_NOTE,
    DIFF_REMOVED,
    PREVIEW_LINES,
    DiffLine,
    compact_file_header,
    has_diff,
    parse_inline_diff,
    render_diff,
)

PLAIN = make_theme("aurora", color=False)
COLOR = make_theme("aurora", color=True, depth="truecolor")
MONO = make_theme("mono", color=True, depth="truecolor")

PAYLOAD = "\n".join(
    [
        "--- a/ohar/tui/dag.py",
        "+++ b/ohar/tui/dag.py",
        "@@ -1,4 +1,5 @@",
        "-import os",
        "+from __future__ import annotations",
        " ",
        " SHUT = '>'",
        "@@ -40,2 +41,3 @@",
        "+# note",
        "-# old",
    ]
)


def kinds(payload: str) -> list[str]:
    return [line.kind for line in parse_inline_diff(payload)]


class DiffLineTests(unittest.TestCase):
    def test_a_line_is_frozen(self) -> None:
        line = DiffLine(DIFF_ADDED, "+x")
        with self.assertRaises(Exception):
            line.text = "+y"  # type: ignore[misc]
        self.assertEqual((line.kind, line.text), ("added", "+x"))

    def test_kind_names_are_the_documented_words(self) -> None:
        self.assertEqual(
            (DIFF_FILE, DIFF_HUNK, DIFF_ADDED, DIFF_REMOVED, DIFF_CONTEXT, DIFF_NOTE),
            ("file", "hunk", "added", "removed", "context", "note"),
        )


class ParseTests(unittest.TestCase):
    def test_the_header_pair_folds_into_one_row(self) -> None:
        lines = parse_inline_diff(PAYLOAD)
        self.assertEqual(lines[0], DiffLine(DIFF_FILE, "ohar/tui/dag.py"))
        self.assertEqual([line.text for line in lines].count("--- a/ohar/tui/dag.py"), 0)

    def test_every_line_is_classified(self) -> None:
        self.assertEqual(
            kinds(PAYLOAD),
            [
                DIFF_FILE,
                DIFF_HUNK,
                DIFF_REMOVED,
                DIFF_ADDED,
                DIFF_CONTEXT,
                DIFF_CONTEXT,
                DIFF_HUNK,
                DIFF_ADDED,
                DIFF_REMOVED,
            ],
        )

    def test_diff_markers_survive(self) -> None:
        lines = parse_inline_diff(PAYLOAD)
        by_kind = {line.kind: line.text for line in lines if line.kind in (DIFF_ADDED, DIFF_REMOVED)}
        self.assertTrue(all(text[:1] in "+-" for text in by_kind.values()))

    def test_padding_is_trimmed_but_the_marker_stays(self) -> None:
        self.assertEqual(parse_inline_diff("@@ -1,1 +1,1 @@\n+added          ")[-1].text, "+added")
        self.assertEqual(parse_inline_diff("@@ -1,1 +1,1 @@\n context       ")[-1].text, " context")

    def test_a_blank_context_line_is_kept(self) -> None:
        lines = parse_inline_diff("@@ -1,2 +1,2 @@\n \n+new")
        self.assertIn(" ", [line.text for line in lines if line.kind == DIFF_CONTEXT])

    def test_ansi_and_control_debris_is_stripped(self) -> None:
        lines = parse_inline_diff("\x1b[31m+added\x1b[0m\x00\r\n-\x07removed")
        self.assertEqual([line.text for line in lines], ["+added", "-removed"])
        self.assertEqual([line.kind for line in lines], [DIFF_ADDED, DIFF_REMOVED])

    def test_a_prejoined_header_reads_as_a_file_row(self) -> None:
        self.assertEqual(kinds("a/old -> b/new\n@@ -1,1 +1,1 @@\n+x"), [DIFF_FILE, DIFF_HUNK, DIFF_ADDED])
        self.assertEqual(parse_inline_diff("a/x -> b/x\n+x")[0].text, "x")

    def test_preamble_notes_read_as_notes(self) -> None:
        payload = "diff --git a/x b/x\nindex 1a2b..3c4d 100644\n--- a/x\n+++ b/x\n@@ -1,1 +1,1 @@\n+x\n\\ No newline at end of file"
        lines = parse_inline_diff(payload)
        self.assertEqual(lines[0].kind, DIFF_NOTE)
        self.assertEqual(lines[1].kind, DIFF_NOTE)
        self.assertEqual(lines[2].kind, DIFF_FILE)
        self.assertEqual(lines[-1].kind, DIFF_NOTE)
        self.assertIn("No newline", lines[-1].text)

    def test_a_three_dash_line_inside_a_hunk_is_a_removed_line(self) -> None:
        self.assertEqual(kinds("@@ -1,1 +1,1 @@\n--- not a header"), [DIFF_HUNK, DIFF_REMOVED])

    def test_a_truncated_preview_reads_as_a_note(self) -> None:
        lines = parse_inline_diff("@@ -1,1 +1,1 @@\n+x\n… (diff truncated)")
        self.assertEqual(lines[-1].kind, DIFF_NOTE)

    def test_a_created_or_deleted_file_keeps_the_side_that_names_it(self) -> None:
        self.assertEqual(parse_inline_diff("--- /dev/null\n+++ b/new.py\n@@ -0,0 +1 @@\n+x")[0].text, "new.py")
        self.assertEqual(parse_inline_diff("--- a/gone.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-x")[0].text, "gone.py")

    def test_empty_and_junk_payloads_parse_to_nothing(self) -> None:
        for payload in ("", "   \n  \n", "\n\n\n", "\x1b[0m\x00"):
            self.assertEqual(parse_inline_diff(payload), [])

    def test_a_huge_payload_is_read_no_further(self) -> None:
        payload = "@@ -1,1 +1,1 @@\n" + "\n".join(f"+line {at}" for at in range(5000))
        self.assertEqual(len(parse_inline_diff(payload)), diffview.MAX_INPUT_LINES)

    def test_crlf_and_lone_returns_are_normalized(self) -> None:
        self.assertEqual(kinds("--- a/x\r\n+++ b/x\r\n@@ -1,1 +1,1 @@\r\n+x\r\n"), [DIFF_FILE, DIFF_HUNK, DIFF_ADDED])


class HasDiffTests(unittest.TestCase):
    def test_a_real_diff_reads_as_one(self) -> None:
        self.assertTrue(has_diff(PAYLOAD))
        self.assertTrue(has_diff("@@ -1,1 +1,1 @@\n+x"))

    def test_prose_and_header_only_payloads_do_not(self) -> None:
        for payload in ("", "just prose", "- a bullet\n- another", "--- a/x\n+++ b/x", "   \n", "+x\n-y"):
            self.assertFalse(has_diff(payload), payload)

    def test_junk_never_raises(self) -> None:
        self.assertFalse(has_diff(None))  # type: ignore[arg-type]
        self.assertFalse(has_diff(12))  # type: ignore[arg-type]


class CompactHeaderTests(unittest.TestCase):
    def test_one_file_edited_reads_once(self) -> None:
        self.assertEqual(compact_file_header("a/ohar/tui/dag.py → b/ohar/tui/dag.py"), "ohar/tui/dag.py")

    def test_ascii_arrow_reads_the_same(self) -> None:
        self.assertEqual(compact_file_header("a/x -> b/x"), "x")

    def test_a_rename_keeps_both_names(self) -> None:
        self.assertEqual(compact_file_header("a/old.py → b/new.py"), "old.py → new.py")

    def test_a_created_or_deleted_file_keeps_one_name(self) -> None:
        self.assertEqual(compact_file_header("/dev/null → b/new.py"), "new.py")
        self.assertEqual(compact_file_header("a/gone.py → /dev/null"), "a/gone.py".replace("a/", ""))

    def test_a_single_side_loses_its_prefix(self) -> None:
        self.assertEqual(compact_file_header("b/only.py"), "only.py")

    def test_debris_is_stripped(self) -> None:
        self.assertEqual(compact_file_header("\x1b[31ma/x\x1b[0m → b/x"), "x")

    def test_nothing_in_nothing_out(self) -> None:
        self.assertEqual(compact_file_header(""), "")
        self.assertEqual(compact_file_header(None), "")  # type: ignore[arg-type]


class RenderTests(unittest.TestCase):
    def test_rows_never_exceed_the_width(self) -> None:
        for width in (6, 10, 20, 40, 80, 120):
            for row in render_diff(PAYLOAD, width, PLAIN):
                self.assertLessEqual(display_width(row), width, f"{width}: {row!r}")

    def test_the_gutter_numbers_lines_from_the_hunk_starts(self) -> None:
        rows = [strip_ansi(row) for row in render_diff(PAYLOAD, 70, PLAIN)]
        self.assertTrue(any(row.startswith("  1 -import os") for row in rows), rows)
        self.assertTrue(any(row.rstrip().startswith("  1 +from") for row in rows), rows)
        self.assertTrue(any(row.startswith("  3  SHUT") for row in rows), rows)
        self.assertTrue(any(row.startswith(" 41 +# note") for row in rows), rows)
        self.assertTrue(any(row.startswith(" 40 -# old") for row in rows), rows)

    def test_a_second_hunk_marks_the_jump(self) -> None:
        rows = [strip_ansi(row) for row in render_diff(PAYLOAD, 70, PLAIN)]
        self.assertTrue(any("⋮" in row for row in rows))
        self.assertFalse(any(row.startswith("@@") for row in rows))
        ascii_rows = [strip_ansi(row) for row in render_diff(PAYLOAD, 70, PLAIN, glyphs="ascii")]
        self.assertTrue(any("!" in row for row in ascii_rows))
        self.assertFalse(any("⋮" in row for row in ascii_rows))

    def test_the_file_header_is_cut_from_the_left(self) -> None:
        payload = f"--- a/deep/nested/path/to/{'dir/' * 20}file.py\n+++ b/deep/nested/path/to/dir/file.py\n@@ -1,1 +1,1 @@\n+x"
        rows = [strip_ansi(row) for row in render_diff(payload, 40, PLAIN)]
        self.assertTrue(rows[0].endswith("file.py"), rows[0])
        self.assertIn("…", rows[0])

    def test_added_and_removed_rows_are_full_width_bands(self) -> None:
        rows = [strip_ansi(row) for row in render_diff(PAYLOAD, 60, PLAIN)]
        banded = [row for row in rows if "+from" in row or "-import" in row]
        self.assertEqual(len(banded), 2)
        for row in banded:
            self.assertEqual(display_width(row), 60)
        context = next(row for row in rows if "SHUT" in row)
        self.assertLess(display_width(context), 60)

    def test_code_lines_hard_wrap_and_repeat_their_marker(self) -> None:
        payload = "@@ -1,1 +1,1 @@\n+" + "x" * 90
        rows = [strip_ansi(row) for row in render_diff(payload, 30, PLAIN)]
        self.assertGreaterEqual(len(rows), 4)
        for row in rows:
            self.assertLessEqual(display_width(row), 30)
            self.assertEqual(row[4:5], "+")  # every row carries its marker past the gutter

    def test_notes_word_wrap(self) -> None:
        payload = "@@ -1,1 +1,2 @@\n+code\n" + "warning: the file is large " * 4
        rows = [strip_ansi(row) for row in render_diff(payload, 40, PLAIN)]
        wrapped = [row for row in rows if "warning" in row or "large" in row]
        self.assertGreaterEqual(len(wrapped), 2)
        for row in wrapped:
            self.assertLessEqual(display_width(row), 40)
            self.assertNotIn("warni\n", row)
            self.assertTrue(row.rstrip().endswith(("warning:", "the", "file", "is", "large")))

    def test_the_preview_is_capped_and_the_rest_is_counted(self) -> None:
        changed = PREVIEW_LINES + 7
        payload = "@@ -1,60 +1,60 @@\n" + "\n".join(f"+line {at}" for at in range(changed))
        rows = [strip_ansi(row) for row in render_diff(payload, 70, PLAIN)]
        shown = [row for row in rows if "line " in row]
        self.assertEqual(len(shown), PREVIEW_LINES - 1)  # the hunk header holds one place in the preview
        self.assertEqual(rows[-1].strip(), f"… omitted {changed - len(shown)} lines")

    def test_one_omitted_line_is_singular(self) -> None:
        payload = "@@ -1,3 +1,3 @@\n+x0\n+x1\n+x2"
        rows = [strip_ansi(row) for row in render_diff(payload, 70, PLAIN, max_lines=3)]
        self.assertIn("omitted 1 line", rows[-1])
        self.assertNotIn("lines", rows[-1])

    def test_ascii_dots_for_an_ascii_terminal(self) -> None:
        payload = "@@ -1,5 +1,5 @@\n" + "\n".join(f"+x{at}" for at in range(5))
        rows = [strip_ansi(row) for row in render_diff(payload, 70, PLAIN, max_lines=2, glyphs="ascii")]
        self.assertIn("... omitted 4 lines", rows[-1])

    def test_zero_max_lines_shows_the_whole_diff(self) -> None:
        payload = "@@ -1,40 +1,40 @@\n" + "\n".join(f"+x{at}" for at in range(40))
        rows = [strip_ansi(row) for row in render_diff(payload, 70, PLAIN, max_lines=0)]
        self.assertEqual(len([row for row in rows if "+x" in row]), 40)
        self.assertFalse(any("omitted" in row for row in rows))

    def test_a_narrow_terminal_gets_no_gutter(self) -> None:
        rows = [strip_ansi(row) for row in render_diff(PAYLOAD, 8, PLAIN)]
        self.assertTrue(rows)
        for row in rows:
            self.assertFalse(row[:4].strip().isdigit(), row)
            self.assertLessEqual(display_width(row), 8)

    def test_no_indent_means_no_gutter(self) -> None:
        rows = [strip_ansi(row) for row in render_diff(PAYLOAD, 60, PLAIN, indent=0)]
        self.assertTrue(rows[0].startswith("ohar/tui/dag.py"), rows[0])
        self.assertTrue(any(row.startswith("-import os") for row in rows))

    def test_color_paints_bands_and_plain_does_not(self) -> None:
        plain_rows = render_diff(PAYLOAD, 60, PLAIN)
        self.assertFalse(any("\x1b" in row for row in plain_rows))
        painted = render_diff(PAYLOAD, 60, COLOR)
        added = next(row for row in painted if "+from" in strip_ansi(row))
        self.assertIn("\x1b[48;2;", added)
        self.assertIn("\x1b[38;2;", added)
        removed = next(row for row in painted if "-import" in strip_ansi(row))
        self.assertIn("\x1b[48;2;", removed)

    def test_a_mono_theme_keeps_the_band_without_a_hue(self) -> None:
        row = next(row for row in render_diff(PAYLOAD, 60, MONO) if "-import" in strip_ansi(row))
        self.assertIn("\x1b[48;2;", row)
        self.assertNotIn("\x1b[38;2;", row)

    def test_no_control_debris_reaches_the_screen(self) -> None:
        rows = render_diff("\x1b[31m+added\x1b[0m\r\n\x00-removed", 60, PLAIN)
        joined = "".join(strip_ansi(row) for row in rows)
        for junk in ("\x00", "\r", "\x1b[31m"):
            self.assertNotIn(junk, joined)

    def test_empty_and_degenerate_arguments(self) -> None:
        self.assertEqual(render_diff("", 60, PLAIN), [])
        self.assertEqual(render_diff(PAYLOAD, 0, PLAIN), [])
        self.assertEqual(render_diff(PAYLOAD, -10, PLAIN), [])
        # prose is not a diff, but it is still printed: as one note row, since `has_diff` is the caller's gate
        self.assertEqual([strip_ansi(row) for row in render_diff("no diff here at all", 60, PLAIN)],
                         ["    no diff here at all"])

    def test_junk_payloads_never_raise(self) -> None:
        for payload in (None, 12, b"bytes", object(), ["list"], "@@@@ \n+++\n---\n@@\n@@ -x,? +y,? @@\n ", "\x00\x01\x02"):
            render_diff(payload, 40, PLAIN)  # type: ignore[arg-type]
            parse_inline_diff(payload)  # type: ignore[arg-type]
            has_diff(payload)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
