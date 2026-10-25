from __future__ import annotations

import unittest
import unicodedata

from ohar.terminal import strip_ansi

from ohar.tui.cells import (
    cell_width,
    clip_line,
    clusters,
    strip_control,
    truncate,
    truncate_left,
    wrap,
)

FAMILY = "\U0001F468‍\U0001F469‍\U0001F467‍\U0001F466"  # one 2-cell glyph
SKIN = "\U0001F466\U0001F3FB"  # one 2-cell glyph with a skin tone
HEART = "❤️"  # VS16 makes a 1-cell symbol a 2-cell emoji
FLAG = "\U0001F1EF\U0001F1F5"  # a regional-indicator pair paints one flag
COMBINING = "e\u0301"  # decomposed é
HANGUL_NFD = unicodedata.normalize("NFD", "한")  # macOS pastes jamo like this
CJK = "中"
CJK2 = "中文"


class ClusterTests(unittest.TestCase):
    def test_zwj_family_is_one_cluster(self) -> None:
        self.assertEqual(clusters(FAMILY), [FAMILY])
        self.assertEqual(cell_width(FAMILY), 2)

    def test_skin_tone_vs16_and_flags_join(self) -> None:
        self.assertEqual(clusters(SKIN), [SKIN])
        self.assertEqual(cell_width(SKIN), 2)
        self.assertEqual(clusters(HEART), [HEART])
        self.assertEqual(cell_width(HEART), 2)
        self.assertEqual(clusters(FLAG), [FLAG])
        self.assertEqual(cell_width(FLAG), 2)

    def test_combining_marks_and_decomposed_hangul(self) -> None:
        self.assertEqual(clusters(COMBINING), [unicodedata.normalize("NFC", COMBINING)])
        self.assertEqual(cell_width(COMBINING), 1)
        self.assertGreater(len(HANGUL_NFD), 1)
        self.assertEqual(clusters(HANGUL_NFD), [unicodedata.normalize("NFC", HANGUL_NFD)])
        self.assertEqual(cell_width(HANGUL_NFD), cell_width("한"))
        self.assertEqual(cell_width("한"), 2)
        self.assertEqual(cell_width(COMBINING + "x"), 2)

    def test_ascii_controls_and_wide_text(self) -> None:
        self.assertEqual(clusters("abc"), ["a", "b", "c"])
        self.assertEqual(clusters(""), [])
        self.assertEqual(cell_width(""), 0)
        self.assertEqual(cell_width("plain text"), 10)
        self.assertEqual(cell_width(CJK2), 4)
        self.assertEqual(cell_width("a\tb"), 2)  # a control paints nothing
        self.assertEqual(cell_width("é"), 1)  # East Asian Ambiguous: one cell

    def test_long_text_measured_like_short_text(self) -> None:
        long = (FAMILY + CJK) * 100  # past the window the shared caches keep
        self.assertEqual(len(clusters(long)), 200)
        self.assertEqual(cell_width(long), 400)
        cut = truncate(long, 20)
        self.assertLessEqual(cell_width(cut), 20)
        self.assertTrue(cut.endswith("…"))


class StripControlTests(unittest.TestCase):
    def test_complete_sequences(self) -> None:
        self.assertEqual(strip_control("\x1b[31mred\x1b[0m"), "red")
        self.assertEqual(strip_control("\x1b[38;2;1;2;3mx\x1b[m"), "x")
        self.assertEqual(strip_control("\x1b]8;;https://example.com\x07link\x1b]8;;\x07"), "link")
        self.assertEqual(strip_control("\x1b[2Jclear"), "clear")
        self.assertEqual(strip_control("plain"), "plain")
        self.assertEqual(strip_control(""), "")

    def test_truncated_sequences_leave_no_parameter_debris(self) -> None:
        self.assertEqual(strip_control("\x1b[38;2;1;2"), "")
        self.assertEqual(strip_control("a\x1b[38;2;1;2"), "a")
        self.assertEqual(strip_control("a\x1b[38;2;1;2more"), "aore")  # 'm' is the final byte
        self.assertEqual(strip_control("\x1b[5"), "")
        self.assertEqual(strip_control("a\x1b"), "a")
        self.assertEqual(strip_control("\x1b]0;title"), "")  # an unterminated OSC swallows the rest
        self.assertEqual(strip_control("\x1bP#qdata"), "")  # DCS
        self.assertEqual(strip_control("\x1bSa"), "a")
        self.assertIn("[38", strip_ansi("\x1b[38;2;1;2"))  # the weaker filter keeps the debris
        self.assertNotIn("[38", strip_control("\x1b[38;2;1;2"))

    def test_string_terminators_and_8bit_introducers(self) -> None:
        self.assertEqual(strip_control("\x1b]0;t\x1b\\rest"), "rest")
        self.assertEqual(strip_control("\x9b31mx"), "x")
        self.assertEqual(strip_control("\x9d8;;url\x9cx"), "x")
        self.assertEqual(strip_control("\x1b(Bplain"), "plain")
        self.assertEqual(strip_control("a\x9bHb"), "ab")

    def test_controls_newlines_tabs_and_bidi(self) -> None:
        self.assertEqual(strip_control("a\x00b\x07c\x08d\x0be\x1ff"), "abcdef")
        self.assertEqual(strip_control("line\r\nnext\rlast"), "line\nnext\nlast")
        self.assertEqual(strip_control("a\tb"), "a    b")
        self.assertEqual(strip_control("a\u202eb\u2066c"), "abc")
        self.assertEqual(strip_control("keep‍join and ❤️"), "keep‍join and ❤️")  # emoji parts survive
        self.assertEqual(strip_control("\x1b]0;a\x07b\rc"), "b\nc")
        cleaned = strip_control("\x1b[31m\ta\r\n")
        self.assertEqual(strip_control(cleaned), cleaned)  # idempotent
        self.assertEqual(cell_width(strip_control("\x1b[31mab")), 2)


class TruncateTests(unittest.TestCase):
    def test_right_truncate(self) -> None:
        self.assertEqual(truncate("abcdef", 10), "abcdef")
        self.assertEqual(truncate("abcdef", 6), "abcdef")
        self.assertEqual(truncate("abcdef", 4), "abc…")
        self.assertEqual(truncate("abcdef", 1), "…")
        self.assertEqual(truncate("abcdef", 0), "")
        self.assertEqual(truncate("abcdef", -3), "")
        self.assertEqual(truncate("abcdef", 4, ellipsis=""), "abcd")
        self.assertEqual(truncate("", 4), "")

    def test_never_splits_a_cluster(self) -> None:
        self.assertEqual(clusters(truncate(FAMILY + "abc", 3)), [FAMILY, "…"])
        self.assertEqual(cell_width(truncate(FAMILY + "abc", 3)), 3)
        self.assertEqual(clusters(truncate(CJK2 + "x", 5)), ["中", "文", "x"])  # fits: no cut
        self.assertEqual(clusters(truncate(CJK2 + CJK2 + "x", 5)), [CJK, "文", "…"])
        self.assertEqual(truncate(CJK2 + CJK2, 4), "中…")  # half a 2-cell glyph is not a unit
        self.assertEqual(truncate(CJK2 + CJK2, 2), "…")  # no cluster fits the room the mark leaves
        self.assertEqual(truncate(CJK2 + CJK2, 1), "…")

    def test_ellipsis_wider_than_the_budget(self) -> None:
        self.assertEqual(truncate("中中中", 2, ellipsis="……"), "……")
        self.assertEqual(truncate("中中中", 1, ellipsis="……"), "")
        self.assertEqual(truncate("abcdef", 3, ellipsis="~~~"), "~~~")
        self.assertEqual(cell_width(truncate("abcdef", 5, ellipsis="~~~")), 5)


class TruncateLeftTests(unittest.TestCase):
    def test_keeps_the_end(self) -> None:
        cut = truncate_left("/tmp/ohar/work/report.md", 12)
        self.assertEqual(cut, "…k/report.md")
        self.assertEqual(cell_width(cut), 12)
        self.assertTrue(cut.endswith("report.md"))
        self.assertEqual(truncate_left("abcdef", 6), "abcdef")
        self.assertEqual(truncate_left("abcdef", 0), "")
        self.assertEqual(truncate_left("abcdef", 4, ellipsis=""), "cdef")

    def test_cluster_safe(self) -> None:
        self.assertEqual(clusters(truncate_left(FAMILY + "abc", 3)), ["…", "b", "c"])
        self.assertEqual(cell_width(truncate_left(CJK2 * 3, 5)), 5)
        self.assertEqual(truncate_left(CJK2 * 3, 5), "…中文")
        self.assertEqual(clusters(truncate_left(CJK2, 1)), ["…"])


class ClipLineTests(unittest.TestCase):
    def test_pad_or_cut(self) -> None:
        self.assertEqual(clip_line("ab", 5), "ab   ")
        self.assertEqual(clip_line("abcde", 5), "abcde")
        self.assertEqual(clip_line("abcdef", 4), "abcd")
        self.assertEqual(clip_line("ab", 0), "")
        self.assertEqual(clip_line("", 3), "   ")
        self.assertEqual(clip_line("ab", 5, fill="."), "ab...")
        self.assertEqual(clip_line("abc", 8, fill=""), "abc")

    def test_a_wide_fill_costs_its_own_cells(self) -> None:
        self.assertEqual(clip_line("ab", 6, fill=CJK), "ab" + CJK + CJK)
        self.assertEqual(clip_line("", 5, fill=CJK), "中中")  # the odd cell is not split
        self.assertEqual(cell_width(clip_line("", 5, fill=CJK)), 4)

    def test_fills_the_row_without_splitting_a_cluster(self) -> None:
        self.assertEqual(cell_width(clip_line(CJK2 + CJK2, 5)), 5)
        self.assertEqual(clip_line(CJK2 + CJK2, 3), "中 ")
        self.assertEqual(clip_line(CJK2 * 3, 5), "中文 ")
        self.assertEqual(clusters(clip_line(FAMILY, 4)), [FAMILY, " ", " "])
        for cells in range(1, 12):
            self.assertEqual(cell_width(clip_line(f"{CJK2}{FAMILY}x", cells)), cells)


class WrapTests(unittest.TestCase):
    def test_word_wrap(self) -> None:
        self.assertEqual(wrap("the quick brown fox", 10), ["the quick", "brown fox"])
        self.assertEqual(wrap("", 10), [""])
        self.assertEqual(wrap("a\n\nb", 5), ["a", "", "b"])
        self.assertEqual(wrap("lead   follow", 20), ["lead   follow"])
        self.assertEqual(wrap("x y", 1), ["x", "y"])
        self.assertEqual(wrap("a\r\nb", 10), ["a", "b"])
        self.assertEqual(wrap("word", 0), [""])

    def test_lines_never_exceed_the_width(self) -> None:
        samples = (
            "one two three four five six seven",
            f"emoji {FAMILY} and {CJK2} text here",
            "https://example.com/a/very/long/path/to/file.py",
            "a" * 30,
            f"path {CJK2} {FAMILY} tail",
        )
        for width in (1, 2, 3, 5, 8, 13, 40):
            for text in samples:
                for line in wrap(text, width):
                    # only a cluster wider than the row itself (a 2-cell glyph in 1 cell) goes over
                    self.assertLessEqual(cell_width(line), max(width, 2), (width, line))

    def test_paths_and_urls_break_between_their_parts(self) -> None:
        self.assertEqual(
            wrap("see https://example.com/some/very/long/path/file.py now", 20),
            ["see", "https://example.com/", "some/very/long/path/", "file.py now"],
        )
        self.assertEqual(wrap("aaaa/bbbb/cccc", 5), ["aaaa/", "bbbb/", "cccc"])
        for width in (3, 5, 8, 9, 12):
            for line in wrap("https://example.com/x", width):
                self.assertTrue(line.strip("/"), (width, line))  # no row is made of slashes alone

    def test_hard_wrap_breaks_anywhere(self) -> None:
        self.assertEqual(wrap("code line    with spaces", 8, hard=True), ["code lin", "e    wit", "h spaces"])
        self.assertEqual(wrap("abcdefg", 3, hard=True), ["abc", "def", "g"])
        self.assertEqual(wrap("diff text", 4, hard=True), ["diff", " tex", "t"])
        self.assertEqual(wrap(strip_control("diff\ttext"), 4, hard=True), ["diff", "    ", "text"])
        self.assertEqual([cell_width(line) for line in wrap("x" * 25, 4, hard=True)], [4] * 6 + [1])

    def test_indent_fronts_every_text_line(self) -> None:
        self.assertEqual(wrap("hello world", 12, indent="  "), ["  hello", "  world"])
        self.assertEqual(wrap("a\nb", 12, indent="> "), ["> a", "> b"])
        self.assertEqual(wrap("", 12, indent="  "), [""])
        self.assertEqual(wrap("a\n\nb", 8, indent="  "), ["  a", "", "  b"])
        self.assertEqual(wrap("text", 4, indent="      "), ["text"])  # no room left: the indent goes
        self.assertTrue(all(cell_width(line) <= 9 for line in wrap("one two three", 9, indent="  ")))

    def test_a_cluster_wider_than_the_line_gets_its_own_row(self) -> None:
        self.assertEqual(wrap(f"{CJK2}x", 1, hard=True), ["中", "文", "x"])
        self.assertEqual(wrap(FAMILY, 1), [FAMILY])

    def test_breaks_lose_no_words(self) -> None:
        source = "alpha beta gamma  delta\nepsilon zeta"
        words = " ".join(" ".join(line.split()) for line in wrap(source, 9)).split()
        self.assertEqual(words, source.split())
        self.assertEqual(" ".join(wrap("one two three", 5)).split(), ["one", "two", "three"])

    def test_hard_wrap_keeps_every_character(self) -> None:
        for source in (f"a/*b  c{CJK}{'x' * 20} d", "line one\nline two", FAMILY + " tail"):
            for width in (1, 3, 6, 11):
                self.assertEqual("".join(wrap(source, width, hard=True)), source.replace("\n", ""))


if __name__ == "__main__":
    unittest.main()
