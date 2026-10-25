from __future__ import annotations

import unittest
from typing import Callable

from ohar.tui.cells import cell_width
from ohar.tui.pager import Key
from ohar.tui.pickers import (
    MAX_ENTRIES,
    MAX_FILTER_CHARS,
    FilterList,
    PickEntry,
    pick_interactive,
    safe_token,
)
from ohar.terminal import display_width, make_theme, strip_ansi

PLAIN = make_theme("aurora", color=False)
COLOR = make_theme("aurora", color=True, depth="truecolor")


def rows(count: int, current: int = -1) -> list[PickEntry]:
    return [PickEntry(f"model-{i}", f"Model {i}", f"provider {i}", i == current) for i in range(count)]


def type_text(listing: FilterList, text: str) -> None:
    for char in text:
        assert listing.feed_key(Key("char", char)) == "move"


def feed(listing: FilterList, *names: str) -> list[str]:
    return [listing.feed_key(Key(name)) for name in names]


def plain(line: str) -> str:
    return strip_ansi(line)


class SafeTokenTests(unittest.TestCase):
    def test_plain_words_pass(self) -> None:
        for value in ("gpt-4o-mini", "o3", "session-7", "A.b_c", "café", "x" * 300):
            with self.subTest(value=value):
                self.assertTrue(safe_token(value))

    def test_whitespace_and_controls_refused(self) -> None:
        for value in ("", "two words", "a\tb", "a\nb", "\x01abc", "a\x1b[31mb", "trail ", " lead"):
            with self.subTest(value=value):
                self.assertFalse(safe_token(value))

    def test_leading_dashes_refused(self) -> None:
        for dash in ("-", "‐", "‑", "‒", "–", "—", "―", "−"):
            with self.subTest(dash=dash):
                self.assertFalse(safe_token(dash))
                self.assertFalse(safe_token(f"{dash}global"))

    def test_a_dash_inside_the_word_is_no_flag(self) -> None:
        self.assertTrue(safe_token("a–b"))
        self.assertTrue(safe_token("x--y"))


class FilterListStateTests(unittest.TestCase):
    def test_highlight_starts_on_the_current_entry(self) -> None:
        listing = FilterList(rows(5, current=3))
        self.assertEqual(len(listing), 5)
        assert listing.selected is not None
        self.assertEqual(listing.selected.value, "model-3")

    def test_highlight_starts_at_the_first_row_without_a_current(self) -> None:
        assert FilterList(rows(5)).selected is not None
        self.assertEqual(FilterList(rows(5)).selected.value, "model-0")

    def test_unsafe_values_are_refused_and_the_rest_is_cleaned(self) -> None:
        listing = FilterList(
            [
                PickEntry("--global", "a flag"),
                PickEntry("two words", "spaced"),
                PickEntry("bad\x1b[31m", "escaped"),
                PickEntry("", "blank"),
                PickEntry("ok", "kept"),
            ]
        )
        self.assertEqual([entry.value for entry in listing.matches], ["bad", "ok"])
        assert listing.selected is not None
        self.assertEqual(listing.selected.label, "escaped")

    def test_labels_fall_back_to_the_value(self) -> None:
        listing = FilterList([PickEntry("solo", "  ")])
        assert listing.selected is not None
        self.assertEqual(listing.selected.label, "solo")

    def test_filter_is_case_insensitive_over_label_and_value(self) -> None:
        listing = FilterList(rows(30))
        type_text(listing, "MODEL 12")
        self.assertEqual([entry.value for entry in listing.matches], ["model-12"])
        listing.filter = ""
        type_text(listing, "-1")
        self.assertEqual(
            [entry.value for entry in listing.matches],
            ["model-1", "model-10", "model-11", "model-12", "model-13", "model-14",
             "model-15", "model-16", "model-17", "model-18", "model-19"],
        )

    def test_detail_is_not_searched(self) -> None:
        listing = FilterList(rows(4))
        type_text(listing, "provider")
        self.assertEqual(listing.matches, [])
        self.assertIsNone(listing.selected)

    def test_typing_moves_the_highlight_back_to_the_first_match(self) -> None:
        listing = FilterList(rows(20, current=9))
        feed(listing, "end")
        assert listing.selected is not None
        self.assertEqual(listing.selected.value, "model-19")
        type_text(listing, "-1")
        assert listing.selected is not None
        self.assertEqual(listing.selected.value, "model-1")

    def test_backspace_shrinks_the_filter(self) -> None:
        listing = FilterList(rows(5))
        type_text(listing, "-12")
        self.assertEqual(listing.filter, "-12")
        self.assertEqual(listing.feed_key(Key("backspace")), "move")
        self.assertEqual(listing.filter, "-1")

    def test_filter_growth_is_bounded(self) -> None:
        listing = FilterList(rows(3))
        type_text(listing, "x" * (MAX_FILTER_CHARS + 40))
        self.assertEqual(len(listing.filter), MAX_FILTER_CHARS)

    def test_moves_clamp_and_page_steps(self) -> None:
        listing = FilterList(rows(20), page_size=5)
        self.assertEqual(feed(listing, "down", "down"), ["move", "move"])
        assert listing.selected is not None
        self.assertEqual(listing.selected.value, "model-2")
        self.assertEqual(listing.feed_key(Key("pagedown")), "move")
        self.assertEqual(listing.selected.value, "model-7")
        self.assertEqual(listing.feed_key(Key("pageup")), "move")
        self.assertEqual(listing.selected.value, "model-2")
        feed(listing, "end", "down", "down")
        self.assertEqual(listing.selected.value, "model-19")
        feed(listing, "up")
        self.assertEqual(listing.selected.value, "model-18")
        self.assertEqual(listing.feed_key(Key("home")), "move")
        self.assertEqual(listing.selected.value, "model-0")

    def test_page_step_follows_the_last_render(self) -> None:
        listing = FilterList(rows(30))
        listing.render(60, PLAIN, title="Model", max_rows=7)  # five list rows
        self.assertEqual(listing.feed_key(Key("page-down")), "move")
        assert listing.selected is not None
        self.assertEqual(listing.selected.value, "model-5")

    def test_a_silly_page_size_still_moves(self) -> None:
        for page_size, picked in ((0, "model-1"), (-8, "model-1"), (10 ** 6, "model-39"), (7, "model-7")):
            with self.subTest(page_size=page_size):
                listing = FilterList(rows(40), page_size=page_size)
                listing.feed_key(Key("pagedown"))
                assert listing.selected is not None
                self.assertEqual(listing.selected.value, picked)

    def test_enter_picks_and_escape_or_ctrl_c_closes(self) -> None:
        listing = FilterList(rows(3))
        self.assertEqual(listing.feed_key(Key("enter")), "pick")
        assert listing.selected is not None
        self.assertEqual(listing.selected.value, "model-0")
        self.assertEqual(listing.feed_key(Key("escape")), "close")
        self.assertEqual(listing.feed_key(Key("char", "\x03")), "close")

    def test_enter_on_an_empty_match_list_picks_nothing(self) -> None:
        listing = FilterList(rows(3))
        type_text(listing, "zzz")
        self.assertEqual(listing.feed_key(Key("enter")), "move")

    def test_unknown_and_ignored_keys_do_nothing(self) -> None:
        listing = FilterList(rows(3))
        for key in (Key("left"), Key("tab"), Key("f1"), Key("char", "\x00"), Key("char", "")):
            with self.subTest(key=key):
                self.assertEqual(listing.feed_key(key), "move")
        self.assertEqual(listing.filter, "")
        assert listing.selected is not None
        self.assertEqual(listing.selected.value, "model-0")

    def test_entries_are_capped(self) -> None:
        self.assertEqual(len(FilterList(rows(MAX_ENTRIES + 50))), MAX_ENTRIES)


class RenderTests(unittest.TestCase):
    def test_frame_shape(self) -> None:
        listing = FilterList(rows(40))
        lines = listing.render(60, PLAIN, title="Model", max_rows=12)
        self.assertEqual(len(lines), 12)
        self.assertIn("Model (1/40)", plain(lines[0]))
        self.assertIn("Model 0", plain(lines[1]))
        self.assertIn("provider 1", plain(lines[2]))
        self.assertTrue(plain(lines[-1]).endswith("type to filter"))

    def test_title_carries_the_position_once_it_scrolls(self) -> None:
        listing = FilterList(rows(162))
        for _ in range(140):
            listing.feed_key(Key("down"))
        lines = listing.render(60, PLAIN, title="Model", max_rows=10)
        self.assertIn("Model (141/162)", plain(lines[0]))

    def test_title_counts_matches_over_total_while_filtering(self) -> None:
        listing = FilterList(rows(30))
        type_text(listing, "-1")
        lines = listing.render(60, PLAIN, title="Model", max_rows=20)
        self.assertIn("Model (11/30)", plain(lines[0]))

    def test_rows_never_pass_the_width_in_any_theme(self) -> None:
        for theme in (PLAIN, COLOR):
            for glyphs in ("unicode", "ascii"):
                for width in (4, 9, 17, 40, 78):
                    listing = FilterList(rows(12, current=3))
                    type_text(listing, "-")
                    for max_rows in (1, 2, 3, 7, 12):
                        lines = listing.render(width, theme, title="Model", max_rows=max_rows, glyphs=glyphs)
                        self.assertLessEqual(len(lines), max_rows)
                        for line in lines:
                            with self.subTest(width=width, max_rows=max_rows, glyphs=glyphs, line=line):
                                self.assertLessEqual(cell_width(plain(line)), width)
                                self.assertLessEqual(display_width(plain(line)), width)

    def test_the_highlighted_row_fills_the_frame(self) -> None:
        listing = FilterList(rows(3))
        lines = listing.render(30, PLAIN, title="Model", max_rows=5)
        self.assertEqual(cell_width(plain(lines[1])), 30)

    def test_the_current_entry_is_marked_and_the_pointer_follows_the_highlight(self) -> None:
        listing = FilterList(rows(6, current=4))
        body = [plain(line) for line in listing.render(40, PLAIN, title="Model", max_rows=8)[1:-1]]
        self.assertEqual(sum("●" in line for line in body), 1)
        marked = next(line for line in body if "●" in line)
        self.assertTrue(marked.startswith("›"))
        listing.feed_key(Key("down"))
        body = [plain(line) for line in listing.render(40, PLAIN, title="Model", max_rows=8)[1:-1]]
        pointed = next(line for line in body if line.startswith("›"))
        self.assertIn("Model 5", pointed)
        self.assertNotIn("●", pointed)

    def test_ascii_glyphs_use_no_box_drawing(self) -> None:
        listing = FilterList(rows(6, current=2))
        joined = "".join(plain(line) for line in listing.render(40, PLAIN, title="Model", glyphs="ascii"))
        for mark in ("─", "›", "●", "❯"):
            self.assertNotIn(mark, joined)
        self.assertIn("*", joined)
        self.assertIn(">", joined)

    def test_a_filter_that_matches_nothing_says_so(self) -> None:
        listing = FilterList(rows(4))
        type_text(listing, "zzz")
        lines = listing.render(40, PLAIN, title="Model")
        self.assertEqual(len(lines), 3)
        self.assertIsNone(listing.selected)
        self.assertIn('no entry matches "zzz"', plain(lines[1]))
        self.assertIn("zzz", plain(lines[2]))

    def test_no_room_for_a_frame_keeps_the_row_and_the_filter(self) -> None:
        listing = FilterList(rows(4))
        self.assertEqual(listing.render(40, PLAIN, title="Model", max_rows=0), [])
        self.assertEqual(listing.render(0, PLAIN, title="Model"), [])
        one = listing.render(40, PLAIN, title="Model", max_rows=1)
        self.assertEqual(len(one), 1)
        self.assertTrue(plain(one[0]).startswith("›"))
        self.assertIn("Model 0", plain(one[0]))
        self.assertEqual(cell_width(plain(one[0])), 40)
        two = listing.render(40, PLAIN, title="Model", max_rows=2)
        self.assertEqual(len(two), 2)
        self.assertIn("Model 0", plain(two[0]))
        self.assertIn("type to filter", plain(two[1]))

    def test_hostile_entry_text_is_cleaned_before_it_shows(self) -> None:
        listing = FilterList([PickEntry("safe", "\x1b[31mEvil\r\n\u202e title", "de\x00tail")])
        joined = "".join(plain(line) for line in listing.render(40, PLAIN, title="Model"))
        for char in ("\x1b", "\r", "\x00", "‮"):
            self.assertNotIn(char, joined)
        self.assertIn("Evil title", joined)

    def test_a_very_long_label_is_truncated(self) -> None:
        listing = FilterList([PickEntry("x", "L" * 4000, "D" * 4000)])
        for line in listing.render(50, COLOR, title="Model", max_rows=6):
            self.assertLessEqual(cell_width(plain(line)), 50)

    def test_a_hostile_title_is_cleaned(self) -> None:
        listing = FilterList(rows(2))
        joined = "".join(plain(line) for line in listing.render(40, PLAIN, title="\x1b]0;evil\x07 Sessions\r\n"))
        self.assertNotIn("\x1b", joined)
        self.assertIn("Sessions", joined)


class PickInteractiveTests(unittest.TestCase):
    def reader(self, *lines: str) -> Callable[[str], str]:
        seen: list[str] = []

        def read(prompt: str) -> str:
            seen.append(prompt)
            return lines[len(seen) - 1] if len(seen) <= len(lines) else "\x1b"

        read.seen = seen  # type: ignore[attr-defined]
        return read

    def test_an_empty_line_picks_the_highlighted_row(self) -> None:
        read = self.reader("")
        picked = pick_interactive(rows(5, current=2), title="Model", theme=PLAIN, width=40, read_line=read)
        self.assertIsNotNone(picked)
        assert picked is not None
        self.assertEqual(picked.value, "model-2")
        self.assertIn("Model (5)", plain(read.seen[0]))  # type: ignore[attr-defined]

    def test_typing_filters_and_the_end_of_the_line_picks(self) -> None:
        read = self.reader("model-7")
        picked = pick_interactive(rows(30), title="Model", theme=PLAIN, width=40, read_line=read)
        self.assertIsNotNone(picked)
        assert picked is not None
        self.assertEqual(picked.value, "model-7")

    def test_the_frame_shows_in_the_prompt(self) -> None:
        read = self.reader("\x1b")
        self.assertIsNone(pick_interactive(rows(3), title="Theme", theme=PLAIN, width=30, read_line=read))
        prompt = read.seen[0]  # type: ignore[attr-defined]
        self.assertIn("Theme (3)", plain(prompt))
        self.assertIn("type to filter", plain(prompt))
        self.assertEqual(prompt.count("\n"), 5)  # a three-row frame: title, the rows, the filter line

    def test_a_line_that_matches_nothing_leaves_empty_handed(self) -> None:
        read = self.reader("no such thing")
        self.assertIsNone(pick_interactive(rows(3), title="Model", theme=PLAIN, width=40, read_line=read))

    def test_escape_closes_without_picking(self) -> None:
        read = self.reader("\x1b")
        self.assertIsNone(pick_interactive(rows(3), title="Model", theme=PLAIN, width=40, read_line=read))

    def test_a_line_that_changes_nothing_closes_the_picker(self) -> None:
        # A reader that hands back a whole line with its newline, and nothing in it the
        # picker can act on: asking again would ask forever, so it stops.
        read = self.reader("\x1b\n")
        self.assertIsNone(pick_interactive(rows(3), title="Model", theme=PLAIN, width=40, read_line=read))
        self.assertEqual(len(read.seen), 1)  # type: ignore[attr-defined]

    def test_eof_and_ctrl_c_from_the_reader_close_without_picking(self) -> None:
        def dying(prompt: str) -> str:
            raise EOFError

        self.assertIsNone(pick_interactive(rows(3), title="Model", theme=PLAIN, width=40, read_line=dying))

        def interrupting(prompt: str) -> str:
            raise KeyboardInterrupt

        self.assertIsNone(
            pick_interactive(rows(3), title="Model", theme=PLAIN, width=40, read_line=interrupting))

    def test_nothing_to_pick_never_asks(self) -> None:
        def boom(prompt: str) -> str:
            raise AssertionError("read_line called with nothing to pick")

        self.assertIsNone(pick_interactive([], title="Model", theme=PLAIN, width=40, read_line=boom))
        self.assertIsNone(
            pick_interactive([PickEntry("-x", "flag")], title="Model", theme=PLAIN, width=40, read_line=boom))


if __name__ == "__main__":
    unittest.main()
