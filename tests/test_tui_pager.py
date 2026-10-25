"""Tests for the HUD's pager: key decoding, the page math, and the alternate-screen run."""

from __future__ import annotations

import io
import os
import tempfile
import threading
import time
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from ohar.terminal import display_width, make_theme, strip_ansi
from ohar.tui.pager import (
    ALT_SCREEN_OFF,
    ALT_SCREEN_ON,
    CHAR,
    CTRL_C,
    DOWN,
    END,
    ESCAPE,
    HOME,
    Key,
    LEFT,
    RIGHT,
    PAGE_DOWN,
    PAGE_UP,
    Pager,
    SHOW_CURSOR,
    UP,
    decode_keys,
    show_lines,
)

try:
    import termios as _termios
except ImportError:  # a platform with no termios cannot page on a pty either
    _termios = None

PLAIN = make_theme("mono", color=False)
COLOR = make_theme("aurora")


def names(data: bytes) -> list[str]:
    return [key.name for key in decode_keys(data)]


def pairs(data: bytes) -> list[tuple[str, str]]:
    return [(key.name, key.char) for key in decode_keys(data)]


class DecodeKeysTests(unittest.TestCase):
    def test_typed_keys_are_their_own_characters(self) -> None:
        self.assertEqual(pairs(b"qjG"), [(CHAR, "q"), (CHAR, "j"), (CHAR, "G")])
        self.assertEqual(pairs(b" "), [(CHAR, " ")])

    def test_arrow_and_page_sequences_decode(self) -> None:
        self.assertEqual(
            names(b"\x1b[A\x1b[B\x1b[C\x1b[D"),
            [UP, DOWN, RIGHT, LEFT],
        )
        self.assertEqual(names(b"\x1b[5~\x1b[6~\x1b[H\x1b[F"), [PAGE_UP, PAGE_DOWN, HOME, END])
        self.assertEqual(names(b"\x1b[1~\x1b[4~\x1b[7~\x1b[8~"), [HOME, END, HOME, END])

    def test_application_mode_arrows_decode(self) -> None:
        self.assertEqual(names(b"\x1bOA\x1bOB\x1bOH"), [UP, DOWN, HOME])
        self.assertEqual(names(b"\x1bOP"), ["f1"])

    def test_control_bytes_become_named_keys(self) -> None:
        self.assertEqual(names(b"\r\n\t\x7f\x03\x04"), ["enter", "enter", "tab", "backspace", CTRL_C, "ctrl+d"])

    def test_lone_escape_is_escape(self) -> None:
        self.assertEqual(names(b"\x1b"), [ESCAPE])
        self.assertEqual(names(b"\x1b\x1b"), [ESCAPE, ESCAPE])

    def test_alt_chord_carries_its_character(self) -> None:
        self.assertEqual(pairs(b"\x1bb"), [("alt", "b")])

    def test_multibyte_characters_stay_one_key(self) -> None:
        self.assertEqual(pairs("漢🙂".encode("utf-8")), [(CHAR, "漢"), (CHAR, "🙂")])

    def test_osc_strings_type_nothing(self) -> None:
        self.assertEqual(pairs(b"\x1b]0;title\x07x"), [(CHAR, "x")])
        self.assertEqual(pairs(b"\x1b]12;rgb:0000/0000/0000\x1b\\y"), [(CHAR, "y")])

    def test_unknown_and_overlong_sequences_are_one_key_each(self) -> None:
        self.assertEqual(names(b"\x1b[<0;12;34M"), ["unknown"])
        self.assertEqual(names(b"\x1b[" + b"9" * 40 + b"~j"), ["unknown", CHAR])

    def test_a_sequence_whose_rest_never_arrives_is_dropped(self) -> None:
        self.assertEqual(decode_keys(b"\x1b["), [])
        self.assertEqual(decode_keys(b"\x1b[5"), [])
        self.assertEqual(decode_keys(b"j\x1bO"), [(Key(CHAR, "j"))])

    def test_empty_and_junk_input_never_raises(self) -> None:
        self.assertEqual(decode_keys(b""), [])
        self.assertIsInstance(decode_keys(bytes(range(256))), list)

    def test_keys_are_frozen_values(self) -> None:
        self.assertEqual(Key(CHAR, "j"), Key(CHAR, "j"))
        self.assertEqual(Key("up").char, "")
        with self.assertRaises(FrozenInstanceError):
            Key(CHAR, "j").name = "k"  # type: ignore[misc]


class PageTests(unittest.TestCase):
    def test_a_page_never_holds_more_rows_than_the_screen(self) -> None:
        pager = Pager([f"row {index}" for index in range(50)])
        for rows in range(1, 12):
            with self.subTest(rows=rows):
                self.assertEqual(len(pager.page(40, rows)), rows)

    def test_the_page_shows_the_requested_window_and_a_footer(self) -> None:
        pager = Pager([f"row {index}" for index in range(4)], title="logs")
        page = pager.page(60, 3)
        self.assertEqual(page[:2], ["row 0", "row 1"])
        self.assertIn("logs", page[2])
        self.assertIn("1-2/4", page[2])

    def test_rows_are_cut_to_the_width_and_the_footer_pads_to_it(self) -> None:
        page = Pager(["x" * 200, "short"]).page(20, 3)
        self.assertEqual([display_width(row) for row in page], [20, 5, 20])

    def test_control_characters_cannot_move_the_cursor(self) -> None:
        pager = Pager(["red\x1b[31mx\x07\nnext", "\x1b[?1049hstage"])
        page = pager.page(40, 3)
        self.assertEqual(page[0], "redx next")
        self.assertEqual(page[1], "stage")
        self.assertNotIn("\x1b", "".join(page))

    def test_a_lazy_source_is_read_once_and_only_when_paged(self) -> None:
        calls: list[int] = []

        def produce() -> list[str]:
            calls.append(1)
            return [f"late {index}" for index in range(10)]

        pager = Pager(produce)
        self.assertEqual(calls, [])
        self.assertEqual(pager.page(30, 2)[0], "late 0")
        pager.handle_key(Key(CHAR, "j"))
        self.assertEqual(pager.page(30, 2)[0], "late 1")
        self.assertEqual(calls, [1])

    def test_a_single_string_becomes_its_own_rows(self) -> None:
        self.assertEqual(Pager("one\ntwo\n").lines, ["one", "two"])

    def test_an_empty_pager_still_says_so(self) -> None:
        page = Pager([], title="history").page(30, 4)
        self.assertEqual(len(page), 1)
        self.assertIn("0-0/0", page[0])

    def test_the_footer_is_painted_only_when_a_theme_is_given(self) -> None:
        plain = Pager(["a"], theme=PLAIN).page(40, 2)
        colored = Pager(["a"], theme=COLOR).page(40, 2)
        self.assertNotIn("\x1b", plain[1])
        self.assertIn("\x1b[", colored[1])
        self.assertEqual(strip_ansi(colored[1]), plain[1])

    def test_a_footer_is_drawn_when_there_is_anything_to_say(self) -> None:
        self.assertEqual(Pager([], hint="", title="").page(20, 5), [])
        page = Pager(["a", "b"], hint="").page(20, 5)
        self.assertEqual(page[:2], ["a", "b"])
        self.assertIn("1-2/2", page[2])


class HandleKeyTests(unittest.TestCase):
    def pager(self, count: int = 40) -> Pager:
        pager = Pager([f"row {index}" for index in range(count)])
        pager.page(60, 11)
        return pager

    def test_line_keys_move_one_row_and_stop_at_both_ends(self) -> None:
        pager = self.pager()
        self.assertTrue(pager.handle_key(Key(CHAR, "j")))
        self.assertEqual(pager.top, 1)
        self.assertTrue(pager.handle_key(Key(UP)))
        self.assertEqual(pager.top, 0)
        for _ in range(3):
            pager.handle_key(Key(UP))
        self.assertEqual(pager.top, 0)

    def test_page_keys_move_by_the_body_and_clamp_at_the_end(self) -> None:
        pager = self.pager()
        pager.handle_key(Key(PAGE_DOWN))
        self.assertEqual(pager.top, 10)
        for _ in range(10):
            pager.handle_key(Key(CHAR, " "))
        self.assertEqual(pager.top, 30)

    def test_space_and_b_page_like_a_terminal(self) -> None:
        pager = self.pager()
        pager.handle_key(Key(CHAR, " "))
        self.assertEqual(pager.top, 10)
        pager.handle_key(Key(CHAR, "b"))
        self.assertEqual(pager.top, 0)

    def test_ends_jump_to_the_first_and_last_page(self) -> None:
        pager = self.pager()
        pager.handle_key(Key(END))
        self.assertEqual(pager.top, 30)
        self.assertEqual(pager.page(60, 11)[-2], "row 39")
        pager.handle_key(Key(HOME))
        self.assertEqual(pager.top, 0)
        pager.handle_key(Key(CHAR, "G"))
        self.assertEqual(pager.top, 30)

    def test_keys_that_are_not_binding_do_not_move_anything(self) -> None:
        pager = self.pager()
        self.assertTrue(pager.handle_key(Key("f1")))
        self.assertTrue(pager.handle_key(Key(CHAR, "x")))
        self.assertEqual(pager.top, 0)

    def test_quit_keys_end_the_pager(self) -> None:
        for key in (Key(ESCAPE), Key(CHAR, "q"), Key(CTRL_C), Key("ctrl+d")):
            with self.subTest(key=key.name or key.char):
                self.assertFalse(self.pager().handle_key(key))

    def test_quitting_with_a_capital_q(self) -> None:
        self.assertFalse(self.pager().handle_key(Key(CHAR, "Q")))


class FrameTests(unittest.TestCase):
    def test_only_the_rows_that_changed_are_repainted(self) -> None:
        pager = Pager([f"row {index}" for index in range(20)])
        first = pager._frame(30, 4)
        self.assertEqual(first.count("row 0"), 1)
        self.assertEqual(pager._frame(30, 4), "")
        pager.handle_key(Key(CHAR, "j"))
        second = pager._frame(30, 4)
        self.assertNotIn("row 0", second)
        self.assertIn("row 3", second)
        self.assertIn("2-4/20", second)

    def test_a_frame_stops_painting_rows_the_page_no_longer_has(self) -> None:
        source = [f"row {index}" for index in range(20)]
        pager = Pager(source)
        pager._frame(30, 6)
        source[:] = ["only one"]
        pager.invalidate()
        pager._frame(30, 6)
        source[:] = []
        frame = pager._frame(30, 6)
        self.assertIn("\x1b[2;1H", frame)

    def test_a_size_change_repaints_everything(self) -> None:
        pager = Pager(["a", "b", "c"])
        pager._frame(30, 6)
        self.assertIn("a", pager._frame(40, 6))


class RunTests(unittest.TestCase):
    def test_a_pipe_prints_the_rows_and_says_it_was_no_pager(self) -> None:
        sink = io.StringIO()
        pager = Pager(["one", "two", "three"], title="t")
        self.assertFalse(pager.run(stdin=io.StringIO(), stdout=sink))
        self.assertEqual(sink.getvalue(), "one\ntwo\nthree\n")

    def test_a_file_falls_back_to_plain_print_in_temp_dir(self) -> None:
        with tempfile.TemporaryDirectory() as worktop:
            path = Path(worktop) / "out.txt"
            with path.open("w", encoding="utf-8") as handle:
                self.assertFalse(Pager(["alpha", "beta\x1b[31m"]).run(stdin=io.StringIO(), stdout=handle))
            self.assertEqual(path.read_text(encoding="utf-8"), "alpha\nbeta\n")

    def test_show_lines_prints_when_there_is_no_terminal(self) -> None:
        sink = io.StringIO()
        self.assertFalse(show_lines(["a", "b"], title="history", stdout=sink))
        self.assertEqual(sink.getvalue(), "a\nb\n")

    def test_an_object_without_a_file_descriptor_is_printed(self) -> None:
        self.assertFalse(Pager(["a"]).run(stdin=object(), stdout=io.StringIO()))


@unittest.skipIf(_termios is None or not hasattr(os, "openpty"), "needs a pty")
class AlternateScreenTests(unittest.TestCase):
    def test_paging_on_a_terminal_leaves_only_the_main_screen(self) -> None:
        painted: list[str] = []

        class Screen:
            def isatty(self) -> bool:
                return True

            def write(self, text: str) -> int:
                painted.append(text)
                return len(text)

            def flush(self) -> None:
                pass

        class Keys:
            def __init__(self, fd: int) -> None:
                self.fd = fd

            def isatty(self) -> bool:
                return True

            def fileno(self) -> int:
                return self.fd

        def type_when_raw(slave: int, master: int, keys: bytes) -> None:
            """Keys typed before `run` takes the terminal raw would sit in the canonical
            buffer and be thrown away by the mode change, so wait for the raw mode."""
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                try:
                    lflag = _termios.tcgetattr(slave)[3]
                except OSError:
                    return
                if not lflag & _termios.ICANON:
                    os.write(master, keys)
                    return
                time.sleep(0.005)

        master, slave = os.openpty()
        before = _termios.tcgetattr(slave)
        try:
            writer = threading.Thread(target=type_when_raw, args=(slave, master, b"j\x1b[Bq"), daemon=True)
            writer.start()
            pager = Pager([f"row {index}" for index in range(40)], title="history")
            self.assertTrue(pager.run(stdin=Keys(slave), stdout=Screen()))
            writer.join(5.0)
            self.assertEqual(_termios.tcgetattr(slave)[:6], before[:6])
            self.assertIn(pager.top, (1, 2))
        finally:
            os.close(master)
            os.close(slave)
        screen = "".join(painted)
        self.assertEqual(screen.count(ALT_SCREEN_ON), 1)
        self.assertIn("row 0", screen)
        self.assertIn("history", strip_ansi(screen))
        self.assertTrue(screen.endswith(SHOW_CURSOR + ALT_SCREEN_OFF))


if __name__ == "__main__":
    unittest.main()
