from __future__ import annotations

import unittest

from ohar.terminal import display_width, make_theme, strip_ansi
from ohar.tui.dag import (
    CONDENSED_BOXES,
    CONDENSED_FAILURES,
    CONDENSED_LAYERS,
    SMALL_GRAPH_LANES,
    DagNode,
    FoldState,
    block_lines,
    dag_nodes,
    fold_keys,
    layers,
    summary_line,
    tree_lines,
)
from ohar.tui.roster import SubagentRoster

PLAIN = make_theme("aurora", color=False)
COLOR = make_theme("aurora", color=True, depth="truecolor")

A = DagNode("a", "read the plan", "running", (), "fast", "claude-x", 3.2, 120, "bash(ohar ls)")
B = DagNode("b", "write the port", "done", ("a",), "fast", "claude-x", 12.0, 3400, "edit_file(dag.py)")
C = DagNode("c", "verify tests", "pending", ("b",))
D = DagNode("d", "ship it", "failed", ("b", "c"), "slow", "gpt-9", 44.0, 900)
CHAIN = [A, B, C, D]


def rows_of(lines: list[str]) -> list[str]:
    return [strip_ansi(row) for row in lines]


def find(rows: list[str], needle: str) -> str:
    return next(row for row in rows if needle in row)


def graph(count: int) -> list[DagNode]:
    return [
        DagNode(f"n{at}", f"node {at}", "running" if at % 3 == 0 else "done", (() if not at else (f"n{at - 1}",)))
        for at in range(count)
    ]


class DagNodeTests(unittest.TestCase):
    def test_edges_and_facts_default_to_nothing(self) -> None:
        node = DagNode("n", "title", "done")
        self.assertEqual((node.parents, node.lane, node.model, node.seconds, node.tokens, node.detail), ((), "", "", 0.0, 0, ""))

    def test_a_node_is_mutable_and_slot(self) -> None:
        node = DagNode("n", "title", "done")
        node.status = "stuck"
        node.tokens = 12
        self.assertEqual((node.status, node.tokens), ("stuck", 12))
        self.assertTrue(hasattr(DagNode, "__slots__"))
        with self.assertRaises(AttributeError):
            node.not_a_field = 1  # type: ignore[attr-defined]


class LayersTests(unittest.TestCase):
    def test_orders_parents_first(self) -> None:
        self.assertEqual([[n.id for n in layer] for layer in layers(CHAIN)], [["a"], ["b"], ["c"], ["d"]])

    def test_one_layer_keeps_the_input_order(self) -> None:
        nodes = [
            DagNode("root", "root", "done"),
            DagNode("one", "one", "pending", ("root",)),
            DagNode("two", "two", "pending", ("root",)),
        ]
        self.assertEqual([[n.id for n in layer] for layer in layers(nodes)], [["root"], ["one", "two"]])

    def test_unknown_and_self_parents_are_ignored(self) -> None:
        nodes = [DagNode("solo", "solo", "done", ("ghost", "solo"))]
        self.assertEqual([[n.id for n in layer] for layer in layers(nodes)], [["solo"]])

    def test_a_cycle_still_places_every_node(self) -> None:
        nodes = [DagNode("x", "x", "done", ("y",)), DagNode("y", "y", "done", ("x",))]
        placed = [n.id for layer in layers(nodes) for n in layer]
        self.assertEqual(sorted(placed), ["x", "y"])
        self.assertTrue(all(layer for layer in layers(nodes)))

    def test_a_longer_cycle_keeps_its_order(self) -> None:
        nodes = [DagNode("p", "p", "done", ("r",)), DagNode("q", "q", "done", ("p",)), DagNode("r", "r", "done", ("q",))]
        placed = [n.id for layer in layers(nodes) for n in layer]
        self.assertEqual(sorted(placed), ["p", "q", "r"])
        self.assertEqual(layers(nodes)[0][0].id, "p")  # the first listed node opens the ring

    def test_duplicate_ids_keep_the_first(self) -> None:
        nodes = [DagNode("k", "first", "done"), DagNode("k", "second", "running")]
        self.assertEqual([n.title for layer in layers(nodes) for n in layer], ["first"])

    def test_empty_graph_has_no_layers(self) -> None:
        self.assertEqual(layers([]), [])


class DagNodesTests(unittest.TestCase):
    def test_roster_tasks_become_nodes(self) -> None:
        roster = SubagentRoster()
        roster.apply({"type": "start", "id": "1", "title": "read", "lane": "fast", "model": "m1"})
        roster.apply({"type": "tool", "id": "1", "name": "read_file"})
        roster.apply({"type": "tool", "id": "1", "name": "grep"})
        roster.apply({"type": "done", "id": "1", "seconds": 2.5, "chars": 400})
        roster.apply({"type": "start", "id": "2", "title": "write"})
        roster.apply({"type": "error", "id": "2", "error": "timed out"})
        nodes = dag_nodes(roster)
        self.assertEqual([node.id for node in nodes], ["1", "2"])
        self.assertEqual(nodes[0].status, "done")
        self.assertEqual(nodes[0].detail, "grep")  # the last tool line
        self.assertEqual((nodes[0].seconds, nodes[0].tokens), (2.5, 400))
        self.assertEqual(nodes[0].lane, "fast")
        self.assertEqual(nodes[0].model, "m1")
        self.assertEqual(nodes[1].status, "failed")
        self.assertEqual(nodes[1].detail, "timed out")

    def test_a_running_task_has_no_detail_yet(self) -> None:
        roster = SubagentRoster()
        roster.apply({"type": "start", "id": "1", "title": "work"})
        node = dag_nodes(roster)[0]
        self.assertEqual((node.status, node.detail), ("running", ""))

    def test_an_unnamed_task_still_reads(self) -> None:
        roster = SubagentRoster()
        roster.apply({"type": "start", "id": "7"})
        node = dag_nodes(roster)[0]
        self.assertEqual((node.id, node.title), ("7", "7"))

    def test_control_characters_are_stripped(self) -> None:
        roster = SubagentRoster()
        roster.apply({"type": "start", "id": "1", "title": "\x1b[31msu\nbom\x00"})
        node = dag_nodes(roster)[0]
        self.assertEqual(node.title, "su bom")
        for junk in ("\x1b", "\x00", "\n"):
            self.assertNotIn(junk, node.title)

    def test_no_tasks_no_nodes(self) -> None:
        self.assertEqual(dag_nodes(SubagentRoster()), [])


class FoldStateTests(unittest.TestCase):
    def test_an_unset_row_answers_each_default_its_own_way(self) -> None:
        fold = FoldState()
        self.assertTrue(fold.is_open("k", True))
        self.assertFalse(fold.is_open("k", False))

    def test_toggle_flips_from_open(self) -> None:
        fold = FoldState()
        fold.toggle("k")
        self.assertFalse(fold.is_open("k", True))
        fold.toggle("k")
        self.assertTrue(fold.is_open("k", False))

    def test_an_override_wins_over_a_toggle(self) -> None:
        fold = FoldState()
        fold.override("k", False)
        fold.toggle("k")
        self.assertTrue(fold.is_open("k", False))
        fold.override("k", True)
        self.assertTrue(fold.is_open("k", False))

    def test_a_key_is_only_kept_once(self) -> None:
        fold = FoldState()
        fold.override("k", True)
        fold.override("k", False)
        self.assertFalse(fold.is_open("k", True))


class FoldKeysTests(unittest.TestCase):
    def test_lists_the_rows_a_fold_can_change(self) -> None:
        self.assertEqual(fold_keys(CHAIN), ["layer:0", "a", "b"])

    def test_a_leaf_with_nothing_under_it_is_not_foldable(self) -> None:
        nodes = [DagNode("only", "only", "done")]
        self.assertEqual(fold_keys(nodes), ["layer:0"])

    def test_every_key_drives_a_row(self) -> None:
        fold = FoldState()
        for key in fold_keys(CHAIN):
            fold.toggle(key)
        rows = rows_of(tree_lines(CHAIN, 90, 40, PLAIN, fold=fold))
        self.assertEqual(len(rows), 1)  # the layer shut: its label is all that is left
        self.assertIn("▸ layer 1", rows[0])

    def test_empty_graph(self) -> None:
        self.assertEqual(fold_keys([]), [])


class TreeLinesTests(unittest.TestCase):
    def test_the_layer_row_counts_what_it_draws(self) -> None:
        rows = rows_of(tree_lines(CHAIN, 90, 40, PLAIN))
        self.assertIn("layer 1 · 4 nodes · 1 running · 1 failed · 1 pending · 1 done", rows[0])

    def test_the_columns_show_lane_model_status_and_tokens(self) -> None:
        head = find(rows_of(tree_lines(CHAIN, 90, 40, PLAIN)), "read the plan")
        for cell in ("fast", "claude-x", "running 3.20s", "↓120"):
            self.assertIn(cell, head)

    def test_columns_line_up_across_rows(self) -> None:
        rows = rows_of(tree_lines(CHAIN, 90, 40, PLAIN))
        starts = {row.index("fast") for row in rows if "fast" in row and "claude-x" in row}
        self.assertEqual(len(starts), 1)

    def test_detail_rows_and_nesting(self) -> None:
        rows = rows_of(tree_lines(CHAIN, 90, 40, PLAIN))
        self.assertIn("› bash(ohar ls)", find(rows, "bash(ohar ls)"))
        self.assertTrue(find(rows, "ship it").startswith(" " * 12))

    def test_moving_work_opens_and_quiet_work_shuts(self) -> None:
        nodes = [
            DagNode("busy", "busy", "running", detail="read_file(a)"),
            DagNode("rest", "rest", "done", detail="read_file(b)"),
        ]
        rows = rows_of(tree_lines(nodes, 90, 40, PLAIN))
        joined = "\n".join(rows)
        self.assertIn("▾ ▶ busy", find(rows, "busy"))  # running: open, its tool line shown
        self.assertIn("› read_file(a)", joined)
        self.assertIn("▸ ✓ rest", find(rows, "rest"))  # done: shut, its one line counted instead
        self.assertNotIn("› read_file(b)", joined)
        self.assertIn("1 detail line", find(rows, "detail line"))

    def test_a_stuck_node_holds_its_parent_open(self) -> None:
        nodes = [
            DagNode("top", "top", "done", detail="tool(x)"),
            DagNode("under", "under", "stuck", ("top",)),
        ]
        rows = rows_of(tree_lines(nodes, 90, 40, PLAIN))
        self.assertIn("▾ ✓ top", rows[1])
        self.assertIn("› tool(x)", "\n".join(rows))

    def test_width_is_never_exceeded(self) -> None:
        for width in (14, 24, 40, 72, 90):
            for row in tree_lines(CHAIN, width, 40, PLAIN):
                self.assertLessEqual(display_width(row), width)

    def test_the_cap_cuts_with_a_count_row(self) -> None:
        fold = FoldState()
        for key in fold_keys(graph(20)):
            fold.override(key, True)  # nothing may shut, so only the cut can bring it to the cap
        rows = rows_of(tree_lines(graph(20), 90, 3, PLAIN, fold=fold))
        self.assertEqual(len(rows), 3)
        self.assertIn("… 19 more nodes", rows[-1])

    def test_the_tree_shuts_to_fit_before_it_cuts(self) -> None:
        rows = rows_of(tree_lines(graph(20), 90, 6, PLAIN))
        self.assertLessEqual(len(rows), 6)
        self.assertFalse(any("more nodes" in row and row.lstrip().startswith("…") for row in rows))

    def test_a_fold_override_beats_the_state_default(self) -> None:
        fold = FoldState()
        fold.override("a", False)
        rows = rows_of(tree_lines(CHAIN, 90, 40, PLAIN, fold=fold))
        self.assertIn("▸ ▶ read the plan", find(rows, "read the plan"))
        self.assertNotIn("bash(ohar ls)", "\n".join(rows))
        self.assertIn("more nodes", "\n".join(rows))

    def test_a_locked_row_is_never_shut_to_fit(self) -> None:
        fold = FoldState()
        fold.override("layer:0", True)
        fold.override("a", True)
        rows = rows_of(tree_lines(CHAIN, 90, 3, PLAIN, fold=fold))
        self.assertIn("read the plan", rows[1])

    def test_the_selected_row_is_one_painted_bar(self) -> None:
        lines = tree_lines(CHAIN, 60, 40, COLOR, selected="a")
        marked = next(row for row in lines if "read the plan" in strip_ansi(row))
        self.assertIn("\x1b[1m", marked)
        self.assertEqual(display_width(marked), 60)
        unmarked = next(row for row in lines if "ship it" in strip_ansi(row))
        self.assertLess(display_width(unmarked), 60)

    def test_ascii_glyphs(self) -> None:
        joined = "\n".join(rows_of(tree_lines(CHAIN, 90, 40, PLAIN, glyphs="ascii")))
        for mark in ("▾", "▸", "▶", "✓", "✗", "›", "↓"):
            self.assertNotIn(mark, joined)
        self.assertIn("> read the plan", joined)
        self.assertIn("+ write the port", joined)

    def test_junk_text_is_cleaned(self) -> None:
        nodes = [DagNode("n", "\x1b[32mgreen\r\nvalue\x00 here", "failed", detail="\x07tool(x)")]
        joined = "".join(rows_of(tree_lines(nodes, 90, 10, PLAIN)))
        for junk in ("\x1b", "\x07", "\x00", "\r"):
            self.assertNotIn(junk, joined)
        self.assertIn("green value here", joined)
        self.assertIn("› tool(x)", joined)

    def test_empty_and_degenerate_arguments(self) -> None:
        self.assertEqual(tree_lines([], 90, 40, PLAIN), [])
        self.assertEqual(tree_lines(CHAIN, 0, 40, PLAIN), [])
        self.assertEqual(tree_lines(CHAIN, -5, 40, PLAIN), [])
        self.assertEqual(tree_lines(CHAIN, 90, 0, PLAIN), [])
        self.assertEqual(tree_lines(CHAIN, 90, -1, PLAIN), [])


class SummaryLineTests(unittest.TestCase):
    def test_counts_by_state(self) -> None:
        self.assertEqual(summary_line(CHAIN, PLAIN), "DAG · 4 nodes · 1 running · 1 failed · 1 pending · 1 done")

    def test_aliases_read_as_their_state(self) -> None:
        nodes = [DagNode("n", "n", "completed"), DagNode("m", "m", "queued"), DagNode("k", "k", "timeout")]
        self.assertEqual(summary_line(nodes, PLAIN), "DAG · 3 nodes · 1 stuck · 1 pending · 1 done")

    def test_no_nodes_no_line(self) -> None:
        self.assertEqual(summary_line([], PLAIN), "")

    def test_painted(self) -> None:
        line = summary_line([DagNode("n", "n", "done")], COLOR)
        self.assertIn("\x1b[38;2", line)
        self.assertEqual(strip_ansi(line), "DAG · 1 node · 1 done")


class BlockLinesTests(unittest.TestCase):
    def test_a_small_graph_prints_whole(self) -> None:
        rows = rows_of(block_lines(CHAIN, "plan", 90, PLAIN))
        self.assertEqual(rows[0], "  dag · plan")
        self.assertTrue(any("ship it" in row for row in rows))
        self.assertFalse(any("more nodes" in row for row in rows))

    def test_condensed_print_counts_layers_and_notes_the_rest(self) -> None:
        count = SMALL_GRAPH_LANES + 8
        rows = rows_of(block_lines(graph(count), "extend +3", 90, PLAIN))
        self.assertTrue(any(f"DAG · {count} nodes" in row for row in rows))
        self.assertTrue(any("layer 1 · 1 node · 1 running" in row for row in rows))
        self.assertTrue(any(f"… {count - CONDENSED_LAYERS} more layers" in row for row in rows))
        self.assertTrue(any("more nodes · /dag shows the whole graph" in row for row in rows))
        self.assertLessEqual(len([row for row in rows if row.startswith("  layer")]), CONDENSED_LAYERS)

    def test_condensed_print_draws_frontier_boxes(self) -> None:
        rows = rows_of(block_lines(graph(20), "plan", 70, PLAIN))
        self.assertEqual(len([row for row in rows if row.startswith("  ┌")]), CONDENSED_BOXES)
        self.assertEqual(len([row for row in rows if row.startswith("  └")]), CONDENSED_BOXES)
        self.assertTrue(any(row.startswith("  │") for row in rows))
        self.assertIn("┌ ▶ n0 · node 0", "\n".join(rows))  # the frontier is the work moving now
        self.assertIn("… 14 more nodes · /dag shows the whole graph", rows[-1])
        for row in rows:
            self.assertLessEqual(display_width(row), 70)

    def test_a_fan_out_prints_boxes_up_to_the_bound(self) -> None:
        wide = [DagNode("root", "root", "done")] + [
            DagNode(f"n{at}", f"node {at}", "running", ("root",)) for at in range(CONDENSED_BOXES + 6)
        ]
        rows = rows_of(block_lines(wide, "plan", 70, PLAIN))
        self.assertEqual(len([row for row in rows if row.startswith("  ┌")]), CONDENSED_BOXES)

    def test_an_end_prints_only_the_unfinished(self) -> None:
        rows = rows_of(block_lines(graph(20), "end · done", 90, PLAIN, ended=True))
        self.assertTrue(any("· running" in row for row in rows))
        self.assertFalse(any(row.startswith("  layer") for row in rows))
        self.assertFalse(any("┌" in row for row in rows))
        self.assertTrue(rows[0].startswith("  dag · end"))

    def test_an_end_notes_the_nodes_it_left_out(self) -> None:
        broken = [DagNode(f"n{at}", f"node {at}", "failed") for at in range(CONDENSED_FAILURES + 5)]
        rows = rows_of(block_lines(broken, "end · failed", 90, PLAIN, ended=True))
        self.assertEqual(len([row for row in rows if row.startswith("  ✗")]), CONDENSED_FAILURES)
        self.assertIn("… 5 more", rows[-1])

    def test_a_narrow_block_names_nodes_instead_of_boxing_them(self) -> None:
        rows = rows_of(block_lines(graph(20), "plan", 12, PLAIN))
        self.assertFalse(any("┌" in row for row in rows))
        self.assertTrue(any("n0" in row for row in rows))
        for row in rows:
            self.assertLessEqual(display_width(row), 12)

    def test_a_wider_block_draws_the_box(self) -> None:
        rows = rows_of(block_lines(graph(20), "plan", 24, PLAIN))
        self.assertTrue(any(row.startswith("  ┌") for row in rows))
        for row in rows:
            self.assertLessEqual(display_width(row), 24)

    def test_ascii_glyphs_box_in_ascii(self) -> None:
        rows = rows_of(block_lines(graph(20), "plan", 60, PLAIN, glyphs="ascii"))
        joined = "\n".join(rows)
        for mark in ("┌", "└", "│", "─", "▶"):
            self.assertNotIn(mark, joined)
        self.assertIn("+", joined)

    def test_a_bad_small_cap_falls_back_to_the_default(self) -> None:
        rows = rows_of(block_lines(CHAIN, "plan", 90, PLAIN, small_cap=0))
        self.assertTrue(any("ship it" in row for row in rows))
        wide = rows_of(block_lines(graph(SMALL_GRAPH_LANES + 1), "plan", 90, PLAIN, small_cap=-3))
        self.assertTrue(any("more nodes" in row for row in wide))

    def test_the_graph_at_its_cap_still_prints_whole(self) -> None:
        rows = rows_of(block_lines(graph(SMALL_GRAPH_LANES), "plan", 90, PLAIN))
        self.assertFalse(any("more nodes" in row for row in rows))

    def test_empty_or_degenerate(self) -> None:
        self.assertEqual(block_lines([], "plan", 90, PLAIN), [])
        self.assertEqual(block_lines(CHAIN, "plan", 0, PLAIN), [])

    def test_junk_title_is_cleaned(self) -> None:
        rows = rows_of(block_lines(CHAIN, "\x1b[31mde\x00fined", 90, PLAIN))
        self.assertEqual(rows[0], "  dag · defined")


if __name__ == "__main__":
    unittest.main()
