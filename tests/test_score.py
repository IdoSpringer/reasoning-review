#!/usr/bin/env python3
"""Unit tests for eval/score.py deterministic scoring. CPU/stdlib only."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
import score  # noqa: E402


REF = """ShowAxes(false)
ShowGrid(false)
A = Point({2, 3})
B = Point({6, 3})
c1 = Circle(A, B)
c2 = Circle(B, A)
C = Intersect(c1, c2, 1)
Poly = Polygon(A, B, C)
"""

GOOD = REF  # identical -> seq_match True, sims 1.0

RENAMED = """ShowAxes(false)
ShowGrid(false)
P = Point({2, 3})
Q = Point({6, 3})
d1 = Circle(P, Q)
d2 = Circle(Q, P)
R = Intersect(d1, d2, 1)
Tri = Polygon(P, Q, R)
"""

MALFORMED = """ShowAxes(false
A = Point(2, 3
this is not code
"""

EMPTY = ""


class TestScore(unittest.TestCase):
    def test_extract_last_block(self):
        text = "reasoning ...\n```geogebra\nA = Point({0,0})\n```\nmore\n```geogebra\nB = Point({1,1})\n```"
        self.assertEqual(score.extract_code(text), "B = Point({1,1})")

    def test_extract_none(self):
        self.assertEqual(score.extract_code("no code here"), "")

    def test_identical(self):
        m = score.structural_metrics(GOOD, REF)
        self.assertTrue(m["seq_match"])
        self.assertEqual(m["obj_sim"], 1.0)
        self.assertEqual(m["cmd_sim"], 1.0)

    def test_renamed_same_structure(self):
        m = score.structural_metrics(RENAMED, REF)
        self.assertFalse(m["seq_match"])          # names differ
        self.assertEqual(m["obj_sim"], 0.0)       # object pairs differ
        self.assertEqual(m["cmd_sim"], 1.0)       # command names identical

    def test_numeric_normalization(self):
        alt = REF.replace("{2, 3}", "{2.0, 3.00}").replace("{6, 3}", "{6.0,3}")
        m = score.structural_metrics(alt, REF)
        self.assertTrue(m["seq_match"])

    def test_valid_and_invalid(self):
        good_stats = score.code_stats(GOOD)
        self.assertTrue(good_stats["heuristic_valid"])
        self.assertEqual(good_stats["n_invalid_lines"], 0)
        bad_stats = score.code_stats(MALFORMED)
        self.assertFalse(bad_stats["heuristic_valid"])
        self.assertGreater(bad_stats["n_invalid_lines"], 0)
        empty_stats = score.code_stats(EMPTY)
        self.assertFalse(empty_stats["heuristic_valid"])

    def test_hex_colors_not_treated_as_comments(self):
        code = 'c = Circle(A, B)\nSetColor(c, "#2E86C1")\nSetFilling(c, 0.3)'
        stats = score.code_stats(code)
        self.assertEqual(stats["n_invalid_lines"], 0)
        self.assertTrue(stats["heuristic_valid"])
        m = score.structural_metrics(code, code)
        self.assertTrue(m["seq_match"])

    def test_lowercase_builtins_valid(self):
        # GeoGebra lowercase functions: sqrt(), y() coordinate extraction
        code = "h = sqrt(L^2 - r^2)\ns = y(I)\nA = Point({1, 2})"
        stats = score.code_stats(code)
        self.assertEqual(stats["n_invalid_lines"], 0)
        self.assertTrue(stats["heuristic_valid"])

    def test_garbage_lowercase_still_invalid(self):
        stats = score.code_stats("foo = bar(1, 2)\n")
        self.assertFalse(stats["heuristic_valid"])

    def test_known_commands_cover_ggbench_vocab(self):
        for cmd in ("ShowAxes", "SetCaption", "PerpendicularBisector", "AngleBisector", "Rotate"):
            self.assertIn(cmd, score.KNOWN_COMMANDS)


if __name__ == "__main__":
    unittest.main()
