"""Unit tests for the Minimax search agent.

Covers tactical positions where the correct move is unambiguous so that
assertions act as a regression guard against search regressions.

Helper
------
`setup_custom_board` -- builds an `OthelloGame` from a compact board
string, making it straightforward to write position-based test cases.
"""
from __future__ import annotations

import unittest

from heuristic import SimpleHeuristic
from othello.constants import BLACK, BOARD_SIZE, EMPTY, WHITE
from othello.engine import OthelloGame
from search.minimax import get_best_move_minimax

# ---------------------------------------------------------------------------
# Board Setup Helper
# ---------------------------------------------------------------------------

def setup_custom_board(board_string: str, current_turn: str) -> OthelloGame:
    """Parse a multi-line board string into a live `OthelloGame` instance.

    Board characters:
        `B` -- Black disc
        `W` -- White disc
        `.` -- Empty square

    Args:
        board_string: An 8x8 grid of characters separated by newlines.
                      Leading/trailing whitespace on each row is stripped.
        current_turn: The player whose turn it is (`BLACK` or `WHITE`).

    Returns:
        A configured `OthelloGame` with a cleared history.
    """
    game                = OthelloGame()
    game.board          = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    game.current_player = current_turn

    rows = board_string.strip().split("\n")
    for r in range(BOARD_SIZE):
        row_str = rows[r].strip()
        for c in range(BOARD_SIZE):
            char = row_str[c]
            if char == "B":
                game.board[r][c] = BLACK
            elif char == "W":
                game.board[r][c] = WHITE

    return game


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

class TestMinimax(unittest.TestCase):
    """Tactical unit tests for `get_best_move_minimax`."""

    def setUp(self) -> None:
        """Create a shared `SimpleHeuristic` instance for all tests."""
        self.heuristic = SimpleHeuristic()

    def test_1_diagonal_capture(self) -> None:
        """Minimax should capture the top-right corner at (0, 7).

        Board layout: Black has an unobstructed diagonal line from (4, 3)
        to (1, 6) through two White discs. Placing at (0, 7) captures the
        corner and flips everything along the diagonal -- the unambiguously
        correct move at any search depth.
        """
        board_str = """
        ........
        ......W.
        .....W..
        ....B...
        ........
        ........
        ........
        ........
        """
        game = setup_custom_board(board_str, current_turn=BLACK)

        best_move, _score = get_best_move_minimax(
            game, max_depth=2, heuristic_func=self.heuristic
        )

        self.assertEqual(best_move, (0, 7), "Expected corner capture at (0, 7)")

    def test_2_endgame_corner_conversion(self) -> None:
        """Minimax should take the bottom-right corner at (7, 7) to win.

        The board is nearly full -- Black dominates but White occupies the
        bottom-right edge. Placing Black at (7, 7) flips the entire bottom-
        right cluster and secures the corner, converting all remaining White
        discs and winning the game.
        """
        board_str = """
        BBBBBBBB
        BBBBBBBB
        BBBBBBBB
        BBBBBBBW
        BBBBBBWW
        BBBBBB.W
        BBBBBB.W
        BBBBBB..
        """
        game = setup_custom_board(board_str, current_turn=BLACK)

        best_move, _score = get_best_move_minimax(
            game, max_depth=3, heuristic_func=self.heuristic
        )

        self.assertEqual(best_move, (7, 7), "Expected endgame corner at (7, 7)")


if __name__ == "__main__":
    unittest.main()
