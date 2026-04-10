"""Unit tests for the Othello game engine (`othello.engine`).

Covers:
- Initial board setup and disc counts.
- Legal move generation from the starting position.
- Disc-flipping correctness after a single move.
- Undo / history restoration.
- Rejection of illegal move coordinates.
"""
from __future__ import annotations

import unittest

from othello.constants import BLACK, EMPTY, WHITE
from othello.engine import OthelloGame


class TestOthelloEngine(unittest.TestCase):
    """Functional tests for `OthelloGame`."""

    def setUp(self) -> None:
        """Create a fresh game instance before every test method."""
        self.game = OthelloGame()

    # ------------------------------------------------------------------
    # Initial State
    # ------------------------------------------------------------------

    def test_initial_setup(self) -> None:
        """The board should start with 2 Black, 2 White, and 60 empty squares."""
        scores = self.game.count_discs()
        self.assertEqual(scores[BLACK], 2)
        self.assertEqual(scores[WHITE], 2)
        self.assertEqual(scores[EMPTY], 60)
        # Black always moves first in standard Othello.
        self.assertEqual(self.game.current_player, BLACK)

    def test_initial_legal_moves(self) -> None:
        """From the opening position Black should have exactly four legal moves."""
        legal    = set(self.game.legal_moves().keys())
        expected = {(2, 3), (3, 2), (4, 5), (5, 4)}
        self.assertEqual(legal, expected)

    # ------------------------------------------------------------------
    # Move Application
    # ------------------------------------------------------------------

    def test_apply_move_flips_disc(self) -> None:
        """Playing (2, 3) should flip the White disc at (3, 3) to Black."""
        outcome = self.game.apply_move(2, 3)
        # The move was made by Black.
        self.assertEqual(outcome.player, BLACK)
        # The target square is now Black.
        self.assertEqual(self.game.get_cell(2, 3), BLACK)
        # The flanked White disc has been flipped.
        self.assertEqual(self.game.get_cell(3, 3), BLACK)
        # Turn should have passed to White.
        self.assertEqual(self.game.current_player, WHITE)

    # ------------------------------------------------------------------
    # Undo / History
    # ------------------------------------------------------------------

    def test_undo_restores_state(self) -> None:
        """Undoing a move should restore both the board and the active player exactly."""
        original_board = [row[:] for row in self.game.board]
        self.game.apply_move(2, 3)
        success = self.game.undo()
        self.assertTrue(success, "undo() should return True when history is non-empty")
        self.assertEqual(self.game.board, original_board)
        self.assertEqual(self.game.current_player, BLACK)

    # ------------------------------------------------------------------
    # Error Handling
    # ------------------------------------------------------------------

    def test_illegal_move_raises(self) -> None:
        """Attempting an illegal move should raise a `ValueError`."""
        with self.assertRaises(ValueError):
            self.game.apply_move(0, 0)  # Corner is not a legal opening move.


if __name__ == "__main__":
    unittest.main()
