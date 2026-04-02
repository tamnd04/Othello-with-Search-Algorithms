import unittest

from othello.constants import BLACK, EMPTY, WHITE
from othello.engine import OthelloGame


class TestOthelloEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.game = OthelloGame()

    def test_initial_setup(self) -> None:
        scores = self.game.count_discs()
        self.assertEqual(scores[BLACK], 2)
        self.assertEqual(scores[WHITE], 2)
        self.assertEqual(scores[EMPTY], 60)
        self.assertEqual(self.game.current_player, BLACK)

    def test_initial_legal_moves(self) -> None:
        legal = set(self.game.legal_moves().keys())
        expected = {(2, 3), (3, 2), (4, 5), (5, 4)}
        self.assertEqual(legal, expected)

    def test_apply_move_flips_disc(self) -> None:
        outcome = self.game.apply_move(2, 3)
        self.assertEqual(outcome.player, BLACK)
        self.assertEqual(self.game.get_cell(2, 3), BLACK)
        self.assertEqual(self.game.get_cell(3, 3), BLACK)
        self.assertEqual(self.game.current_player, WHITE)

    def test_undo_restores_state(self) -> None:
        original_board = [row[:] for row in self.game.board]
        self.game.apply_move(2, 3)
        self.assertTrue(self.game.undo())
        self.assertEqual(self.game.board, original_board)
        self.assertEqual(self.game.current_player, BLACK)

    def test_illegal_move_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.game.apply_move(0, 0)


if __name__ == "__main__":
    unittest.main()
