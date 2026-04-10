"""Disc-fraction based heuristic.
"""
from __future__ import annotations

from othello.constants import BOARD_SIZE, EMPTY
from othello.engine import OthelloGame
from heuristic.base import OthelloHeuristic


class SimpleHeuristic(OthelloHeuristic):
    """Disc-fraction heuristic.

    Scores the position as the fraction of placed discs owned by
    `game.current_player`. Values range from `0.0` (no discs) to
    `1.0` (all discs). This is fast to compute and sufficient for
    shallow search depths or as a lightweight test fixture.

    Example::

        h     = SimpleHeuristic()
        score = h(game)   # float in [0.0, 1.0]
    """

    def evaluate(self, game: OthelloGame) -> float:
        """Return the current player's share of all placed discs.

        Args:
            game: Current game state.

        Returns:
            A float in `[0.0, 1.0]`, or `0.0` if no discs are placed.
        """
        counts      = game.count_discs()
        total_discs = (BOARD_SIZE * BOARD_SIZE) - counts[EMPTY]
        if total_discs == 0:
            return 0.0
        return counts[game.current_player] / total_discs

    def name(self) -> str:
        """Return the heuristic label.
        """
        return "SimpleHeuristic"
