"""Weighted positional heuristic with pre-defined weights.

Weight matrix references:
- https://github.com/Bryson-Lichtenberg/Othello/blob/main/evaluation.py
- https://github.com/yarinTabashi/Othello-AI/blob/master/heuristics.py
"""
from __future__ import annotations

from othello.constants import BOARD_SIZE
from othello.engine import OthelloGame
from heuristic.base import OthelloHeuristic


class PositionalHeuristic(OthelloHeuristic):
    """Weight-table heuristic.

    Each square is assigned a strategic weight that reflects its typical
    value in Othello. Corners score highest (+100); C-squares and
    X-squares score negatively because they tend to give the opponent
    access to corners.

    The evaluation is::

        score = Σ weight[r][c] for each disc owned by current_player
              - Σ weight[r][c] for each disc owned by opponent

    Class Attributes:
        WEIGHTS: 8x8 table of integer weights indexed by `[row][col]`.

    Example::

        h     = PositionalHeuristic()
        score = h(game)  # higher = better for current player
    """

    WEIGHTS: list[list[int]] = [
        [100, -20,  10,  5,  5,  10, -20, 100],
        [-20, -50,  -2, -2, -2,  -2, -50, -20],
        [ 10,  -2,   5,  1,  1,   5,  -2,  10],
        [  5,  -2,   1,  0,  0,   1,  -2,   5],
        [  5,  -2,   1,  0,  0,   1,  -2,   5],
        [ 10,  -2,   5,  1,  1,   5,  -2,  10],
        [-20, -50,  -2, -2, -2,  -2, -50, -20],
        [100, -20,  10,  5,  5,  10, -20, 100],
    ]

    def evaluate(self, game: OthelloGame) -> float:
        """Evaluate the board using the positional weight table.

        Args:
            game: Current game state; `game.current_player` defines "us".

        Returns:
            A float score; higher is better for the current player.
        """
        player   = game.current_player
        opponent = game.opponent(player)
        score    = 0

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                cell = game.get_cell(r, c)
                if cell == player:
                    score += self.WEIGHTS[r][c]
                elif cell == opponent:
                    score -= self.WEIGHTS[r][c]

        return float(score)

    def name(self) -> str:
        """Return the heuristic label.
        """
        return "PositionalHeuristic"
