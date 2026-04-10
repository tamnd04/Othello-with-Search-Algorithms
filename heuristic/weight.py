"""Weighted positional heuristic with pre-defined weights.

Weight matrix references: [Baier, Hendrik & Winands, Mark. (2018). MCTS-Minimax Hybrids with State Evaluations. Journal of Artificial Intelligence Research. 62. 193-231. 10.1613/jair.1.11208.](https://www.researchgate.net/figure/The-move-ordering-for-Othello_fig1_326235384)
"""
from __future__ import annotations

from othello.constants import BOARD_SIZE
from othello.engine import OthelloGame
from heuristic.base import OthelloHeuristic


class WeightHeuristic(OthelloHeuristic):
    """Alternative weight-table heuristic.

    Scores the position as the sum of weights for the current player's
    discs minus the sum of weights for the opponent's discs::

        score = Σ WEIGHTS[r][c] for each disc owned by current_player
              - Σ WEIGHTS[r][c] for each disc owned by opponent

    Class Attributes:
        WEIGHTS: 8x8 table of integer weights indexed by `[row][col]`.

    Example::

        h     = WeightHeuristic()
        score = h(game)  # higher = better for current player
    """

    WEIGHTS: list[list[int]] = [
        [5, 2, 4, 3, 3, 4, 2, 5],
        [2, 1, 3, 3, 3, 3, 1, 2],
        [4, 3, 4, 4, 4, 4, 3, 4],
        [3, 3, 4, 4, 4, 4, 3, 3],
        [3, 3, 4, 4, 4, 4, 3, 3],
        [4, 3, 4, 4, 4, 4, 3, 4],
        [2, 1, 3, 3, 3, 3, 1, 2],
        [5, 2, 4, 3, 3, 4, 2, 5],
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
        return "WeightHeuristic"
