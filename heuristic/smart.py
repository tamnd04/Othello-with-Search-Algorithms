"""Weighted positional heuristic with customizable weights.
"""
from __future__ import annotations

from othello.constants import BOARD_CORNERS
from othello.engine import OthelloGame
from heuristic.base import OthelloHeuristic


class SmartHeuristic(OthelloHeuristic):
    """Weighted positional heuristic for strong Othello play.

    Evaluates the board from `game.current_player`'s perspective across
    three factors (in descending priority):

    1. **Corners** (+1000 per corner captured, -1000 per corner conceded) -
       corners are permanent and the most valuable squares on the board.
    2. **X-squares** (-300 per X-square owned, +300 per X-square conceded) -
       X-squares next to un-taken corners risk gifting that corner to the
       opponent.
    3. **Disc delta** (x10) - raw piece-count tiebreaker.

    Class Attributes:
        CORNERS:   The four corner squares.
        X_SQUARES: The four diagonal corner-neighbour squares.

    Example::

        h     = SmartHeuristic()
        score = h(game)  # higher = better for current player
    """

    CORNERS:   frozenset[tuple[int, int]] = BOARD_CORNERS
    X_SQUARES: frozenset[tuple[int, int]] = frozenset({(1, 1), (1, 6), (6, 1), (6, 6)})

    def evaluate(self, game: OthelloGame) -> float:
        """Evaluate the board using corners, X-squares, and disc delta.

        Args:
            game: Current game state; `game.current_player` defines "us".

        Returns:
            A float score; higher is better for the current player.
        """
        player   = game.current_player
        opponent = game.opponent(player)

        # Corner advantage - permanent squares that dominate edge control.
        my_corners  = sum(1 for c in self.CORNERS   if game.get_cell(*c) == player)
        opp_corners = sum(1 for c in self.CORNERS   if game.get_cell(*c) == opponent)

        # X-square penalty - occupying these before the corner is taken is risky.
        my_x  = sum(1 for x in self.X_SQUARES if game.get_cell(*x) == player)
        opp_x = sum(1 for x in self.X_SQUARES if game.get_cell(*x) == opponent)

        # Raw disc-count advantage used as a tiebreaker.
        counts     = game.count_discs()
        disc_delta = counts[player] - counts[opponent]

        return float(
            (my_corners - opp_corners) * 1000
            - (my_x - opp_x) * 300
            + disc_delta * 10
        )

    def name(self) -> str:
        """Return the heuristic label.
        """
        return "SmartHeuristic"
