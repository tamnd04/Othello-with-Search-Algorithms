"""Greedy AI agent for Othello, conforming to the `OthelloAgent` interface.

Provides a lightweight baseline that selects moves using a simple greedy
heuristic. Three difficulty levels are available:

  - easy:   Random legal move.
  - medium: Maximises immediate disc flips with a positional tiebreaker.
  - hard:   One-ply look-ahead evaluated by an injected `OthelloHeuristic`.
"""
from __future__ import annotations

import random
from typing import Optional

from heuristic import OthelloHeuristic, SmartHeuristic
from othello.constants import BOARD_CORNERS
from othello.engine import Coord, OthelloGame
from search.model import OthelloAgent


class GreedyAgent(OthelloAgent):
    """Greedy Othello agent with three configurable difficulty levels.

    This agent does not perform any deep tree search. It is designed as a
    fast, readable baseline against which the Minimax and MCTS agents are
    benchmarked.

    Class Attributes:
        CORNERS: The four corner squares - highest-value positions on an
                 Othello board because they can never be flipped once taken.

    Example::

        agent = GreedyAgent(difficulty="hard")
        move  = agent.choose_move(game)
    """

    CORNERS: frozenset[Coord] = BOARD_CORNERS

    def __init__(self, difficulty: str = "hard", heuristic: Optional[OthelloHeuristic] = None) -> None:
        """Initialise the agent.

        Args:
            difficulty: One of `"easy"`, `"medium"`, or `"hard"`
                        (case-insensitive). Defaults to `"hard"`.
            heuristic:  Board-evaluation heuristic used by the hard
                        difficulty's one-ply look-ahead. Defaults to
                        :class:`~heuristic.SmartHeuristic`.
        """
        self._difficulty: str              = difficulty.strip().lower()
        self._heuristic:  OthelloHeuristic = heuristic if heuristic is not None else SmartHeuristic()

    # ------------------------------------------------------------------
    # OthelloAgent interface
    # ------------------------------------------------------------------

    def choose_move(self, game: OthelloGame) -> Optional[Coord]:
        """Return the best move for the current player at the configured difficulty.

        Args:
            game: The live game state. `game.current_player` identifies
                  which colour this agent is playing.

        Returns:
            A `(row, col)` coordinate, or `None` if no legal moves exist.
        """
        legal = game.legal_moves(game.current_player)
        if not legal:
            return None

        if self._difficulty == "easy":
            # Easy: pick any legal move at random - useful for testing.
            return random.choice(list(legal.keys()))

        if self._difficulty == "hard":
            return self._hard_move(game)

        # Default to medium difficulty for any unrecognised difficulty string.
        return self._medium_move(game)

    def agent_name(self) -> str:
        """Return a label that includes the configured difficulty and heuristic (if applicable).
        """
        if self._difficulty == "hard" and self._heuristic is not None:
            return f"Greedy ({self._difficulty.capitalize()}, {self._heuristic.__class__.__name__})"
        return f"Greedy ({self._difficulty.capitalize()})"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _medium_move(self, game: OthelloGame) -> Coord:
        """Medium-difficulty strategy: maximise immediate disc flips.

        Uses a three-key sort: corners first, then edge squares, then the raw
        flip count. No lookahead is performed.

        Args:
            game: The live game state.

        Returns:
            The highest-scoring legal `(row, col)` coordinate.
        """
        legal = game.legal_moves(game.current_player)

        def score(item: tuple[Coord, list[Coord]]) -> tuple[int, int, int]:
            move, flipped = item
            row, col = move
            is_corner = 1 if move in self.CORNERS else 0
            # Edges (row/col == 0 or 7) are stable once placed.
            is_edge = 1 if row in {0, 7} or col in {0, 7} else 0
            return (is_corner, is_edge, len(flipped))

        return max(legal.items(), key=score)[0]

    def _hard_move(self, game: OthelloGame) -> Coord:
        """Hard-difficulty strategy: one-ply look-ahead evaluated by the injected heuristic.

        For every legal move the agent simulates the resulting position and
        scores it with `self._heuristic` from the original player's perspective.

        Args:
            game: The live game state.

        Returns:
            The legal `(row, col)` coordinate with the highest heuristic score.
        """
        legal  = game.legal_moves(game.current_player)
        player = game.current_player

        best_move:  Optional[Coord] = None
        best_score: Optional[float] = None

        for move in legal:
            # Clone the game so simulation does not touch the real board state.
            simulated = game.clone()
            simulated.apply_move(move[0], move[1])
            # Override the active player so the heuristic always evaluates
            # from the original mover's perspective, regardless of passes.
            simulated.current_player = player
            score = self._heuristic(simulated)

            if best_score is None or score > best_score:
                best_score = score
                best_move  = move

        # `legal` is non-empty (checked at the top of `choose_move`),
        # so `best_move` is always set by this point.
        return best_move if best_move is not None else next(iter(legal.keys()))
