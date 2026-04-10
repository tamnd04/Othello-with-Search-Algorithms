"""Abstract base class for all Othello position heuristics.

Every heuristic is a callable object: instances can be passed wherever a
`Callable[[OthelloGame], float]` is expected because `__call__` delegates
to the abstract :meth:`evaluate` method.

Example
-------------

    class MyHeuristic(OthelloHeuristic):
        def evaluate(self, game: OthelloGame) -> float:
            return float(game.count_discs()[game.current_player])

        def name(self) -> str:
            return "My Heuristic"

    h = MyHeuristic()
    score = h(game)          # via __call__
    score = h.evaluate(game) # direct call - identical result

Remarks
-------------

A heuristic is a scoring function that estimates how "good" a board position 
is for the current player, it converts a game state into a number. Since it's 
impossible to search every possible game outcome in real time, heuristics make 
that approximation cheap and fast.

Not every model agent needs a heuristic, only those that stop searching early 
and estimate the value of non-terminal positions. Others, such as Monte Carlo 
Tree Search (MCTS), simulates games to completion (or near-completion) and 
derives value from actual outcomes rather than a static estimate.

"""
from __future__ import annotations

from abc import ABC, abstractmethod

from othello.engine import OthelloGame


class OthelloHeuristic(ABC):
    """Abstract interface for an Othello board-evaluation heuristic.

    Subclasses implement :meth:`evaluate` to return a float score from the
    perspective of `game.current_player`. Higher scores indicate a more
    favourable position for that player.

    The class also acts as a callable via :meth:`__call__`, so heuristic
    instances are drop-in replacements anywhere a plain function of type
    `Callable[[OthelloGame], float]` is accepted.
    """

    # ------------------------------------------------------------------
    # Required interface
    # ------------------------------------------------------------------

    @abstractmethod
    def evaluate(self, game: OthelloGame) -> float:
        """Evaluate the board and return a score for the current player.

        Args:
            game: The current game state. Must not be permanently mutated.

        Returns:
            A float; higher values indicate a better position for
            `game.current_player`.
        """

    @abstractmethod
    def name(self) -> str:
        """Return a short human-readable label for this heuristic.
        Used in log output and agent descriptions.
        """

    # ------------------------------------------------------------------
    # Callable protocol
    # ------------------------------------------------------------------

    def __call__(self, game: OthelloGame) -> float:
        """Delegate to :meth:`evaluate` so the instance is callable.

        Args:
            game: The current game state.

        Returns:
            The result of :meth:`evaluate`.
        """
        return self.evaluate(game)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
