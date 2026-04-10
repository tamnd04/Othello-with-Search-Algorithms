"""Abstract base class for all Othello AI agents.

Defines the common interface that every concrete agent must implement,
enabling polymorphic use in tournaments, benchmarks, and the game UI.

Usage Example
-------------
Every agent subclass is interchangeable through this interface::

    agents: list[OthelloAgent] = [GreedyAgent(), MCTSAgent(), MinimaxAgent()]
    for agent in agents:
        move = agent.choose_move(game)
        print(f"{agent.agent_name()} chose {move}")
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from othello.engine import Coord, OthelloGame


class OthelloAgent(ABC):
    """Abstract interface for an Othello AI agent.

    Concrete subclasses must implement `choose_move` and `agent_name`.
    All other algorithm-specific state (depth, iterations, heuristics, etc.)
    is encapsulated in the subclass constructor, keeping the call-site uniform.
    """

    # ------------------------------------------------------------------
    # Required interface
    # ------------------------------------------------------------------

    @abstractmethod
    def choose_move(self, game: OthelloGame) -> Optional[Coord]:
        """Select and return a move for the current player.

        The implementation must not permanently mutate `game`. Any
        look-ahead should be performed on a clone (`game.clone()`) or
        reversed via `game.undo()`.

        Args:
            game: The live game state. `game.current_player` identifies
                  which colour this agent is controlling.

        Returns:
            A `(row, col)` coordinate representing the chosen move,
            or `None` if no legal moves are available.
        """

    @abstractmethod
    def agent_name(self) -> str:
        """Return a short, human-readable label for this agent.

        Used in tournament output, log messages, and UI labels.

        Returns:
            A descriptive string, e.g. `"Greedy (Hard)"`,
            `"MCTS (300 iters)"`, or `"Minimax (depth 6, SimpleHeuristic)"`.
        """

    # ------------------------------------------------------------------
    # Optional hooks (safe defaults provided)
    # ------------------------------------------------------------------

    def on_game_start(self) -> None:
        """Called once before a new game begins.

        Override to reset any per-game state (e.g. clearing a transposition
        table that should not carry over between games). The default
        implementation does nothing.
        """

    def on_game_end(self) -> None:
        """Called once after a game concludes.

        Override to perform cleanup or collect statistics. The default
        implementation does nothing.
        """

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.agent_name()!r})"
