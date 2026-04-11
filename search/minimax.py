"""Minimax AI agent for Othello with Alpha-Beta Pruning, Move Ordering,
Transposition Tables, and Iterative Deepening.

Algorithm Overview
------------------
1. **Iterative Deepening**: The search runs at depth 1, 2, 3, … up to
   `max_depth`. Each completed shallow search primes the move-ordering
   hint for the next deeper search - a technique called *aspiration search*.

2. **Alpha-Beta Pruning**: Cuts branches that cannot affect the final
   decision, reducing the effective branching factor from ~20 to ~√20.

3. **Move Ordering**: Legal moves are sorted by a static positional weight
   matrix before searching. Better moves are tried first, which maximises
   the number of alpha-beta cutoffs.

4. **Transposition Table**: A hash map keyed on `(board_string, player)`
   caches previously computed scores. Entries store the depth, score, and a
   flag indicating whether the score is EXACT, a LOWERBOUND, or an UPPERBOUND
   (standard "enhanced transposition table" technique).

References
----------
- Alpha-Beta tutorial:
  https://webdocs.cs.ualberta.ca/~mmueller/courses/2014-AAAI-games-tutorial/slides/AAAI-14-Tutorial-Games-3-AlphaBeta.pdf
- Move-ordering weights:
  https://www.researchgate.net/figure/The-move-ordering-for-Othello_fig1_326235384
"""
from __future__ import annotations

import time
from typing import Callable, Optional, Tuple

from heuristic import OthelloHeuristic, SmartHeuristic
from othello.constants import DEFAULT_MINIMAX_DEPTH, DEFAULT_MINIMAX_TIMELIMIT
from othello.engine import Coord, OthelloGame
from search.model import OthelloAgent

# ---------------------------------------------------------------------------
# Transposition-Table Flag Constants
# ---------------------------------------------------------------------------
# EXACT:      The subtree was fully searched; the stored score is precise.
# LOWERBOUND: An alpha cutoff occurred; the true score is >= the stored value.
# UPPERBOUND: A beta cutoff occurred; the true score is <= the stored value.
EXACT      = 0
LOWERBOUND = 1
UPPERBOUND = 2

# ---------------------------------------------------------------------------
# Transposition Table
# ---------------------------------------------------------------------------
# Maps board_key → {'score', 'flag', 'depth', 'best_move'}.
# Module-level dict lets the table persist across multiple calls within a
# single session, benefiting repeated searches from similar positions.
transposition_table: dict = {}

# ---------------------------------------------------------------------------
# Positional Weight Matrix
# ---------------------------------------------------------------------------
# Static priority grid for an 8×8 Othello board.
# Higher values = preferred squares. Corners (5) are top priority;
# X-squares (1) next to corners are actively avoided.
_move_priority: list[list[int]] = [
    [5, 2, 4, 3, 3, 4, 2, 5],
    [2, 1, 3, 3, 3, 3, 1, 2],
    [4, 3, 4, 4, 4, 4, 3, 4],
    [3, 3, 4, 4, 4, 4, 3, 3],
    [3, 3, 4, 4, 4, 4, 3, 3],
    [4, 3, 4, 4, 4, 4, 3, 4],
    [2, 1, 3, 3, 3, 3, 1, 2],
    [5, 2, 4, 3, 3, 4, 2, 5],
]

# Flatten to a dict for O(1) priority look-ups during move sorting.
FLAT_PRIORITY: dict[Coord, int] = {
    (r, c): _move_priority[r][c] for r in range(8) for c in range(8)
}


# ---------------------------------------------------------------------------
# Move Ordering
# ---------------------------------------------------------------------------

def order_move(valid_moves: list[Coord]) -> list[Coord]:
    """Sort moves in descending positional-weight order.

    Placing stronger squares first maximises alpha-beta cutoffs by ensuring
    the search explores the most promising branches before weaker ones.

    Args:
        valid_moves: Unordered list of `(row, col)` legal move coordinates.

    Returns:
        The same coordinates sorted by `FLAT_PRIORITY` (highest first).
    """
    return sorted(valid_moves, key=FLAT_PRIORITY.get, reverse=True)


# ---------------------------------------------------------------------------
# Transposition Table Key
# ---------------------------------------------------------------------------

def _board_to_key(game: OthelloGame) -> tuple[str, str]:
    """Create a hashable transposition-table key from the current game state.

    Both the board layout and the active player are included so that identical
    boards with different players-to-move are stored as separate entries.

    Args:
        game: The current game state.

    Returns:
        A `(board_string, current_player)` tuple.
    """
    return (str(game.board), game.current_player)


# ---------------------------------------------------------------------------
# Core Recursive Search
# ---------------------------------------------------------------------------

def _minimax(
    game: OthelloGame,
    depth: int,
    alpha: float,
    beta: float,
    is_max: bool,
    heuristic_func: Callable[[OthelloGame], float],
) -> float:
    """Recursive alpha-beta minimax with move ordering and a transposition table.

    Moves are applied and undone in-place on `game` to avoid the cost of
    cloning the board at every node.

    Args:
        game:           Mutable game state. Moves are applied and undone
                        in-place to avoid expensive cloning.
        depth:          Remaining search depth. Recursion stops at 0.
        alpha:          Best score the maximiser is guaranteed so far.
        beta:           Best score the minimiser is guaranteed so far.
        is_max:         `True` if this is a maximising node.
        heuristic_func: Leaf-node evaluation function. Receives the game
                        state and returns a float score from the current
                        player's perspective.

    Returns:
        The minimax score for this node.
    """
    # --- Base Case ---
    # Stop recursing when the depth budget runs out or the game is finished.
    if depth == 0 or game.is_game_over():
        res = heuristic_func(game)
        return res if is_max else res * -1  # Negate score for minimiser nodes to maintain perspective.

    key = _board_to_key(game)
    tt_best_move: Optional[Coord] = None

    # --- Transposition Table Lookup ---
    # If this exact board + player combo has been solved at >= this depth,
    # reuse (or tighten alpha/beta using) the cached result.
    if key in transposition_table:
        tt_entry     = transposition_table[key]
        tt_best_move = tt_entry.get("best_move")

        if tt_entry["depth"] >= depth:
            tt_score = tt_entry["score"]
            tt_flag  = tt_entry["flag"]

            if tt_flag == EXACT:
                # Perfect score; no further search needed for this node.
                return tt_score
            elif tt_flag == LOWERBOUND:
                # Raise alpha: the true score is at least this good.
                alpha = max(alpha, tt_score)
            elif tt_flag == UPPERBOUND:
                # Lower beta: the true score is at most this good.
                beta = min(beta, tt_score)

            if alpha >= beta:
                # Window already closed - the cached bound is sufficient.
                return tt_score

    valid_moves = list(game.legal_moves().keys())

    # --- Move Ordering ---
    # Front-load the transposition table's best move so it gets evaluated
    # first - it is the strongest move from a previously completed search
    # at this node and will trigger the most beta-cutoffs.
    if tt_best_move and tt_best_move in valid_moves:
        ordered_moves = [tt_best_move] + order_move(
            [m for m in valid_moves if m != tt_best_move]
        )
    else:
        ordered_moves = order_move(valid_moves)

    best_move_this_node: Optional[Coord] = None

    if is_max:
        # ------------------------------------------------------------------
        # Maximising Node
        # ------------------------------------------------------------------
        max_eval       = float("-inf")
        original_alpha = alpha  # Saved to determine the correct TT flag later.

        for move_coord in ordered_moves:
            current_player = game.current_player
            game.apply_move(move_coord[0], move_coord[1])

            # Detect the Othello "pass" rule: if the active player did not
            # change after the move, the opponent had no legal moves and the
            # same player must move again - the child is also a max node.
            next_is_max = not is_max if game.current_player != current_player else is_max

            eval_score = _minimax(game, depth - 1, alpha, beta, next_is_max, heuristic_func)

            if eval_score > max_eval:
                max_eval            = eval_score
                best_move_this_node = move_coord

            alpha = max(alpha, eval_score)
            game.undo()  # Restore the board to its state before this move.

            # Beta cutoff: the minimiser would never allow this subtree.
            if beta <= alpha:
                break

        # Determine the appropriate TT flag for this node.
        if max_eval <= original_alpha:
            flag = UPPERBOUND  # All children were worse than what we already had.
        elif max_eval >= beta:
            flag = LOWERBOUND  # Cut off early; true value might be even higher.
        else:
            flag = EXACT

        # Only overwrite the TT entry if this search went deeper than before.
        tt_entry = transposition_table.get(key)
        if tt_entry is None or depth >= tt_entry["depth"]:
            transposition_table[key] = {
                "score":     max_eval,
                "flag":      flag,
                "depth":     depth,
                "best_move": best_move_this_node,
            }

        return max_eval

    else:
        # ------------------------------------------------------------------
        # Minimising Node
        # ------------------------------------------------------------------
        min_eval      = float("inf")
        original_beta = beta  # Saved to determine the correct TT flag later.

        for move_coord in ordered_moves:
            current_player = game.current_player
            game.apply_move(move_coord[0], move_coord[1])

            next_is_max = not is_max if game.current_player != current_player else is_max

            eval_score = _minimax(game, depth - 1, alpha, beta, next_is_max, heuristic_func)

            if eval_score < min_eval:
                min_eval            = eval_score
                best_move_this_node = move_coord

            beta = min(beta, eval_score)
            game.undo()

            # Alpha cutoff: the maximiser already has a better option elsewhere.
            if beta <= alpha:
                break

        # Determine the appropriate TT flag for this node.
        if min_eval <= alpha:
            flag = UPPERBOUND
        elif min_eval >= original_beta:
            flag = LOWERBOUND
        else:
            flag = EXACT

        tt_entry = transposition_table.get(key)
        if tt_entry is None or depth >= tt_entry["depth"]:
            transposition_table[key] = {
                "score":     min_eval,
                "flag":      flag,
                "depth":     depth,
                "best_move": best_move_this_node,
            }

        return min_eval


# ---------------------------------------------------------------------------
# Public Search Entry Point
# ---------------------------------------------------------------------------

def get_best_move_minimax(
    game: OthelloGame,
    max_depth: int,
    heuristic_func: Callable[[OthelloGame], float],
    time_limit: float = DEFAULT_MINIMAX_TIMELIMIT,
) -> Tuple[Optional[Coord], float]:
    """Find the best move using Iterative-Deepening Alpha-Beta Minimax.

    Searches depth 1 → 2 → … → `max_depth` one level at a time. Each
    completed iteration updates the "best move so far", which seeds move
    ordering in the next iteration - a simple form of aspiration search.
    If the clock expires mid-iteration, the best result from the last
    *completed* iteration is returned immediately to avoid an inconsistent
    partial result.

    Args:
        game:           The current game state. Not mutated.
        max_depth:      Maximum search depth in plies.
        heuristic_func: Leaf evaluation function (see `_minimax`).
        time_limit:     Wall-clock budget in seconds.

    Returns:
        A `(move, score)` tuple where `move` is the recommended
        `(row, col)` coordinate and `score` is its heuristic value.
        Returns `(None, heuristic)` if no legal moves exist.
    """
    valid_moves = list(game.legal_moves().keys())

    if not valid_moves:
        # No moves available; return the static evaluation of this position.
        return None, heuristic_func(game)

    best_move_overall:  Optional[Coord] = None
    best_score_overall: float           = float("-inf")

    start_time = time.time()

    # --- Iterative Deepening Loop ---
    for current_depth in range(1, max_depth + 1):
        alpha = float("-inf")
        beta  = float("inf")

        best_move_this_depth:  Optional[Coord] = None
        best_score_this_depth: float           = float("-inf")

        # Seed move ordering with the best move from the previous iteration.
        # A better first move = more beta-cutoffs = faster search.
        if best_move_overall and best_move_overall in valid_moves:
            ordered_moves = [best_move_overall] + order_move(
                [m for m in valid_moves if m != best_move_overall]
            )
        else:
            ordered_moves = order_move(valid_moves)

        for move_coord in ordered_moves:
            # --- Time Check ---
            # If the budget is exhausted, abort immediately and return the
            # best result from the last *completed* depth rather than a
            # potentially inconsistent partial result.
            if time.time() - start_time > time_limit:
                fallback = best_move_overall if best_move_overall else move_coord
                return fallback, best_score_overall

            current_player = game.current_player
            game.apply_move(move_coord[0], move_coord[1])

            # If the same player must move again (pass rule), the child node
            # is also a maximising node from our perspective.
            next_is_max = False if game.current_player != current_player else True

            score = _minimax(
                game,
                current_depth - 1,
                alpha,
                beta,
                next_is_max,
                heuristic_func,
            )

            if score > best_score_this_depth:
                best_score_this_depth = score
                best_move_this_depth  = move_coord

            alpha = max(alpha, best_score_this_depth)
            game.undo()

        # Commit this depth's best move before starting the next, deeper iteration.
        best_move_overall  = best_move_this_depth
        best_score_overall = best_score_this_depth

    return best_move_overall, best_score_overall


# ---------------------------------------------------------------------------
# OthelloAgent Class
# ---------------------------------------------------------------------------

class MinimaxAgent(OthelloAgent):
    """Iterative-Deepening Alpha-Beta Minimax agent.

    Exposes `get_best_move_minimax` through the uniform `OthelloAgent`
    interface. All search parameters are set once at construction time.

    Attributes:
        _max_depth:  Maximum search depth in plies.
        _heuristic:  Board evaluation heuristic.
        _time_limit: Per-move wall-clock budget in seconds.

    Example::

        agent = MinimaxAgent(max_depth=6, heuristic=SmartHeuristic())
        move  = agent.choose_move(game)
    """

    def __init__(
        self,
        max_depth:  int                                      = DEFAULT_MINIMAX_DEPTH,
        heuristic:  Optional[Callable[[OthelloGame], float]] = None,
        time_limit: float                                    = DEFAULT_MINIMAX_TIMELIMIT,
    ) -> None:
        """Initialise the Minimax agent.

        Args:
            max_depth:  Maximum iterative-deepening depth.
            heuristic:  A callable `(OthelloGame) -> float` used to score
                        leaf nodes. Accepts any :class:`~heuristic.OthelloHeuristic`
                        instance or plain function. Defaults to
                        :class:`~heuristic.SmartHeuristic`.
            time_limit: Maximum seconds allowed per move.
        """
        self._max_depth:  int                            = max_depth
        self._heuristic:  Callable[[OthelloGame], float] = (
            heuristic if heuristic is not None else SmartHeuristic()
        )
        self._time_limit: float = time_limit

    # ------------------------------------------------------------------
    # OthelloAgent interface
    # ------------------------------------------------------------------

    def choose_move(self, game: OthelloGame) -> Optional[Coord]:
        """Return the minimax-optimal move for the current player.

        Args:
            game: The live game state.

        Returns:
            A `(row, col)` coordinate, or `None` if no legal moves exist.
        """
        move, _score = get_best_move_minimax(
            game,
            max_depth      = self._max_depth,
            heuristic_func = self._heuristic,
            time_limit     = self._time_limit,
        )
        return move

    def agent_name(self) -> str:
        """Return a label that encodes the configured depth and heuristic.
        """
        if self._heuristic is not None:
            return f"Minimax (depth {self._max_depth}, {self._heuristic.__class__.__name__})"
        return f"Minimax (depth {self._max_depth})"

    # ------------------------------------------------------------------
    # Optional lifecycle hooks
    # ------------------------------------------------------------------

    def on_game_start(self) -> None:
        """Clear the transposition table before a new game begins.

        Prevents stale entries from a previous game affecting the current
        search. The table is module-level and shared across all calls.
        """
        transposition_table.clear()
