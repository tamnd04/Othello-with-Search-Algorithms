"""Monte Carlo Tree Search (MCTS) agent for Othello.

Algorithm Overview
------------------
Each call to `get_best_move_mcts` runs a fixed number of MCTS *iterations*.
Every iteration proceeds through four standard phases:

1. **Selection**   - Walk the existing tree using the UCB1 formula, balancing
   exploitation (nodes with high win rates) against exploration (nodes visited
   few times).

2. **Expansion**   - When a node with untried moves is reached, create one new
   child by applying an untried move to the scratchpad board.

3. **Simulation**  - Play random moves from the new node until the game ends
   or a depth cap (`dept_roll`) is hit, yielding a winner.

4. **Backpropagation** - Propagate the simulation result back up the tree,
   updating `wins` and `visits` for every ancestor node.

After all iterations the child of the root with the most *visits* is returned
- visit count is more robust than raw win-rate as the final decision criterion
because highly-visited nodes have been explored most thoroughly.

Performance Notes
-----------------
- The real game object is *cloned* once per iteration to avoid corrupting the
  live board. All simulation moves are applied to the clone; tree nodes store
  only lightweight metadata (no full board copy).
- Move ordering from `minimax.order_move` is applied at the root to
  bias initial exploration toward positionally strong moves.
"""
from __future__ import annotations

import math
import random
from typing import Optional

from othello.constants import DEFAULT_MCTS_DEPTH, DEFAULT_MCTS_ITERATIONS
from othello.engine import Coord, OthelloGame
from .minimax import order_move


class MCTSNode:
    """A single node in the Monte Carlo search tree.

    Each node corresponds to a game state reached by applying `move` from
    the parent's state. To save memory, the full board is *not* stored here;
    instead, the state is re-derived during each iteration by replaying moves
    on a cloned board.

    Attributes:
        parent:            The parent node, or `None` for the root.
        move:              The `(row, col)` move that created this node,
                           or `None` for the root.
        children:          Child nodes discovered so far.
        wins:              Cumulative reward (1.0 per win, 0.5 per draw, 0.0
                           per loss) from the perspective of
                           `player_just_moved`.
        visits:            Total number of times this node has been visited.
        untried_moves:     Legal moves from this state that have not yet been
                           expanded into child nodes.
        player_just_moved: The colour (`BLACK` / `WHITE`) that *just moved*
                           to reach this node's state. Used during
                           backpropagation to credit wins correctly.
    """

    def __init__(
        self,
        parent:            Optional[MCTSNode] = None,
        move:              Optional[Coord]    = None,
        untried_moves:     Optional[list[Coord]] = None,
        player_just_moved: Optional[str]     = None,
    ) -> None:
        self.parent            = parent
        self.move              = move
        self.children:         list[MCTSNode] = []
        self.wins:             float          = 0.0
        self.visits:           int            = 0
        self.untried_moves:    list[Coord]    = untried_moves or []
        self.player_just_moved = player_just_moved

    # ------------------------------------------------------------------
    # Phase 1: Selection
    # ------------------------------------------------------------------

    def uct_select_child(self, exploration_weight: float = math.sqrt(2)) -> MCTSNode:
        """Select the child node with the highest UCB1 score.

        UCB1 formula:

            UCB1(c) = wins/visits  +  C * sqrt(ln(N + 1) / visits_c)

        where `N` is this node's visit count, `C` is the exploration
        constant (sqrt(2) by default), and `visits_c` is the child's visit count.
        A child with zero visits is given infinite priority so every child
        is explored at least once before any is revisited.

        Args:
            exploration_weight: The exploration constant `C`. Higher values
                                favour exploring less-visited children.

        Returns:
            The child node with the highest UCB1 value.
        """
        return max(
            self.children,
            key=lambda c: (
                float("inf")
                if c.visits == 0
                else (c.wins / c.visits)
                + exploration_weight * math.sqrt(math.log(self.visits + 1) / c.visits)
            ),
        )

    # ------------------------------------------------------------------
    # Phase 2: Expansion
    # ------------------------------------------------------------------

    def expand(self, state: OthelloGame) -> MCTSNode:
        """Pop one untried move, apply it to `state`, and create a child node.

        The move is applied directly to the shared scratchpad board (`state`)
        so the caller can continue simulation from the resulting position
        without any additional setup.

        Args:
            state: The mutable game state for this iteration. After this call
                   `state` reflects the expanded move.

        Returns:
            The newly created child node.
        """
        # Take the next untried move from the front of the queue.
        move = self.untried_moves.pop(0)
        state.apply_move(move[0], move[1])

        # Capture the metadata needed for the new node before the state changes.
        new_untried          = list(state.legal_moves().keys())
        new_player_just_moved = state.opponent(state.current_player)

        # Store only lightweight metadata - not the entire board.
        child_node = MCTSNode(
            parent            = self,
            move              = move,
            untried_moves     = new_untried,
            player_just_moved = new_player_just_moved,
        )

        self.children.append(child_node)
        return child_node

    # ------------------------------------------------------------------
    # Phase 4: Backpropagation
    # ------------------------------------------------------------------

    def update(self, winner: Optional[str]) -> None:
        """Update this node's statistics with the simulation result.

        Credits a full win (1.0) if `winner` matches `player_just_moved`,
        a draw (0.5) if `winner` is `None`, or nothing (0.0) for a loss.

        Args:
            winner: The colour that won the simulation, or `None` for a draw.
        """
        self.visits += 1
        if winner == self.player_just_moved:
            self.wins += 1.0
        elif winner is None:
            # Draw: award half a point to avoid unfairly penalising drawn lines.
            self.wins += 0.5


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_best_move_mcts(
    game: OthelloGame,
    iterations: int = DEFAULT_MCTS_ITERATIONS,
    dept_roll: int = DEFAULT_MCTS_DEPTH,
) -> Optional[Coord]:
    """Run MCTS and return the recommended move for the current player.

    Args:
        game:       The live game state. This object is *never* mutated -
                    a fresh clone is used as the scratchpad for each iteration.
        iterations: Number of MCTS iterations to run. More iterations yield
                    stronger play at the cost of additional computation time.
        dept_roll:  Maximum number of random moves per simulation rollout.
                    Capping the rollout depth bounds the per-iteration cost
                    while still providing a useful signal about game outcomes.

    Returns:
        The `(row, col)` coordinate of the recommended move, or `None`
        if no legal moves are available.
    """
    if not game.legal_moves():
        return None

    # --- Initialise the Root Node ---
    # Pre-order the root's untried moves so the first expansions prefer
    # positionally strong squares (corners over X-squares, etc.).
    root_untried = order_move(list(game.legal_moves().keys()))
    root_player  = game.opponent(game.current_player)  # Player who "just moved" to reach root.
    root = MCTSNode(
        parent            = None,
        move              = None,
        untried_moves     = root_untried,
        player_just_moved = root_player,
    )

    # --- Main MCTS Loop ---
    for _ in range(iterations):
        node  = root
        state = game.clone()  # Fresh scratchpad board for this iteration.

        # Phase 1: Selection - walk the tree until a node with untried moves.
        while not node.untried_moves and node.children:
            node = node.uct_select_child()
            state.apply_move(node.move[0], node.move[1])

        # Phase 2: Expansion - add one new child node if the state is not terminal.
        if node.untried_moves and not state.is_game_over():
            node = node.expand(state)

        # Phase 3: Simulation (Rollout) - play random moves until terminal or cap.
        depth = 0
        while depth < dept_roll:
            valid_moves = list(state.legal_moves().keys())

            if valid_moves:
                # Random move selection keeps rollouts fast and unbiased.
                move = valid_moves[random.randint(0, len(valid_moves) - 1)]
                state.apply_move(*move)
                depth += 1
                continue

            # No valid moves for the current player - try passing to the opponent.
            opponent = state.opponent(state.current_player)
            if state.can_player_move(opponent):
                state.current_player = opponent
            else:
                # Neither player can move: game is truly over.
                break

        # Determine the winner of the simulated game.
        winner = state.winner()

        # Phase 4: Backpropagation - update every ancestor with the simulation result.
        while node is not None:
            node.update(winner)
            node = node.parent

    # --- Final Decision ---
    # Return the child with the highest *visit count* - more visits means more
    # thorough exploration, making it a robust choice over raw win rate.
    if not root.children:
        return None

    best_child = max(root.children, key=lambda c: c.visits)
    return best_child.move


# ---------------------------------------------------------------------------
# OthelloAgent wrapper
# ---------------------------------------------------------------------------

from search.model import OthelloAgent  # noqa: E402 - imported here to avoid circular deps


class MCTSAgent(OthelloAgent):
    """MCTS agent conforming to the `OthelloAgent` interface.

    Wraps :func:`get_best_move_mcts` and exposes it through the uniform
    `OthelloAgent` interface. All MCTS parameters are set once at
    construction time.

    Attributes:
        _iterations: Number of MCTS iterations per move. More iterations =
                     stronger play at the cost of higher computation time.
        _dept_roll:  Maximum rollout depth per simulation. Bounding this keeps
                     per-iteration cost predictable while still yielding a
                     useful win/loss signal.

    Example::

        agent = MCTSAgent(iterations=300, dept_roll=20)
        move  = agent.choose_move(game)
    """

    def __init__(self, iterations: int = 150, dept_roll: int = 15) -> None:
        """Initialise the MCTS agent.

        Args:
            iterations: Number of tree-search iterations to run per move.
            dept_roll:  Maximum number of random plies per simulation rollout.
        """
        self._iterations: int = iterations
        self._dept_roll:  int = dept_roll

    # ------------------------------------------------------------------
    # OthelloAgent interface
    # ------------------------------------------------------------------

    def choose_move(self, game: OthelloGame) -> Optional[Coord]:
        """Return the MCTS-recommended move for the current player.

        Args:
            game: The live game state. Never mutated - each iteration clones
                  the board internally.

        Returns:
            A `(row, col)` coordinate, or `None` if no legal moves exist.
        """
        return get_best_move_mcts(
            game,
            iterations = self._iterations,
            dept_roll  = self._dept_roll,
        )

    def agent_name(self) -> str:
        """Return a label that encodes the configured iteration count.
        """
        return f"MCTS ({self._iterations} iters)"