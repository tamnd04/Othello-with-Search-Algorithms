from __future__ import annotations

import random
from typing import Optional, Tuple

from .constants import BLACK, WHITE
from .engine import OthelloGame

Coord = Tuple[int, int]


class GreedyAI:
    """A simple baseline AI that prefers moves flipping the most discs,
    then corners, then edges. This is intentionally lightweight so the
    search-based agents can replace it later.
    """

    CORNERS = {(0, 0), (0, 7), (7, 0), (7, 7)}
    X_SQUARES = {(1, 1), (1, 6), (6, 1), (6, 6)}

    def choose_move(self, game: OthelloGame, difficulty: str = "medium") -> Optional[Coord]:
        legal = game.legal_moves(game.current_player)
        if not legal:
            return None

        level = difficulty.strip().lower()
        if level == "easy":
            return random.choice(list(legal.keys()))
        if level == "hard":
            return self._hard_move(game)
        return self._greedy_move(game)

    def _greedy_move(self, game: OthelloGame) -> Coord:
        legal = game.legal_moves(game.current_player)

        def score(item: tuple[Coord, list[Coord]]) -> tuple[int, int, int]:
            move, flipped = item
            row, col = move
            is_corner = 1 if move in self.CORNERS else 0
            is_edge = 1 if row in {0, 7} or col in {0, 7} else 0
            return (is_corner, is_edge, len(flipped))

        return max(legal.items(), key=score)[0]

    def _hard_move(self, game: OthelloGame) -> Coord:
        legal = game.legal_moves(game.current_player)
        player = game.current_player
        opponent = WHITE if player == BLACK else BLACK

        def evaluate_position(simulated: OthelloGame, move: Coord) -> tuple[int, int, int, int, int]:
            scores = simulated.count_discs()
            my_count = scores[player]
            opp_count = scores[opponent]

            my_moves = len(simulated.legal_moves(player))
            opp_moves = len(simulated.legal_moves(opponent))

            my_corners = sum(1 for corner in self.CORNERS if simulated.get_cell(*corner) == player)
            opp_corners = sum(1 for corner in self.CORNERS if simulated.get_cell(*corner) == opponent)

            # Avoid taking risky X-squares unless backed by better positional value.
            x_square_penalty = 1 if move in self.X_SQUARES else 0

            corner_delta = my_corners - opp_corners
            mobility_delta = my_moves - opp_moves
            disc_delta = my_count - opp_count

            return (corner_delta, mobility_delta, disc_delta, -x_square_penalty, len(legal[move]))

        best_move: Optional[Coord] = None
        best_score: Optional[tuple[int, int, int, int, int]] = None

        for move in legal:
            simulated = game.clone()
            simulated.apply_move(move[0], move[1])
            score = evaluate_position(simulated, move)

            if best_score is None or score > best_score:
                best_score = score
                best_move = move

        # legal is non-empty, so best_move will always be set.
        return best_move if best_move is not None else next(iter(legal.keys()))
