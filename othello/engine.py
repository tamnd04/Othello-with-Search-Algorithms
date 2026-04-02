from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .constants import BLACK, BOARD_SIZE, DIRECTIONS, EMPTY, PLAYER_NAMES, WHITE

Coord = Tuple[int, int]
Board = List[List[str]]


@dataclass(frozen=True)
class MoveOutcome:
    move: Coord
    player: str
    flipped: Tuple[Coord, ...]
    passed_player: Optional[str]
    next_player: Optional[str]
    game_over: bool


class OthelloGame:
    def __init__(self) -> None:
        self.board: Board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.current_player: str = BLACK
        self.history: List[Tuple[Board, str]] = []
        self.reset()

    def reset(self) -> None:
        self.board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        mid = BOARD_SIZE // 2
        self.board[mid - 1][mid - 1] = WHITE
        self.board[mid][mid] = WHITE
        self.board[mid - 1][mid] = BLACK
        self.board[mid][mid - 1] = BLACK
        self.current_player = BLACK
        self.history = []

    def clone(self) -> "OthelloGame":
        cloned = OthelloGame()
        cloned.board = self._copy_board(self.board)
        cloned.current_player = self.current_player
        cloned.history = [(self._copy_board(board), player) for board, player in self.history]
        return cloned

    @staticmethod
    def _copy_board(board: Board) -> Board:
        return [row[:] for row in board]

    @staticmethod
    def opponent(player: str) -> str:
        return WHITE if player == BLACK else BLACK

    @staticmethod
    def in_bounds(row: int, col: int) -> bool:
        return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE

    def get_cell(self, row: int, col: int) -> str:
        return self.board[row][col]

    def legal_moves(self, player: Optional[str] = None) -> Dict[Coord, List[Coord]]:
        player = player or self.current_player
        moves: Dict[Coord, List[Coord]] = {}

        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if self.board[row][col] != EMPTY:
                    continue
                flipped = self._flips_for_move(row, col, player)
                if flipped:
                    moves[(row, col)] = flipped
        return moves

    def _flips_for_move(self, row: int, col: int, player: str) -> List[Coord]:
        if self.board[row][col] != EMPTY:
            return []

        opponent = self.opponent(player)
        all_flips: List[Coord] = []

        for d_row, d_col in DIRECTIONS:
            path: List[Coord] = []
            r, c = row + d_row, col + d_col
            while self.in_bounds(r, c) and self.board[r][c] == opponent:
                path.append((r, c))
                r += d_row
                c += d_col

            if path and self.in_bounds(r, c) and self.board[r][c] == player:
                all_flips.extend(path)

        return all_flips

    def can_player_move(self, player: str) -> bool:
        return bool(self.legal_moves(player))

    def is_game_over(self) -> bool:
        return not self.can_player_move(BLACK) and not self.can_player_move(WHITE)

    def count_discs(self) -> Dict[str, int]:
        black_count = sum(row.count(BLACK) for row in self.board)
        white_count = sum(row.count(WHITE) for row in self.board)
        empty_count = sum(row.count(EMPTY) for row in self.board)
        return {BLACK: black_count, WHITE: white_count, EMPTY: empty_count}

    def winner(self) -> Optional[str]:
        scores = self.count_discs()
        if scores[BLACK] > scores[WHITE]:
            return BLACK
        if scores[WHITE] > scores[BLACK]:
            return WHITE
        return None

    def save_snapshot(self) -> None:
        self.history.append((self._copy_board(self.board), self.current_player))

    def undo(self) -> bool:
        if not self.history:
            return False
        board, player = self.history.pop()
        self.board = self._copy_board(board)
        self.current_player = player
        return True

    def apply_move(self, row: int, col: int) -> MoveOutcome:
        legal = self.legal_moves(self.current_player)
        if (row, col) not in legal:
            raise ValueError(f"Illegal move at {(row, col)} for {self.current_player}")

        self.save_snapshot()
        player = self.current_player
        flipped = legal[(row, col)]
        self.board[row][col] = player
        for flip_row, flip_col in flipped:
            self.board[flip_row][flip_col] = player

        opponent = self.opponent(player)
        passed_player: Optional[str] = None
        next_player: Optional[str]

        if self.can_player_move(opponent):
            self.current_player = opponent
            next_player = opponent
        elif self.can_player_move(player):
            self.current_player = player
            passed_player = opponent
            next_player = player
        else:
            next_player = None

        return MoveOutcome(
            move=(row, col),
            player=player,
            flipped=tuple(flipped),
            passed_player=passed_player,
            next_player=next_player,
            game_over=self.is_game_over(),
        )

    def notation_for_move(self, row: int, col: int) -> str:
        return f"{chr(ord('A') + col)}{row + 1}"

    def status_text(self) -> str:
        if self.is_game_over():
            winner = self.winner()
            if winner is None:
                return "Game over — Draw"
            return f"Game over — {PLAYER_NAMES[winner]} wins"
        return f"{PLAYER_NAMES[self.current_player]}'s turn"
