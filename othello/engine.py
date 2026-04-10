"""Core Othello game engine.

Provides the board representation, legal-move generation, move application
with automatic pass/end-of-game detection, undo, and disc counting.

The engine is intentionally stateful and in-place: :meth:`OthelloGame.apply_move`
mutates the board directly and saves a snapshot to `history` so that
:meth:`OthelloGame.undo` can restore the previous state without cloning.
Search algorithms that need non-destructive exploration should call
:meth:`OthelloGame.clone` to get an independent copy first.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .constants import BLACK, BOARD_SIZE, DIRECTIONS, EMPTY, PLAYER_NAMES, WHITE

Coord = Tuple[int, int]
Board = List[List[str]]


@dataclass(frozen=True)
class MoveOutcome:
    """Immutable record of the result of a single move.

    Attributes:
        move:          The `(row, col)` square where the disc was placed.
        player:        The player who made the move.
        flipped:       Squares whose discs were flipped by this move.
        passed_player: The player who was forced to pass (no legal moves),
                       or `None` if no pass occurred.
        next_player:   The player whose turn it is next, or `None` if the
                       game is over.
        game_over:     `True` if neither player has any remaining legal moves.
    """

    move: Coord
    player: str
    flipped: Tuple[Coord, ...]
    passed_player: Optional[str]
    next_player: Optional[str]
    game_over: bool


class OthelloGame:
    """Mutable Othello game state.

    Attributes:
        board:          8×8 grid of `BLACK`, `WHITE`, or `EMPTY` strings.
        current_player: The player who moves next.
        history:        Stack of `(board_snapshot, player)` tuples used by
                        :meth:`undo` to restore previous states.
    """

    def __init__(self) -> None:
        """Initialise and immediately reset to the standard opening position."""
        self.board: Board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.current_player: str = BLACK
        self.history: List[Tuple[Board, str]] = []
        self.reset()

    def reset(self) -> None:
        """Reset the board to the standard 4-disc opening position.

        Clears `history` and sets `current_player` to Black.
        """
        self.board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        mid = BOARD_SIZE // 2
        self.board[mid - 1][mid - 1] = WHITE
        self.board[mid][mid] = WHITE
        self.board[mid - 1][mid] = BLACK
        self.board[mid][mid - 1] = BLACK
        self.current_player = BLACK
        self.history = []

    def clone(self) -> "OthelloGame":
        """Return a deep copy of this game state.

        The clone shares no references with the original, so mutations on
        either object do not affect the other. History is also cloned so that
        :meth:`undo` works correctly on the copy.

        Returns:
            An independent :class:`OthelloGame` in the same state.
        """
        cloned = OthelloGame()
        cloned.board = self._copy_board(self.board)
        cloned.current_player = self.current_player
        cloned.history = [(self._copy_board(board), player) for board, player in self.history]
        return cloned

    @staticmethod
    def _copy_board(board: Board) -> Board:
        """Return a shallow-row copy of `board` (each row is a new list).

        Args:
            board: The 8×8 board to copy.

        Returns:
            A new `Board` with independent rows.
        """
        return [row[:] for row in board]

    @staticmethod
    def opponent(player: str) -> str:
        """Return the opposing player constant.

        Args:
            player: `BLACK` or `WHITE`.

        Returns:
            The other player constant.
        """
        return WHITE if player == BLACK else BLACK

    @staticmethod
    def in_bounds(row: int, col: int) -> bool:
        """Return `True` if `(row, col)` is a valid board coordinate.

        Args:
            row: Row index.
            col: Column index.

        Returns:
            `True` when both indices are in `[0, BOARD_SIZE)`.
        """
        return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE

    def get_cell(self, row: int, col: int) -> str:
        """Return the occupant of a single board cell.

        Args:
            row: Row index (0-based).
            col: Column index (0-based).

        Returns:
            `BLACK`, `WHITE`, or `EMPTY`.
        """
        return self.board[row][col]

    def legal_moves(self, player: Optional[str] = None) -> Dict[Coord, List[Coord]]:
        """Return all legal moves for `player` and the discs each would flip.

        Args:
            player: The player to query. Defaults to `current_player`.

        Returns:
            A dict mapping each legal `(row, col)` to the list of
            `(row, col)` squares that would be flipped by that move.
            Empty dict if the player has no legal moves.
        """
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
        """Return the discs that would be flipped by placing `player` at `(row, col)`.

        Walks each of the eight directions from the target square, collecting
        opponent discs until a friendly disc is found. An empty list is
        returned when the square is occupied or no discs would be flipped.

        Args:
            row:    Target row index.
            col:    Target column index.
            player: The player placing the disc.

        Returns:
            List of `(row, col)` coordinates that would be flipped.
        """
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
        """Return `True` if `player` has at least one legal move.

        Args:
            player: `BLACK` or `WHITE`.

        Returns:
            `True` when :meth:`legal_moves` returns a non-empty dict.
        """
        return bool(self.legal_moves(player))

    def is_game_over(self) -> bool:
        """Return `True` when neither player has any legal move remaining.

        Returns:
            `True` if the game has ended.
        """
        return not self.can_player_move(BLACK) and not self.can_player_move(WHITE)

    def count_discs(self) -> Dict[str, int]:
        """Count the discs of each colour and the number of empty squares.

        Returns:
            A dict with keys `BLACK`, `WHITE`, and `EMPTY` mapping to
            their respective counts on the current board.
        """
        black_count = sum(row.count(BLACK) for row in self.board)
        white_count = sum(row.count(WHITE) for row in self.board)
        empty_count = sum(row.count(EMPTY) for row in self.board)
        return {BLACK: black_count, WHITE: white_count, EMPTY: empty_count}

    def winner(self) -> Optional[str]:
        """Return the player with the most discs, or `None` on a draw.

        Should only be called after :meth:`is_game_over` returns `True`;
        calling it mid-game returns the current leader, not the final winner.

        Returns:
            `BLACK`, `WHITE`, or `None` for a draw.
        """
        scores = self.count_discs()
        if scores[BLACK] > scores[WHITE]:
            return BLACK
        if scores[WHITE] > scores[BLACK]:
            return WHITE
        return None

    def save_snapshot(self) -> None:
        """Push the current board and active player onto the undo stack.

        Called automatically by :meth:`apply_move` before mutating state.
        """
        self.history.append((self._copy_board(self.board), self.current_player))

    def undo(self) -> bool:
        """Restore the board and active player to the state before the last move.

        Returns:
            `True` if a move was undone, `False` if the history was empty.
        """
        if not self.history:
            return False
        board, player = self.history.pop()
        self.board = self._copy_board(board)
        self.current_player = player
        return True

    def apply_move(self, row: int, col: int) -> MoveOutcome:
        """Place a disc for the current player and update game state.

        Saves a snapshot, places the disc, flips captured discs, then
        determines the next player. If the opponent cannot move the current
        player continues (pass). If neither can move the game ends.

        Args:
            row: Target row index (0-based).
            col: Target column index (0-based).

        Returns:
            A :class:`MoveOutcome` describing what happened.

        Raises:
            ValueError: If `(row, col)` is not a legal move for the current
                        player.
        """
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
        """Convert `(row, col)` indices to standard Othello notation (e.g. `"D3"`).

        Args:
            row: Row index (0-based).
            col: Column index (0-based).

        Returns:
            A string like `"A1"` through `"H8"`.
        """
        return f"{chr(ord('A') + col)}{row + 1}"

    def status_text(self) -> str:
        """Return a human-readable description of the current game state.

        Returns:
            `"Game over - <Player> wins"`, `"Game over - Draw"`, or
            `"<Player>'s turn"` depending on the current state.
        """
        if self.is_game_over():
            winner = self.winner()
            if winner is None:
                return "Game over - Draw"
            return f"Game over - {PLAYER_NAMES[winner]} wins"
        return f"{PLAYER_NAMES[self.current_player]}'s turn"
