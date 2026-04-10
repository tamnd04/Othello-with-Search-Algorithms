BOARD_SIZE = 8
EMPTY = "."
BLACK = "B"
WHITE = "W"

DIRECTIONS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)

BOARD_CORNERS = frozenset([
    (0, 0), 
    (0, BOARD_SIZE - 1), 
    (BOARD_SIZE - 1, 0), 
    (BOARD_SIZE - 1, BOARD_SIZE - 1)
    ])

PLAYER_NAMES = {
    BLACK: "Black",
    WHITE: "White",
}

PLAYER_COLORS = {
    BLACK: "#111827",
    WHITE: "#F9FAFB",
}

DEFAULT_MCTS_ITERATIONS = 100
DEFAULT_MCTS_DEPTH = 20
DEFAULT_MINIMAX_DEPTH = 6
DEFAULT_MINIMAX_TIMELIMIT = 5.0