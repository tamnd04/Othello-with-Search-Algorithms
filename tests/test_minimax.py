import unittest
from othello.constants import BOARD_SIZE, BLACK, EMPTY, WHITE
from othello.engine import OthelloGame

from search.minimaxorder import get_best_move_minimax 
def smart_heuristic(game: OthelloGame) -> float:
    player = game.current_player
    opponent = game.opponent(player)

    # 1. Corners (Massive Weight)
    CORNERS = {(0, 0), (0, 7), (7, 0), (7, 7)}
    my_corners = sum(1 for c in CORNERS if game.get_cell(*c) == player)
    opp_corners = sum(1 for c in CORNERS if game.get_cell(*c) == opponent)
    
    # 2. X-Squares Penalty (Dangerous squares next to corners)
    X_SQUARES = {(1, 1), (1, 6), (6, 1), (6, 6)}
    my_x = sum(1 for x in X_SQUARES if game.get_cell(*x) == player)
    opp_x = sum(1 for x in X_SQUARES if game.get_cell(*x) == opponent)

    # 3. Raw Discs (Low weight in midgame)
    scores = game.count_discs()
    disc_delta = scores[player] - scores[opponent]

    # Calculate final score
    score = ((my_corners - opp_corners) * 1000) - ((my_x - opp_x) * 300) + (disc_delta * 10)
    return float(score)

def simple_heuristic(game: OthelloGame):
    dct = game.count_discs()
    total_discs = (BOARD_SIZE * BOARD_SIZE) - dct[EMPTY]
    if total_discs == 0: 
        return 0
    return dct[game.current_player] / total_discs

def setup_custom_board(board_string: str, current_turn: str) -> OthelloGame:
    game = OthelloGame()
    game.board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    game.current_player = current_turn
    
    rows = board_string.strip().split('\n')
    for r in range(BOARD_SIZE):
        row_str = rows[r].strip() 
        for c in range(BOARD_SIZE):
            char = row_str[c]
            if char == 'B': game.board[r][c] = BLACK
            elif char == 'W': game.board[r][c] = WHITE
    return game


class TestMinimax(unittest.TestCase):

    def test_1(self):
        """
        Test 1: 
        """
        board_str = """
        ........
        ......W.
        .....W..
        ....B...
        ........
        ........
        ........
        ........
        """
        game = setup_custom_board(board_str, current_turn=BLACK)
        
        best_move, score = get_best_move_minimax(game, max_depth=2, heuristic_func=simple_heuristic)
        
        self.assertEqual(best_move, (0, 7), "FAILED")


    def test_2(self):
        """
        Test 2: 
        """
        board_str = """
        BBBBBBBB
        BBBBBBBB
        BBBBBBBB
        BBBBBBBW
        BBBBBBWW
        BBBBBB.W
        BBBBBB.W
        BBBBBB..
        """
        game = setup_custom_board(board_str, current_turn=BLACK)
        
        best_move, score = get_best_move_minimax(game, max_depth=3, heuristic_func=simple_heuristic)
        
        self.assertEqual(best_move, (7, 7), "FAILED")
    


if __name__ == '__main__':
    unittest.main()