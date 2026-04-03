import unittest
from othello.constants import BOARD_SIZE, BLACK, EMPTY, WHITE
from othello.engine import OthelloGame

# Giả sử hàm get_best_move của bạn nằm trong file ai.py
# (Hãy sửa lại đường dẫn import này cho đúng với project của bạn nhé)
from search.minimax import get_best_move 

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

    def test_mate_in_one(self):
        """
        Test 1: AI (Đen) có một nước đi lật cờ để kết thúc ván và Thắng.
        Nó PHẢI chọn được nước đi đó thay vì các nước đi khác.
        """
        board_str = """
        ........
        ......W.
        .....W..
        ....BW..
        ........
        ........
        ........
        ........
        """
        game = setup_custom_board(board_str, current_turn=BLACK)
        
        best_move = get_best_move(game, depth=2, heuristic_func=simple_heuristic)
        
        self.assertEqual(best_move, (0, 7), "AI đã không tìm ra nước đi Win in 1!")


    def test_force_opponent_to_pass(self):
        """
        Test 2: AI (Đen) đứng trước 2 lựa chọn.
        Một lựa chọn bình thường, và một lựa chọn khiến Trắng bị PASS (mất lượt).
        AI PHẢI tìm ra nước đi ép đối thủ Pass.
        """
        # Bàn cờ kịch bản:
        # Nếu Đen đánh (7, 7) -> Trắng mất lượt.
        board_str = """
        BBBBBBBB
        BBBBBBBB
        BBBBBBBB
        BBBBBBBW
        BBBBBB.W
        BBBBBB.W
        BBBBBB.W
        BBBBBB..
        """
        game = setup_custom_board(board_str, current_turn=BLACK)
        
        best_move = get_best_move(game, depth=3, heuristic_func=simple_heuristic)
        game.apply_move(best_move[0], best_move[1])
        best_move = get_best_move(game, depth=3, heuristic_func=simple_heuristic)
        
        if (not best_move):
            print("hu")
            return
        self.assertEqual(best_move, (7, 7), "AI đã không biết cách ép đối thủ mất lượt!")


if __name__ == '__main__':
    unittest.main()