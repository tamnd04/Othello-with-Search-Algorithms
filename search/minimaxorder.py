import time
from othello.engine import OthelloGame, Coord
from typing import Callable, Tuple, Optional

# This Othello engine uses Minimax with Alpha-Beta Pruning, Move Ordering, 
# Transposition Tables, and Iterative Deepening to maximize search depth and speed.

# --- TRANSPOSITION TABLE FLAGS ---
# EXACT: We searched this branch fully. The score is 100% accurate.
# LOWERBOUND: The move was too good (Beta cutoff). The score is AT LEAST this much.
# UPPERBOUND: The move was terrible (Alpha cutoff). The score is AT MOST this much.
# Reference: https://webdocs.cs.ualberta.ca/~mmueller/courses/2014-AAAI-games-tutorial/slides/AAAI-14-Tutorial-Games-3-AlphaBeta.pdf
EXACT = 0
LOWERBOUND = 1
UPPERBOUND = 2

# A cache to remember previously calculated board states and save processing time.
transposition_table = {}

# --- POSITIONAL WEIGHT MATRIX ---
# Static priority grid for an 8x8 Othello board. 
# Corners (5) are highly prioritized, while squares next to corners (1, 2) are avoided.
# Reference: https://www.researchgate.net/figure/The-move-ordering-for-Othello_fig1_326235384

move_priority = [
    [5, 2, 4, 3, 3, 4, 2, 5],
    [2, 1, 3, 3, 3, 3, 1, 2],
    [4, 3, 4, 4, 4, 4, 3, 4],
    [3, 3, 4, 4, 4, 4, 3, 3],
    [3, 3, 4, 4, 4, 4, 3, 3],
    [4, 3, 4, 4, 4, 4, 3, 4],
    [2, 1, 3, 3, 3, 3, 1, 2],
    [5, 2, 4, 3, 3, 4, 2, 5]
]

FLAT_PRIORITY = {(r, c): move_priority[r][c] for r in range(8) for c in range(8)}

# Sorts valid moves so the AI checks the most promising squares first.
def order_move(valid_moves: list) -> list:
    return sorted(valid_moves, key=FLAT_PRIORITY.get, reverse=True)

# Converts the current board layout and player turn into a unique string ID.
# This ID acts as the lookup key for the transposition table.
def board_to_key(game: OthelloGame):
    return (str(game.board), game.current_player)

def minimax_order(
    game: OthelloGame, 
    depth: int, 
    alpha: float, 
    beta: float, 
    is_max: bool,
    heuristic_func: Callable[[OthelloGame], float]
) -> float:
    if depth == 0 or game.is_game_over():
        return heuristic_func(game)
    
    key = board_to_key(game)
    tt_best_move = None

    # --- TRANSPOSITION TABLE LOOKUP ---
    # Check if we have analyzed this exact board state before.
    # If we have, and we searched deep enough last time, we can reuse the saved score.
    if key in transposition_table:
        tt_entry = transposition_table[key]
        tt_best_move = tt_entry.get('best_move') 
        
        if tt_entry['depth'] >= depth:
            tt_score = tt_entry['score']
            tt_flag = tt_entry['flag']
            
            if tt_flag == EXACT:
                return tt_score
            elif tt_flag == LOWERBOUND:
                alpha = max(alpha, tt_score)
            elif tt_flag == UPPERBOUND:
                beta = min(beta, tt_score)
                
            if alpha >= beta:
                return tt_score 

    valid_moves = list(game.legal_moves().keys())
    
    # Put the best move from the transposition table at the very front of the line.
    if tt_best_move and tt_best_move in valid_moves:
        ordered_moves = [tt_best_move] + order_move([m for m in valid_moves if m != tt_best_move])
    else:
        ordered_moves = order_move(valid_moves)
    
    best_move_this_node = None
    
    if is_max:
        maxEval = float('-inf')
        original_alpha = alpha
        
        for move_coord in ordered_moves:
            current_player = game.current_player
            game.apply_move(move_coord[0], move_coord[1])
            
            next_is_max = not is_max if game.current_player != current_player else is_max
            eval_score = minimax_order(game, depth - 1, alpha, beta, next_is_max, heuristic_func)
            
            if eval_score > maxEval:
                maxEval = eval_score
                best_move_this_node = move_coord
                
            alpha = max(alpha, eval_score)
            game.undo()
            
            if beta <= alpha: 
                break
                
        flag = EXACT
        if maxEval <= original_alpha:
            flag = UPPERBOUND
        elif maxEval >= beta:
            flag = LOWERBOUND
            
        tt_entry = transposition_table.get(key)
        if tt_entry is None or depth >= tt_entry['depth']:
            transposition_table[key] = {
                'score': maxEval, 
                'flag': flag, 
                'depth': depth,
                'best_move': best_move_this_node
            }

        return maxEval
        
    else:
        minEval = float('inf')
        original_beta = beta
        
        for move_coord in ordered_moves:
            current_player = game.current_player
            game.apply_move(move_coord[0], move_coord[1])

            next_is_max = not is_max if game.current_player != current_player else is_max
            eval_score = minimax_order(game, depth - 1, alpha, beta, next_is_max, heuristic_func)
            
            if eval_score < minEval:
                minEval = eval_score
                best_move_this_node = move_coord
                
            beta = min(beta, eval_score)
            game.undo()
            
            if beta <= alpha: 
                break
                
        flag = EXACT
        if minEval <= alpha:
            flag = UPPERBOUND
        elif minEval >= original_beta:
            flag = LOWERBOUND
            
        tt_entry = transposition_table.get(key)
        if tt_entry is None or depth >= tt_entry['depth']:
            transposition_table[key] = {
                'score': minEval, 
                'flag': flag, 
                'depth': depth,
                'best_move': best_move_this_node
            }
            
        return minEval


def get_best_move(
    game: OthelloGame, 
    max_depth: int, 
    heuristic_func: Callable[[OthelloGame], float],
    time_limit: float = 5.0 
) -> Tuple[Optional[Coord], float]:
    
    valid_moves = list(game.legal_moves().keys())
    
    if not valid_moves:
        return None, heuristic_func(game)
        
    best_move_overall = None
    best_score_overall = float('-inf')
    
    start_time = time.time()
    
    # --- ITERATIVE DEEPENING ---
    # Iterative Deepening searches the game tree at Depth 1, then Depth 2, then Depth 3, etc., 
    # instead of just jumping straight to the maximum depth. 
    for current_depth in range(1, max_depth + 1):
        alpha = float('-inf')
        beta = float('inf')
        
        best_move_this_depth = None
        best_score_this_depth = float('-inf')
        
        # Take the "best move" found in the previous, put it at the very front.
        if best_move_overall and best_move_overall in valid_moves:
            ordered_moves = [best_move_overall] + order_move([m for m in valid_moves if m != best_move_overall])
        else:
            ordered_moves = order_move(valid_moves)
        
        for move_coord in ordered_moves:

            # TIME MANAGEMENT: Check the clock before evaluating a new move. 
            # If out of time, immediately abort and safely return the best move found so far.
            if time.time() - start_time > time_limit:
                fallback_move = best_move_overall if best_move_overall else move_coord
                return fallback_move, best_score_overall

            current_player = game.current_player
            game.apply_move(move_coord[0], move_coord[1])
            
            # Handle the Othello 'pass' rule: if a move causes the opponent to have zero legal moves, 
            # the turn skips back to the original player.
            next_is_max = False if game.current_player != current_player else True
            
            score = minimax_order(
                game, 
                current_depth - 1, 
                alpha, 
                beta, 
                next_is_max, 
                heuristic_func
            )

            if score > best_score_this_depth:
                best_score_this_depth = score
                best_move_this_depth = move_coord
                
            alpha = max(alpha, best_score_this_depth)
            game.undo()
            
        # Update our overall best move before starting the next, deeper iteration.
        best_move_overall = best_move_this_depth
        best_score_overall = best_score_this_depth
            
    return best_move_overall, best_score_overall