from othello.engine import OthelloGame, Coord
from typing import Callable, Tuple

#with alpha beta proning
def minimax(
    game: OthelloGame, 
    depth: int, 
    alpha: float, 
    beta: float, 
    is_max: bool,
    heuristic_func: Callable[[OthelloGame], float]
) -> float:
    #already check if none player have legal moves
    if depth == 0 or game.is_game_over():
        return heuristic_func(game)
        
    valid_moves = list(game.legal_moves().keys())
    

    if is_max:
        maxEval = float('-inf')
        for move_coord in valid_moves:
            simulated_game = game.clone()
            simulated_game.apply_move(move_coord[0], move_coord[1])
            

            #check in case next one is still the same player (the component do not have legal move)
            next_is_max = not is_max if simulated_game.current_player != game.current_player else is_max
            eval_score = minimax(simulated_game, depth - 1, alpha, beta, next_is_max, heuristic_func)
            
            maxEval = max(maxEval, eval_score)
            alpha = max(alpha, eval_score)
            
            if beta <= alpha: 
                break
        return maxEval
        
    else:
        minEval = float('inf')
        for move_coord in valid_moves:
            simulated_game = game.clone()   
            simulated_game.apply_move(move_coord[0], move_coord[1])

            #check in case next one is still the same player (the component do not have legal move)
            next_is_max = not is_max if simulated_game.current_player != game.current_player else is_max
            eval_score = minimax(simulated_game, depth - 1, alpha, beta, next_is_max, heuristic_func)
            
            minEval = min(minEval, eval_score)
            beta = min(beta, eval_score)
            
            if beta <= alpha: 
                break
        return minEval


"""
pseudo-code
function minimax(position, depth, alpha, beta, maximizingPlayer)
    if depth == 0 or game over in position
        return static evaluation of position
    if maximizingPlayer 
        maxEval = -infinity
        for each child of position
            eval minimax (child, depth -1, alpha, beta, false)
            maxEval = max (maxEval, eval)
            alpha = max(alph, eval)
            if beta <= alpha
                break
        return maxEval
    else
        minEval = +infinity
        for each child of position
            eval minimax(child, depth -1, alpha, beta, true)
            minEval = min(minEval, eval)
            beta min(beta, eval)
            if beta <= alpha
                break
            return minEval
"""

def get_best_move(game: OthelloGame, depth: int, heuristic_func: Callable[[OthelloGame], float]) -> Coord:
    valid_moves = list(game.legal_moves().keys())
    
    if not valid_moves:
        return None
        
    best_move = None
    best_score = float('-inf')
    
    for move_coord in valid_moves:
        simulated_game = game.clone()
        simulated_game.apply_move(move_coord[0], move_coord[1])
        
        next_is_max = False if simulated_game.current_player != game.current_player else True
        
        score = minimax(
            simulated_game, 
            depth - 1, 
            float('-inf'), 
            float('inf'), 
            next_is_max, 
            heuristic_func
        )
        print("Move")
        print(move_coord[0])
        print(move_coord[1])
        print(score)
        if score > best_score:
            best_score = score
            best_move = move_coord
            
    return best_move, best_score


"""
How to use:
- call get_best_move ()
- it calls for legal moves (with default current_player)
- if none legal move: then return None (but this case it already handle by engine where after each move, the turn is skiped if no legal move and end if no more move for both sides)
--> if return none, then mean both side does not have the legal move -> end game
- find the best branch by store the best of each branch
- return row and col.
"""