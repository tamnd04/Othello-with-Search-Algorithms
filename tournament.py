import time
from typing import Callable, Tuple

# Import game engine and constants
from othello.engine import OthelloGame
from othello.constants import BLACK, WHITE, EMPTY

# Import AI agents
from othello.greedy_ai import GreedyAI
from search.minimaxorder import get_best_move_minimax
from search.mcts import get_best_move_mcts
from tests.test_minimax import simple_heuristic, smart_heuristic

# ---------------------------------------------------------
# 1. AI WRAPPER FUNCTIONS
# ---------------------------------------------------------
def greedy_hard_agent(game: OthelloGame) -> Tuple[int, int]:
    return GreedyAI().choose_move(game, difficulty="hard")

def mcts_agent(game: OthelloGame) -> Tuple[int, int]:
    return get_best_move_mcts(game, iterations=300, dept_roll=20)

def minimax_agent(game: OthelloGame) -> Tuple[int, int]:
    move, score = get_best_move_minimax(game, max_depth=6, heuristic_func=smart_heuristic, time_limit=10)
    return move

# Dictionary to easily reference the agents by name
AGENTS = {
    "Greedy (Hard)": greedy_hard_agent,
    "MCTS": mcts_agent,
    "Minimax": minimax_agent
}

# ---------------------------------------------------------
# 2. MATCH EXECUTION LOGIC
# ---------------------------------------------------------
def play_single_game(black_name: str, white_name: str) -> str:
    """Plays one full game between two AIs and returns the winner's name (or 'Draw')."""
    game = OthelloGame()
    
    black_func = AGENTS[black_name]
    white_func = AGENTS[white_name]

    while not game.is_game_over():
        current_player_color = game.current_player
        
        # If the player has no legal moves, Othello rules dictate they pass.
        if not game.legal_moves():
            opponent = WHITE if current_player_color == BLACK else BLACK
            game.current_player = opponent
            continue

        # Choose the correct AI function based on whose turn it is
        if current_player_color == BLACK:
            move = black_func(game)
        else:
            move = white_func(game)

        # Apply the move
        if move is not None:
            game.apply_move(move[0], move[1])
        else:
            # Failsafe if AI returns None despite having legal moves
            opponent = WHITE if current_player_color == BLACK else BLACK
            game.current_player = opponent

    # Game Over. Determine the winner based on piece count.
    scores = game.count_discs()
    if scores[BLACK] > scores[WHITE]:
        return black_name
    elif scores[WHITE] > scores[BLACK]:
        return white_name
    else:
        return "Draw"

def run_matchup(ai_1_name: str, ai_2_name: str, total_games: int = 10) -> Tuple[int, int, int]:
    """Runs a series of games between two AIs, returning the win counts."""
    print(f"\n==================================================")
    print(f" MATCHUP: {ai_1_name} vs {ai_2_name}")
    print(f"==================================================")
    
    ai_1_wins = 0
    ai_2_wins = 0
    draws = 0

    for game_num in range(1, total_games + 1):
        # Swap colors so each AI plays Black (goes first) 50% of the time
        if game_num % 2 != 0:
            black = ai_1_name
            white = ai_2_name
        else:
            black = ai_2_name
            white = ai_1_name
            
        start_time = time.time()
        print(f"Game {game_num:02d} ({black} as Black vs {white} as White)... ", end="", flush=True)
        
        winner = play_single_game(black, white)
        duration = time.time() - start_time
        
        if winner == ai_1_name:
            ai_1_wins += 1
            print(f"Winner: {ai_1_name} ({duration:.1f}s)")
        elif winner == ai_2_name:
            ai_2_wins += 1
            print(f"Winner: {ai_2_name} ({duration:.1f}s)")
        else:
            draws += 1
            print(f"Draw ({duration:.1f}s)")

    print(f"\n--- MATCHUP RESULTS ---")
    print(f"{ai_1_name}: {ai_1_wins} wins")
    print(f"{ai_2_name}: {ai_2_wins} wins")
    print(f"Draws: {draws}")
    
    return ai_1_wins, ai_2_wins, draws


# ---------------------------------------------------------
# 3. MAIN TOURNAMENT LOOP
# ---------------------------------------------------------
if __name__ == "__main__":
    print("Starting Othello AI Tournament...\n")
    
    # Track overall wins for the final percentage calculation
    global_scores = {
        "Greedy (Hard)": 0,
        "MCTS (200 Iterations)": 0,
        "Minimax (Depth 6)": 0
    }
    
    # 1. Greedy vs MCTS (10 Games)
    w1, w2, draws = run_matchup("Greedy (Hard)", "MCTS", total_games=10)
    global_scores["Greedy (Hard)"] += w1
    global_scores["MCTS"] += w2
    
    # 2. Greedy vs Minimax (10 Games)
    w1, w2, draws = run_matchup("Greedy (Hard)", "Minimax", total_games=10)
    global_scores["Greedy (Hard)"] += w1
    global_scores["Minimax"] += w2
    
    # 3. MCTS vs Minimax (10 Games)
    w1, w2, draws = run_matchup("MCTS", "Minimax", total_games=10)
    global_scores["MCTS"] += w1
    global_scores["Minimax"] += w2
    
    print("\n==================================================")
    print(" TOURNAMENT COMPLETE - FINAL WIN PERCENTAGES")
    print("==================================================")
    
    # Each AI plays 20 games total in this round-robin tournament
    TOTAL_GAMES_PER_AI = 20 
    
    for ai_name, total_wins in global_scores.items():
        win_percentage = (total_wins / TOTAL_GAMES_PER_AI) * 100
        print(f"{ai_name}: {win_percentage:.1f}% Win Rate ({total_wins}/{TOTAL_GAMES_PER_AI} games)")