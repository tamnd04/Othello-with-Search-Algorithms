"""Othello AI Tournament Runner.

Runs a round-robin tournament between three AI agents:

  - Greedy (Hard): One-ply look-ahead greedy agent.
  - MCTS:          Monte Carlo Tree Search with 300 iterations per move.
  - Minimax:       Alpha-Beta Minimax with iterative deepening to depth 6.

Each matchup consists of 10 games. Colours (Black / White) alternate each
game so that each AI plays first in exactly half the games. Results are
printed after every game and summarised at the end of each matchup. A final
overall win-rate table is printed when all matchups are complete.
"""
from __future__ import annotations

import time
from typing import Tuple

from othello.constants import BLACK, WHITE
from othello.engine import OthelloGame
from search.model import OthelloAgent
from search.greedy import GreedyAgent
from search.mcts import MCTSAgent
from search.minimax import MinimaxAgent

def play_single_game(black: OthelloAgent, white: OthelloAgent) -> str:
    """Play one complete game between two agents and return the winner's name.

    Calls `on_game_start` / `on_game_end` on both agents so that per-game
    state (e.g. transposition tables) is correctly initialised and torn down.

    Args:
        black: The agent playing Black (moves first).
        white: The agent playing White.

    Returns:
        The winning agent's `agent_name()` string, or `"Draw"`.
    """
    game = OthelloGame()

    # Notify both agents that a new game is beginning.
    black.on_game_start()
    white.on_game_start()

    while not game.is_game_over():
        current_player_color = game.current_player

        # If the current player has no legal moves, Othello rules require a pass.
        if not game.legal_moves():
            opponent = WHITE if current_player_color == BLACK else BLACK
            game.current_player = opponent
            continue

        # Dispatch to the correct agent based on whose turn it is.
        agent = black if current_player_color == BLACK else white
        move  = agent.choose_move(game)

        if move is not None:
            game.apply_move(move[0], move[1])
        else:
            # Failsafe: agent returned None despite legal moves existing.
            # Pass the turn to the opponent to avoid an infinite loop.
            opponent = WHITE if current_player_color == BLACK else BLACK
            game.current_player = opponent

    # Notify both agents that the game has ended.
    black.on_game_end()
    white.on_game_end()

    # Determine the winner from the final disc counts.
    scores = game.count_discs()
    if scores[BLACK] > scores[WHITE]:
        return black.agent_name()
    elif scores[WHITE] > scores[BLACK]:
        return white.agent_name()
    else:
        return "Draw"


def run_matchup(
    agent_1:     OthelloAgent,
    agent_2:     OthelloAgent,
    total_games: int = 10,
) -> Tuple[int, int, int]:
    """Run a series of games between two agents and report the results.

    Colours alternate each game so each agent plays Black in half the games.

    Args:
        agent_1:     The first agent.
        agent_2:     The second agent.
        total_games: Number of games to play in this matchup.

    Returns:
        A `(agent_1_wins, agent_2_wins, draws)` tuple.
    """
    name_1 = agent_1.agent_name()
    name_2 = agent_2.agent_name()

    print(f"\n{'=' * 50}")
    print(f" MATCHUP: {name_1} vs {name_2}")
    print(f"{'=' * 50}")

    agent_1_wins = 0
    agent_2_wins = 0
    draws        = 0

    for game_num in range(1, total_games + 1):
        # Alternate which agent plays Black so neither has a colour bias.
        if game_num % 2 != 0:
            black, white = agent_1, agent_2
        else:
            black, white = agent_2, agent_1

        print(
            f"Game {game_num:<2}: "
            f"{f'<Black> {black.agent_name()}':<45} vs. {f'<White> {white.agent_name()}':<45} ",
            end="",
            flush=True,
        )

        start_time = time.time()
        winner     = play_single_game(black, white)
        duration   = time.time() - start_time

        if winner == name_1:
            agent_1_wins += 1
            print(f"--> Winner: {name_1} [{duration:.1f}s]")
        elif winner == name_2:
            agent_2_wins += 1
            print(f"--> Winner: {name_2} [{duration:.1f}s]")
        else:
            draws += 1
            print(f"--> Draw [{duration:.1f}s]")

    print(f"\n--- MATCHUP RESULTS ---")
    print(f"{name_1}: {agent_1_wins} wins")
    print(f"{name_2}: {agent_2_wins} wins")
    print(f"Draws: {draws}")

    return agent_1_wins, agent_2_wins, draws