import math
import random
from othello.engine import OthelloGame, Coord
from typing import Tuple, Optional

from tests.test_minimax import simple_heuristic
from .minimaxorder import get_best_move, order_move

class MCTSNode:
    def __init__(self, parent=None, move=None, untried_moves=None, player_just_moved=None):
        self.parent = parent
        self.move = move
        self.children = []

        self.wins = 0.0
        self.visits = 0

        self.untried_moves = untried_moves or []
        self.player_just_moved = player_just_moved
        
        
        

    def uct_select_child(self, exploration_weight=math.sqrt(2)):
        """Step 1: SELECTION (Use UCB1)"""
        return max(
        self.children,
        key=lambda c: (
            float('inf') if c.visits == 0
            else (c.wins / c.visits) +
                 exploration_weight * math.sqrt(math.log(self.visits + 1) / c.visits)
        )
    )

    def expand(self, state: OthelloGame):
        """Step 2: EXPANSION"""
        move = self.untried_moves.pop(0)
        
        # 1. Apply the move directly to the scratchpad board
        state.apply_move(move[0], move[1])
        
        # 2. Get the new untried moves and current player from the updated scratchpad
        new_untried = list(state.legal_moves().keys())
        new_player_just_moved = state.opponent(state.current_player)
        
        # 3. Create the lightweight child node (NO STATE STORED!)
        child_node = MCTSNode(
            parent=self, 
            move=move, 
            untried_moves=new_untried, 
            player_just_moved=new_player_just_moved
        )
        
        self.children.append(child_node)
        return child_node

    def update(self, winner: str):
        """Step 4: BACKPROPAGATION"""
        self.visits += 1
        if winner == self.player_just_moved:
            self.wins += 1.0
        elif winner is None: 
            self.wins += 0.5



def get_best_move_mcts(game: OthelloGame, iterations=150, debug=True, dept_roll=15) -> Optional[Coord]:    
    if not game.legal_moves():
        return None
        
    # Setup the root node manually since __init__ changed
    root_untried = order_move(list(game.legal_moves().keys()))
    root_player = game.opponent(game.current_player)
    root = MCTSNode(parent=None, move=None, untried_moves=root_untried, player_just_moved=root_player)
    
    if debug: 
        print(f"\n========== STARTING MCTS ({iterations} Iterations) ==========")

    for i in range(iterations):
        node = root
        
        # THE ONLY CLONE! This is our scratchpad for this iteration.
        state = game.clone() 

        # 1. SELECTION
        # Walk down the tree, applying moves to our scratchpad
        while not node.untried_moves and node.children:
            node = node.uct_select_child()
            state.apply_move(node.move[0], node.move[1]) 

        # 2. EXPANSION
        if node.untried_moves and not state.is_game_over():
            # Pass the scratchpad to expand. It applies the move and returns the child.
            node = node.expand(state) 

        # 3. SIMULATION (Truncated)
        depth = 0
        while depth < dept_roll:
            valid_moves = list(state.legal_moves().keys())

            if valid_moves:
                move = valid_moves[random.randint(0, len(valid_moves)-1)]
                state.apply_move(*move)
                depth += 1
                continue

            opponent = state.opponent(state.current_player)
            if state.can_player_move(opponent):
                state.current_player = opponent  # PASS
            else:
                break  # Game over

        
        winner = state.winner()

        # 4. BACKPROPAGATION
        while node is not None:
            node.update(winner)
            node = node.parent

    # --- FINAL RESULTS ---
    if not root.children:
        return None
        
    if debug:
        print("\n========== MCTS FINISHED ==========")
        print("Final statistics for all top-level moves:")
        for child in root.children:
            win_rate = (child.wins / child.visits) * 100 if child.visits > 0 else 0
            print(f"  Move {child.move}: Visits={child.visits}, Wins={child.wins} ({win_rate:.1f}% win rate)")

    # Select the child with the most visits
    best_child = max(root.children, key=lambda c: c.visits)
    
    if debug:
        print(f"\n=> BEST MOVE CHOSEN: {best_child.move} (Most visits: {best_child.visits})")
        
    return best_child.move