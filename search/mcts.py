import math
import random
from othello.engine import OthelloGame, Coord
from typing import Optional

from .minimaxorder import  order_move

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
        """
        Step 1: SELECTION
        Uses the UCB1 formula to choose the next node. 
        It balances 'Exploitation' (picking moves with high win rates) 
        and 'Exploration' (picking moves we haven't tried much yet).
        """
        return max(
            self.children,
            key=lambda c: (
                float('inf') if c.visits == 0
                else (c.wins / c.visits) +
                     exploration_weight * math.sqrt(math.log(self.visits + 1) / c.visits)
            )
        )

    def expand(self, state: OthelloGame):
        """
        Step 2: EXPANSION
        Takes an untried move, applies it to the board, and creates a new child node in the tree.
        """
        move = self.untried_moves.pop(0)
        
        # 1. Apply the move directly to the scratchpad board
        state.apply_move(move[0], move[1])
        
        # 2. Get the new valid moves and the player who just moved
        new_untried = list(state.legal_moves().keys())
        new_player_just_moved = state.opponent(state.current_player)
        
        # 3. Create the lightweight child node (We do not store the whole board state to save memory)
        child_node = MCTSNode(
            parent=self, 
            move=move, 
            untried_moves=new_untried, 
            player_just_moved=new_player_just_moved
        )
        
        self.children.append(child_node)
        return child_node

    def update(self, winner: str):
        """
        Step 4: BACKPROPAGATION
        Updates this node's statistics based on the result of the random simulation.
        """
        self.visits += 1
        if winner == self.player_just_moved:
            self.wins += 1.0
        elif winner is None: 
            self.wins += 0.5


def get_best_move_mcts(game: OthelloGame, iterations=150, dept_roll=15) -> Optional[Coord]:    
    if not game.legal_moves():
        return None
        
    # --- INITIALIZATION ---
    # Setup the root node representing the current real board state.
    root_untried = order_move(list(game.legal_moves().keys()))
    root_player = game.opponent(game.current_player)
    root = MCTSNode(parent=None, move=None, untried_moves=root_untried, player_just_moved=root_player)
    
    # Run the MCTS algorithm for the specified number of iterations
    for i in range(iterations):
        node = root
        
        # THE CLONE: We create a temporary scratchpad board so we don't mess up the real game.
        state = game.clone() 

        # --- 1. SELECTION ---
        # Walk down the tree we've built so far using the UCB1 formula,
        # applying the moves to our scratchpad as we go.
        while not node.untried_moves and node.children:
            node = node.uct_select_child()
            state.apply_move(node.move[0], node.move[1]) 

        # --- 2. EXPANSION ---
        # Once we reach a node that still has untried moves, pick one and add it to the tree.
        if node.untried_moves and not state.is_game_over():
            node = node.expand(state) 

        # --- 3. SIMULATION (Rollout) ---
        # From the new node, play completely random moves until the game ends 
        # OR until we hit our depth limit (dept_roll).
        depth = 0
        while depth < dept_roll:
            valid_moves = list(state.legal_moves().keys())

            if valid_moves:
                move = valid_moves[random.randint(0, len(valid_moves)-1)]
                state.apply_move(*move)
                depth += 1
                continue

            # Handle the Othello 'pass' rule
            opponent = state.opponent(state.current_player)
            if state.can_player_move(opponent):
                state.current_player = opponent  
            else:
                break  # Game is entirely over

        # Determine who won the random simulation
        winner = state.winner()

        # --- 4. BACKPROPAGATION ---
        # Take the result of the simulation and pass it all the way back up to the root,
        # updating the win/visit counts for every node we touched along the way.
        while node is not None:
            node.update(winner)
            node = node.parent

    # --- FINAL DECISION ---
    if not root.children:
        return None
        
    best_child = max(root.children, key=lambda c: c.visits)
        
    return best_child.move