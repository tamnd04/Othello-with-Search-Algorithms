# Othello / Reversi — Python GUI Project

A polished Python/Tkinter Othello (Reversi) game with animated disc flips, undo and hint tools, and flexible play modes including Human vs Human, Human vs Computer, and Computer vs Computer with adjustable AI difficulty.

## Project Structure

```text
.
├── heuristic
│   ├── base.py             # base class from which all heuristics inherit
│   ├── positional.py
│   ├── simple.py
│   ├── smart.py
│   └── weight.py
├── LICENSE
├── main.py                 # CLI entry point
├── othello
│   ├── constants.py        # tunable parameters for search agents
│   ├── engine.py
│   └── ui.py
├── README.md
├── search
│   ├── greedy.py
│   ├── mcts.py
│   ├── minimax.py
│   └── model.py            # base class from which all agents inherit
└── tests
    ├── test_engine.py
    ├── test_minimax.py
    └── tournament.py       # runs a series of games between two agents and reports the results
```

## Features

### Game engine
- Standard 8x8 Othello board
- Legal move detection in all 8 directions
- Automatic disc flipping
- Turn switching and automatic pass handling
- Game-over detection when neither side can move
- Winner detection and score counting
- Undo support through state snapshots

### Interface and game modes
- Modern Tkinter UI with board labels, hover highlight, move log, and score panel
- Flip animation for captured discs
- Hint toggle button (enabled by default)
- Human vs Human mode
- Human vs Computer mode with selectable AI difficulty
- Computer vs Computer mode with separate Black and White AI difficulty
- Auto Play / Pause and Step controls for Computer vs Computer simulations

## Requirements

- Python 3.10+ recommended

## Run

```bash
# Run the Othello GUI application
python main.py

# Tournament - models with default heuristics
python main.py --tournament --model1 minimax --model2 greedy

# Tournament - models with explicit heuristics
python main.py --tournament --model1 minimax --heuristic1 positional --model2 greedy --heuristic2 weight
```

## Add a new model
1. Implement the model in a new file under `search/`, e.g. `search/new.py` with a class `NewAgent` that inherits from `OthelloAgent`.
2. Import the new model in `search/__init__.py` and add it to `__all__`.
3. Import the new model in `main.py` and add it to `_MODEL_CHOICES` and the model factory logic in `main()`.

## Add a new heuristic
1. Implement the heuristic in a new file under `heuristic/`, e.g. `heuristic/my_heuristic.py` with a class `MyHeuristic` that inherits from `OthelloHeuristic`.
2. Import the new heuristic in `heuristic/__init__.py` and add it to `__all__`.
3. Add the new heuristic to `_HEURISTIC_CHOICES` and the heuristic factory logic in `main()`.
