# Othello / Reversi — Python GUI Project

A polished Python/Tkinter Othello (Reversi) game with animated disc flips, undo and hint tools, and flexible play modes including Human vs Human, Human vs Computer, and Computer vs Computer with adjustable AI difficulty.

## Project Structure

```text
othello_project/
├── main.py
├── README.md
├── requirements.txt
├── .gitignore
├── othello/
│   ├── __init__.py
│   ├── constants.py
│   ├── engine.py
│   ├── greedy_ai.py
│   └── ui.py
├── search/
│   ├── minimax.py
└── tests/
    └── test_engine.py
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
- No third-party Python packages required (standard library only)

## Run

```bash
python main.py
```

## Tests

```bash
python -m unittest discover -s tests -v
```

## Future improvements
- Replace the greedy AI with Minimax / Alpha-Beta search
- Add stronger evaluation heuristics and deeper lookahead
- Add board themes and additional UI customization
- Export and replay match logs
