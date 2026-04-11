# Othello with Search Algorithms

A Python implementation of Othello (Reversi) featuring an interactive Tkinter GUI and multiple AI search agents.

## What this project includes

- Complete 8x8 Othello engine with legal move generation, pass logic, undo support, and game-over detection
- GUI play modes:
  - Human vs Human
  - Human vs Computer
  - Computer vs Computer (Auto Play/Pause and Step)
- AI agents:
  - Rule-based Greedy
  - Minimax with alpha-beta pruning
  - Monte Carlo Tree Search (MCTS)
- Built-in evaluation panel in the GUI with:
  - Overall match progress bar
  - Per-game ply progress bar
  - Terminal summary table
  - Report export to `agent_evaluation_report.txt`

## Repository structure

```text
.
|-- main.py
|-- README.md
|-- LICENSE
|-- heuristic/
|   |-- __init__.py
|   |-- base.py
|   |-- positional.py
|   |-- simple.py
|   |-- smart.py
|   `-- weight.py
|-- othello/
|   |-- __init__.py
|   |-- constants.py
|   |-- engine.py
|   `-- ui.py
|-- search/
|   |-- greedy.py
|   |-- mcts.py
|   |-- minimax.py
|   `-- model.py
`-- tests/
    |-- test_engine.py
    |-- test_minimax.py
    `-- tournament.py
```

## Requirements

- Python 3.10+
- No third-party dependencies (standard library only)

## Run the GUI

```bash
python main.py
```

## Tournament mode (CLI)

Run head-to-head AI matches directly from the command line.

Models:
- `greedy`
- `minimax`
- `mcts`

Heuristics:
- `simple`
- `smart`
- `positional`
- `weight`

Note: `mcts` does not use static board heuristics, so it cannot be combined with `--heuristic1` or `--heuristic2`.

Examples:

```bash
# Default 10 games
python main.py --tournament --model1 minimax --model2 greedy

# 20 games
python main.py --tournament --model1 mcts --model2 minimax --matches 20

# Explicit heuristics for supported models
python main.py --tournament --model1 minimax --heuristic1 positional --model2 greedy --heuristic2 weight --matches 30
```

## GUI agent evaluation output

The GUI Agent Evaluation section can benchmark one configured agent against another and reports:

- Wins/losses/draws and win rate
- Average disc differential
- Average move time
- Score-rate confidence interval
- Wilson interval for decisive-game win rate
- Elo difference estimate
- Color-split performance (as Black vs as White)
- Average game length and game runtime
- Per-game move logs

A detailed report is saved to `agent_evaluation_report.txt` in the project root.

## Run tests

```bash
python -m unittest discover -s tests -v
```

## Extending the project

Add a new agent:
1. Create a new class in `search/` implementing the `OthelloAgent` interface.
2. Register it in the CLI factory in `main.py`.
3. Register it in GUI mapping logic in `othello/ui.py` if you want it selectable in the app.

Add a new heuristic:
1. Create a new class in `heuristic/` implementing the heuristic base interface.
2. Register it in heuristic factory logic in `main.py`.
3. Wire it into agent configuration logic as needed.
