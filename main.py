from __future__ import annotations

import argparse
import sys


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

_HEURISTIC_CHOICES = ("simple", "smart", "positional", "weight")
_MODEL_CHOICES     = ("greedy", "mcts", "minimax")

# Models that derive value from simulations and do not use a static board-evaluation heuristic.
_HEURISTIC_FREE_MODELS = frozenset({"mcts"})


def _build_heuristic(name: str):
    """Return a heuristic instance for *name*, or exit with an error."""
    from heuristic import SimpleHeuristic, SmartHeuristic, PositionalHeuristic, WeightHeuristic

    registry = {
        "simple":     SimpleHeuristic,
        "smart":      SmartHeuristic,
        "positional": PositionalHeuristic,
        "weight":     WeightHeuristic,
    }
    cls = registry.get(name.lower())
    if cls is None:
        print(
            f"error: unknown heuristic '{name}'. "
            f"Valid choices: {', '.join(_HEURISTIC_CHOICES)}",
            file=sys.stderr,
        )
        sys.exit(2)
    return cls()


def _build_agent(model: str, heuristic_name: str | None, slot: int):
    """Instantiate and return an *OthelloAgent* for *model*.

    Args:
        model:          One of `greedy`, `mcts`, `minimax`.
        heuristic_name: Optional heuristic key; `None` means use the
                        agent's default.
        slot:           `1` or `2` - used only in error messages.
    """
    from search.greedy import GreedyAgent
    from search.mcts import MCTSAgent
    from search.minimax import MinimaxAgent

    key = model.lower()

    if key not in _MODEL_CHOICES:
        print(
            f"error: unknown model '{model}'. "
            f"Valid choices: {', '.join(_MODEL_CHOICES)}",
            file=sys.stderr,
        )
        sys.exit(2)

    # Some models do not use heuristics
    if key in _HEURISTIC_FREE_MODELS and heuristic_name is not None:
        print(
            f"error: --heuristic{slot} is not compatible with model '{model}'. "
            f"{model.upper()} derives move value from random game simulations "
            f"and does not use a static board-evaluation heuristic.",
            file=sys.stderr,
        )
        sys.exit(2)

    heuristic = _build_heuristic(heuristic_name) if heuristic_name else None

    if key == "greedy":
        return GreedyAgent(difficulty="hard", heuristic=heuristic)
    if key == "mcts":
        return MCTSAgent()
    if key == "minimax":
        return MinimaxAgent(heuristic=heuristic)
    # Add more entires here as new models are implemented.


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Othello - launch the interactive GUI (default) or run a "
            "head-to-head AI tournament with --tournament."
        ),
    )
    parser.add_argument(
        "--tournament",
        action="store_true",
        help="Run a tournament between two AI agents instead of launching the GUI.",
    )
    parser.add_argument(
        "--model1",
        metavar="MODEL",
        help=f"First agent ({' | '.join(_MODEL_CHOICES)}). Required with --tournament.",
    )
    parser.add_argument(
        "--model2",
        metavar="MODEL",
        help=f"Second agent ({' | '.join(_MODEL_CHOICES)}). Required with --tournament.",
    )
    parser.add_argument(
        "--heuristic1",
        metavar="HEURISTIC",
        help=(
            f"Heuristic for model1 ({' | '.join(_HEURISTIC_CHOICES)}). "
        ),
    )
    parser.add_argument(
        "--heuristic2",
        metavar="HEURISTIC",
        help=(
            f"Heuristic for model2 ({' | '.join(_HEURISTIC_CHOICES)}). "
        ),
    )
    parser.add_argument(
        "--matches",
        metavar="N",
        type=int,
        default=10,
        help="Number of games to simulate in the tournament matchup (default: 10).",
    )

    args = parser.parse_args()

    if args.tournament:
        if not args.model1 or not args.model2:
            parser.error("--tournament requires both --model1 and --model2.")

        agent1 = _build_agent(args.model1, args.heuristic1, 1)
        agent2 = _build_agent(args.model2, args.heuristic2, 2)

        if args.matches < 1:
            parser.error("--matches must be a positive integer.")

        from tests.tournament import run_matchup
        run_matchup(agent1, agent2, total_games=args.matches)

    else:
        # Warn if the user forgot --tournament but supplied agent flags.
        spurious = [
            f"--{f}" for f in ("model1", "model2", "heuristic1", "heuristic2")
            if getattr(args, f) is not None
        ] + (["--matches"] if args.matches != 10 else [])
        if spurious:
            parser.error(
                f"{', '.join(spurious)} can only be used together with --tournament."
            )

        from othello.ui import launch_app
        launch_app()


if __name__ == "__main__":
    main()
