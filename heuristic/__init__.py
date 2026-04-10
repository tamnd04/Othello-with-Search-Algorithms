"""Othello position-evaluation heuristics.

Exports the abstract base class and all concrete heuristic implementations
so callers can import from a single location::

    from heuristic import OthelloHeuristic, SimpleHeuristic, SmartHeuristic, PositionalHeuristic, WeightHeuristic
"""
from heuristic.base import OthelloHeuristic
from heuristic.simple import SimpleHeuristic
from heuristic.smart import SmartHeuristic
from heuristic.positional import PositionalHeuristic
from heuristic.weight import WeightHeuristic

__all__ = ["OthelloHeuristic", "SimpleHeuristic", "SmartHeuristic", "PositionalHeuristic", "WeightHeuristic"]
