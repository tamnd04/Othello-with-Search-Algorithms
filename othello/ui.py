"""Tkinter-based graphical user interface for Othello.

Provides an interactive board, a sidebar with match status and controls,
and support for three game modes:

- **Human vs Human**    - two players share the same keyboard/mouse.
- **Human vs Computer** - the human plays Black; the AI plays White.
- **Computer vs Computer** - both sides are driven by AI agents, with
  optional Auto Play or single-step controls.

AI players are selected in the GUI and instantiated at move time via
:meth:`OthelloUI._agent_for_config`, supporting:

- :class:`~search.greedy.GreedyAgent` (rule-based)
- :class:`~search.minimax.MinimaxAgent` (search-based)
- :class:`~search.mcts.MCTSAgent` (search-based)

Each agent uses a user-selected skill level in the range 1..10.
"""
from __future__ import annotations

import math
import os
import statistics
import time
import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Optional, Tuple

from heuristic import PositionalHeuristic, SimpleHeuristic, SmartHeuristic
from .constants import BLACK, BOARD_SIZE, EMPTY, PLAYER_COLORS, PLAYER_NAMES, WHITE
from .engine import OthelloGame
from search.greedy import GreedyAgent
from search.mcts import MCTSAgent
from search.minimax import MinimaxAgent
from search.model import OthelloAgent

Coord = Tuple[int, int]


class OthelloUI(tk.Tk):
    AGENT_CHOICES = ("Rule-based (Greedy)", "Minimax", "MCTS")

    CELL_SIZE = 80
    BOARD_PADDING = 34
    BOARD_PIXELS = CELL_SIZE * BOARD_SIZE
    CANVAS_SIZE = BOARD_PIXELS + BOARD_PADDING * 2

    BG = "#0B1120"
    PANEL = "#111827"
    PANEL_ALT = "#172033"
    PANEL_SOFT = "#1E293B"
    TEXT = "#E5E7EB"
    MUTED = "#94A3B8"
    ACCENT = "#38BDF8"
    SUCCESS = "#34D399"
    BOARD = "#0F7A47"
    BOARD_DARK = "#0B5E37"
    GRID = "#E2E8F0"
    HINT = "#A7F3D0"
    HOVER = "#FDE68A"
    STAR_POINT = "#7DD3FC"

    FLIP_ANIMATION_STEPS = 8
    FLIP_ANIMATION_FRAME_MS = 42

    def __init__(self) -> None:
        """Initialise the application window, game state, and all UI widgets."""
        super().__init__()
        self.title("Othello - Python GUI")
        self.configure(bg=self.BG)
        self.geometry("1460x920")
        self.minsize(1360, 860)

        self.game = OthelloGame()
        self.vs_computer = tk.BooleanVar(value=False)
        self.computer_vs_computer = tk.BooleanVar(value=False)
        self.cvc_autoplay = tk.BooleanVar(value=False)
        self.white_agent_var = tk.StringVar(value="Rule-based (Greedy)")
        self.black_agent_var = tk.StringVar(value="Rule-based (Greedy)")
        self.white_level_var = tk.IntVar(value=6)
        self.black_level_var = tk.IntVar(value=6)
        self.white_ai_title_var = tk.StringVar(value="White AI Agent")
        self.white_skill_var = tk.StringVar(value="")
        self.black_skill_var = tk.StringVar(value="")

        self.eval_search_agent_var = tk.StringVar(value="Minimax")
        self.eval_search_level_var = tk.IntVar(value=8)
        self.eval_opponent_agent_var = tk.StringVar(value="Rule-based (Greedy)")
        self.eval_opponent_level_var = tk.IntVar(value=6)
        self.eval_matches_var = tk.IntVar(value=20)
        self.eval_progress_var = tk.DoubleVar(value=0.0)
        self.eval_progress_text_var = tk.StringVar(value="Progress: 0 / 0")
        self.eval_game_progress_var = tk.DoubleVar(value=0.0)
        self.eval_game_progress_text_var = tk.StringVar(value="Current game: not started")
        self.eval_output_path = os.path.join(os.getcwd(), "agent_evaluation_report.txt")

        self.show_hints = tk.BooleanVar(value=True)
        self.hovered_cell: Optional[Coord] = None
        self.move_counter = 0
        self.flip_animation_cells: List[Coord] = []
        self.flip_animation_step = 0
        self.flip_animation_to_player: Optional[str] = None
        self.flip_animation_job: Optional[str] = None

        self._configure_style()
        self._build_layout()
        self._refresh_ui()

    def _configure_style(self) -> None:
        """Apply a dark-theme ttk style used by sidebar widgets."""
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(
            "Sidebar.TFrame",
            background=self.PANEL,
        )
        style.configure(
            "Card.TFrame",
            background=self.PANEL_ALT,
        )
        style.configure(
            "Title.TLabel",
            background=self.PANEL,
            foreground=self.TEXT,
            font=("Segoe UI", 22, "bold"),
        )
        style.configure(
            "CardTitle.TLabel",
            background=self.PANEL_ALT,
            foreground=self.TEXT,
            font=("Segoe UI", 12, "bold"),
        )
        style.configure(
            "CardValue.TLabel",
            background=self.PANEL_ALT,
            foreground=self.TEXT,
            font=("Segoe UI", 12),
        )
        style.configure(
            "Muted.TLabel",
            background=self.PANEL,
            foreground=self.MUTED,
            font=("Segoe UI", 10),
        )

        # Keep combobox styling conservative for reliable click behavior on Windows Tk.

    def _build_layout(self) -> None:
        """Construct the two-column layout: board canvas on the left, sidebar on the right."""
        self.columnconfigure(0, weight=5)
        self.columnconfigure(1, weight=3)
        self.rowconfigure(0, weight=1)

        board_shell = tk.Frame(self, bg=self.BG, padx=20, pady=20)
        board_shell.grid(row=0, column=0, sticky="nsew")
        board_shell.rowconfigure(0, weight=1)
        board_shell.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            board_shell,
            width=self.CANVAS_SIZE,
            height=self.CANVAS_SIZE,
            bg=self.BG,
            bd=0,
            highlightthickness=0,
            relief="flat",
            cursor="hand2",
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Motion>", self._on_canvas_motion)
        self.canvas.bind("<Leave>", self._on_canvas_leave)

        sidebar = ttk.Frame(self, style="Sidebar.TFrame", padding=22)
        sidebar.grid(row=0, column=1, sticky="nsew")
        sidebar.columnconfigure(0, weight=1)

        ttk.Label(sidebar, text="Othello", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            sidebar,
            text="Python GUI",
            style="Muted.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(0, 18))

        self.status_var = tk.StringVar(value="")
        self.score_var = tk.StringVar(value="")
        self.mode_var = tk.StringVar(value="Human vs Human")
        self.turn_var = tk.StringVar(value="")

        status_card = self._make_card(sidebar, "Match Status")
        status_card.grid(row=2, column=0, sticky="ew", pady=(0, 14))
        ttk.Label(status_card, textvariable=self.status_var, style="CardValue.TLabel", wraplength=320).grid(
            row=1, column=0, sticky="w", pady=(8, 4)
        )
        ttk.Label(status_card, textvariable=self.turn_var, style="CardValue.TLabel").grid(
            row=2, column=0, sticky="w"
        )
        ttk.Label(status_card, textvariable=self.score_var, style="CardValue.TLabel").grid(
            row=3, column=0, sticky="w", pady=(4, 0)
        )

        controls_card = self._make_card(sidebar, "Controls")
        controls_card.grid(row=3, column=0, sticky="ew", pady=(0, 14))

        mode_toggle = tk.Checkbutton(
            controls_card,
            text="Human vs Computer",
            variable=self.vs_computer,
            command=self._toggle_human_vs_computer,
            bg=self.PANEL_ALT,
            fg=self.TEXT,
            selectcolor=self.PANEL_SOFT,
            activebackground=self.PANEL_ALT,
            activeforeground=self.TEXT,
            font=("Segoe UI", 11),
            relief="flat",
            bd=0,
            highlightthickness=0,
        )
        mode_toggle.grid(row=1, column=0, sticky="w", pady=(8, 4))

        cvc_toggle = tk.Checkbutton(
            controls_card,
            text="Computer vs Computer",
            variable=self.computer_vs_computer,
            command=self._toggle_computer_vs_computer,
            bg=self.PANEL_ALT,
            fg=self.TEXT,
            selectcolor=self.PANEL_SOFT,
            activebackground=self.PANEL_ALT,
            activeforeground=self.TEXT,
            font=("Segoe UI", 11),
            relief="flat",
            bd=0,
            highlightthickness=0,
        )
        cvc_toggle.grid(row=2, column=0, sticky="w", pady=(0, 8))

        self.black_ai_frame = tk.Frame(controls_card, bg=self.PANEL_ALT)
        self.black_ai_frame.grid(row=3, column=0, sticky="ew", pady=(2, 8))
        ttk.Label(self.black_ai_frame, text="Black AI Agent", style="CardValue.TLabel").grid(
            row=0, column=0, sticky="w", columnspan=3
        )
        self.black_agent_combo = ttk.Combobox(
            self.black_ai_frame,
            state="readonly",
            values=self.AGENT_CHOICES,
            textvariable=self.black_agent_var,
            width=18,
        )
        self.black_agent_combo.grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.black_agent_combo.bind("<<ComboboxSelected>>", self._on_ai_config_change)
        ttk.Label(self.black_ai_frame, text="Level", style="CardValue.TLabel").grid(
            row=1, column=1, padx=(10, 4), sticky="w"
        )
        self.black_level_spin = tk.Spinbox(
            self.black_ai_frame,
            from_=1,
            to=10,
            textvariable=self.black_level_var,
            width=4,
            command=self._on_ai_config_change,
            bg=self.PANEL_SOFT,
            fg=self.TEXT,
            insertbackground=self.TEXT,
            readonlybackground=self.PANEL_SOFT,
            buttonbackground=self.PANEL,
            relief="flat",
            highlightthickness=0,
        )
        self.black_level_spin.grid(row=1, column=2, sticky="w")
        self.black_level_spin.bind("<FocusOut>", self._on_ai_config_change)
        self.black_level_spin.bind("<Return>", self._on_ai_config_change)
        ttk.Label(self.black_ai_frame, textvariable=self.black_skill_var, style="Muted.TLabel").grid(
            row=2, column=0, sticky="w", columnspan=3, pady=(2, 0)
        )

        self.white_ai_frame = tk.Frame(controls_card, bg=self.PANEL_ALT)
        self.white_ai_frame.grid(row=4, column=0, sticky="ew", pady=(2, 8))
        ttk.Label(self.white_ai_frame, textvariable=self.white_ai_title_var, style="CardValue.TLabel").grid(
            row=0, column=0, sticky="w", columnspan=3
        )
        self.white_agent_combo = ttk.Combobox(
            self.white_ai_frame,
            state="readonly",
            values=self.AGENT_CHOICES,
            textvariable=self.white_agent_var,
            width=18,
        )
        self.white_agent_combo.grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.white_agent_combo.bind("<<ComboboxSelected>>", self._on_ai_config_change)
        ttk.Label(self.white_ai_frame, text="Level", style="CardValue.TLabel").grid(
            row=1, column=1, padx=(10, 4), sticky="w"
        )
        self.white_level_spin = tk.Spinbox(
            self.white_ai_frame,
            from_=1,
            to=10,
            textvariable=self.white_level_var,
            width=4,
            command=self._on_ai_config_change,
            bg=self.PANEL_SOFT,
            fg=self.TEXT,
            insertbackground=self.TEXT,
            readonlybackground=self.PANEL_SOFT,
            buttonbackground=self.PANEL,
            relief="flat",
            highlightthickness=0,
        )
        self.white_level_spin.grid(row=1, column=2, sticky="w")
        self.white_level_spin.bind("<FocusOut>", self._on_ai_config_change)
        self.white_level_spin.bind("<Return>", self._on_ai_config_change)
        ttk.Label(self.white_ai_frame, textvariable=self.white_skill_var, style="Muted.TLabel").grid(
            row=2, column=0, sticky="w", columnspan=3, pady=(2, 0)
        )

        button_row = tk.Frame(controls_card, bg=self.PANEL_ALT)
        button_row.grid(row=5, column=0, sticky="ew")

        self._make_button(button_row, "New Game", self._new_game).grid(row=0, column=0, padx=(0, 8), pady=4)
        self._make_button(button_row, "Undo", self._undo).grid(row=0, column=1, padx=(0, 8), pady=4)
        self.hint_button = self._make_button(button_row, "Hint: Off", self._flash_hint)
        self.hint_button.configure(width=10)
        self.hint_button.grid(row=0, column=2, pady=4)

        self.cvc_button_row = tk.Frame(controls_card, bg=self.PANEL_ALT)
        self.cvc_button_row.grid(row=6, column=0, sticky="w", pady=(4, 0))

        self.autoplay_button = self._make_button(self.cvc_button_row, "Auto Play", self._toggle_cvc_autoplay)
        self.autoplay_button.configure(width=10, state="disabled")
        self.autoplay_button.grid(row=0, column=0, pady=4)

        self.step_button = self._make_button(self.cvc_button_row, "Step", self._step_cvc_once)
        self.step_button.configure(width=10, state="disabled")
        self.step_button.grid(row=0, column=1, padx=(8, 0), pady=4)

        info_card = self._make_card(sidebar, "Mode")
        info_card.grid(row=4, column=0, sticky="ew", pady=(0, 14))
        ttk.Label(info_card, textvariable=self.mode_var, style="CardValue.TLabel").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )

        eval_card = self._make_card(sidebar, "Agent Evaluation")
        eval_card.grid(row=5, column=0, sticky="ew", pady=(0, 14))

        ttk.Label(eval_card, text="Agent A", style="CardValue.TLabel").grid(
            row=1, column=0, sticky="w", pady=(8, 4)
        )
        self.eval_search_combo = ttk.Combobox(
            eval_card,
            state="readonly",
            values=self.AGENT_CHOICES,
            textvariable=self.eval_search_agent_var,
            width=20,
        )
        self.eval_search_combo.grid(row=2, column=0, sticky="w")

        ttk.Label(eval_card, text="Agent B", style="CardValue.TLabel").grid(
            row=3, column=0, sticky="w", pady=(6, 4)
        )
        self.eval_opponent_combo = ttk.Combobox(
            eval_card,
            state="readonly",
            values=self.AGENT_CHOICES,
            textvariable=self.eval_opponent_agent_var,
            width=20,
        )
        self.eval_opponent_combo.grid(row=4, column=0, sticky="w")

        eval_levels = tk.Frame(eval_card, bg=self.PANEL_ALT)
        eval_levels.grid(row=5, column=0, sticky="w", pady=(6, 0))
        ttk.Label(eval_levels, text="A Level", style="CardValue.TLabel").grid(row=0, column=0, sticky="w")
        self.eval_search_level_spin = tk.Spinbox(
            eval_levels,
            from_=1,
            to=10,
            textvariable=self.eval_search_level_var,
            width=4,
            bg=self.PANEL_SOFT,
            fg=self.TEXT,
            insertbackground=self.TEXT,
            readonlybackground=self.PANEL_SOFT,
            buttonbackground=self.PANEL,
            relief="flat",
            highlightthickness=0,
        )
        self.eval_search_level_spin.grid(row=0, column=1, padx=(6, 12), sticky="w")

        ttk.Label(eval_levels, text="B Level", style="CardValue.TLabel").grid(row=0, column=2, sticky="w")
        self.eval_opponent_level_spin = tk.Spinbox(
            eval_levels,
            from_=1,
            to=10,
            textvariable=self.eval_opponent_level_var,
            width=4,
            bg=self.PANEL_SOFT,
            fg=self.TEXT,
            insertbackground=self.TEXT,
            readonlybackground=self.PANEL_SOFT,
            buttonbackground=self.PANEL,
            relief="flat",
            highlightthickness=0,
        )
        self.eval_opponent_level_spin.grid(row=0, column=3, padx=(6, 0), sticky="w")

        eval_matches = tk.Frame(eval_card, bg=self.PANEL_ALT)
        eval_matches.grid(row=6, column=0, sticky="w", pady=(6, 0))
        ttk.Label(eval_matches, text="Matches", style="CardValue.TLabel").grid(row=0, column=0, sticky="w")
        self.eval_matches_spin = tk.Spinbox(
            eval_matches,
            from_=1,
            to=500,
            textvariable=self.eval_matches_var,
            width=6,
            bg=self.PANEL_SOFT,
            fg=self.TEXT,
            insertbackground=self.TEXT,
            readonlybackground=self.PANEL_SOFT,
            buttonbackground=self.PANEL,
            relief="flat",
            highlightthickness=0,
        )
        self.eval_matches_spin.grid(row=0, column=1, padx=(8, 0), sticky="w")

        self.eval_button = self._make_button(eval_card, "Run Evaluation", self._run_agent_evaluation)
        self.eval_button.grid(row=7, column=0, sticky="w", pady=(8, 0))

        ttk.Label(
            eval_card,
            text=f"Output: {os.path.basename(self.eval_output_path)}",
            style="Muted.TLabel",
        ).grid(row=8, column=0, sticky="w", pady=(6, 0))

        ttk.Label(eval_card, textvariable=self.eval_progress_text_var, style="Muted.TLabel").grid(
            row=9, column=0, sticky="w", pady=(6, 2)
        )
        self.eval_progress_bar = ttk.Progressbar(
            eval_card,
            orient="horizontal",
            mode="determinate",
            maximum=100.0,
            variable=self.eval_progress_var,
            length=280,
        )
        self.eval_progress_bar.grid(row=10, column=0, sticky="ew", pady=(0, 2))

        ttk.Label(eval_card, textvariable=self.eval_game_progress_text_var, style="Muted.TLabel").grid(
            row=11, column=0, sticky="w", pady=(6, 2)
        )
        self.eval_game_progress_bar = ttk.Progressbar(
            eval_card,
            orient="horizontal",
            mode="determinate",
            maximum=100.0,
            variable=self.eval_game_progress_var,
            length=280,
        )
        self.eval_game_progress_bar.grid(row=12, column=0, sticky="ew", pady=(0, 2))

        log_card = self._make_card(sidebar, "Move Log")
        log_card.grid(row=6, column=0, sticky="nsew")
        sidebar.rowconfigure(6, weight=1)

        self.log_box = tk.Listbox(
            log_card,
            bg=self.PANEL_SOFT,
            fg=self.TEXT,
            selectbackground=self.ACCENT,
            selectforeground=self.BG,
            relief="flat",
            highlightthickness=0,
            bd=0,
            font=("Consolas", 11),
            activestyle="none",
            height=18,
        )
        self.log_box.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        log_card.rowconfigure(1, weight=1)
        log_card.columnconfigure(0, weight=1)

    def _make_card(self, parent: tk.Widget, title: str) -> ttk.Frame:
        """Create a titled card frame used to group related sidebar sections.

        Args:
            parent: The parent widget.
            title:  Card heading text.

        Returns:
            A configured :class:`ttk.Frame` with the title label already placed.
        """
        frame = ttk.Frame(parent, style="Card.TFrame", padding=16)
        frame.columnconfigure(0, weight=1)
        ttk.Label(frame, text=title, style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")
        return frame

    def _make_button(self, parent: tk.Widget, text: str, command) -> tk.Button:
        """Create a uniformly styled accent button.

        Args:
            parent:  The parent widget.
            text:    Button label text.
            command: Callback invoked on click.

        Returns:
            A configured :class:`tk.Button`.
        """
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=self.ACCENT,
            fg=self.BG,
            activebackground="#7DD3FC",
            activeforeground=self.BG,
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            padx=16,
            pady=8,
            cursor="hand2",
            bd=0,
            highlightthickness=0,
        )

    def _toggle_human_vs_computer(self) -> None:
        """Handle the Human vs Computer checkbox toggle."""
        if self.vs_computer.get() and self.computer_vs_computer.get():
            self.computer_vs_computer.set(False)
        if not self.computer_vs_computer.get():
            self.cvc_autoplay.set(False)
        self._sync_mode_controls()
        self.mode_var.set("Human vs Computer" if self.vs_computer.get() else "Human vs Human")
        self._refresh_ui()
        self.after(120, self._maybe_play_ai_turn)

    def _toggle_computer_vs_computer(self) -> None:
        """Handle the Computer vs Computer checkbox toggle."""
        if self.computer_vs_computer.get() and self.vs_computer.get():
            self.vs_computer.set(False)
        if not self.computer_vs_computer.get():
            self.cvc_autoplay.set(False)
        self._sync_mode_controls()
        self._refresh_ui()
        self.after(120, self._maybe_play_ai_turn)

    def _toggle_cvc_autoplay(self) -> None:
        """Toggle Auto Play in Computer vs Computer mode on or off."""
        if not self.computer_vs_computer.get():
            return
        self.cvc_autoplay.set(not self.cvc_autoplay.get())
        self._refresh_ui()
        if self.cvc_autoplay.get():
            self.after(120, self._maybe_play_ai_turn)

    def _step_cvc_once(self) -> None:
        """Advance exactly one AI move when Auto Play is off ("Step" button)."""
        if not self.computer_vs_computer.get() or self.game.is_game_over():
            return
        if self.flip_animation_cells:
            return
        self.after(20, lambda: self._maybe_play_ai_turn(force_once=True))

    def _sync_mode_controls(self) -> None:
        """Show or hide AI controls based on active game mode."""
        show_hvc = self.vs_computer.get() and not self.computer_vs_computer.get()
        show_cvc = self.computer_vs_computer.get()

        if not show_cvc:
            self.cvc_autoplay.set(False)

        black_state = "readonly" if show_cvc else "disabled"
        if str(self.black_agent_combo.cget("state")) != black_state:
            self.black_agent_combo.configure(state=black_state)
        self.black_level_spin.configure(state="normal" if show_cvc else "disabled")

        white_enabled = show_hvc or show_cvc
        white_state = "readonly" if white_enabled else "disabled"
        if str(self.white_agent_combo.cget("state")) != white_state:
            self.white_agent_combo.configure(state=white_state)
        self.white_level_spin.configure(state="normal" if white_enabled else "disabled")

        if show_cvc:
            self.black_ai_frame.grid()
            self.white_ai_title_var.set("White AI Agent")
            self.white_ai_frame.grid()
            self.cvc_button_row.grid()
        elif show_hvc:
            self.black_ai_frame.grid_remove()
            self.white_ai_title_var.set("AI Agent (White)")
            self.white_ai_frame.grid()
            self.cvc_button_row.grid_remove()
        else:
            self.black_ai_frame.grid_remove()
            self.white_ai_frame.grid_remove()
            self.cvc_button_row.grid_remove()

    @staticmethod
    def _normalise_level(raw_value: int | str) -> int:
        """Clamp a user-selected skill level into the inclusive range [1, 10]."""
        try:
            level = int(raw_value)
        except (TypeError, ValueError):
            level = 5
        return max(1, min(10, level))

    @staticmethod
    def _skill_band(level: int) -> str:
        """Return a coarse skill label for a numeric level."""
        if level <= 3:
            return "Poor"
        if level <= 7:
            return "Average"
        return "Good"

    @staticmethod
    def _short_agent_name(name: str) -> str:
        if name.startswith("Rule-based"):
            return "Greedy"
        return name

    @staticmethod
    def _wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
        """Return a Wilson score confidence interval for a binomial proportion."""
        if trials <= 0:
            return 0.0, 0.0
        p_hat = successes / trials
        denominator = 1.0 + (z * z) / trials
        center = (p_hat + (z * z) / (2.0 * trials)) / denominator
        margin = z * math.sqrt((p_hat * (1.0 - p_hat) + (z * z) / (4.0 * trials)) / trials) / denominator
        return max(0.0, center - margin), min(1.0, center + margin)

    @staticmethod
    def _elo_from_score(score_rate: float) -> float:
        """Estimate Elo difference from expected score rate in [0, 1]."""
        bounded = min(max(score_rate, 1e-6), 1.0 - 1e-6)
        return -400.0 * math.log10((1.0 / bounded) - 1.0)

    def _on_ai_config_change(self, _: Optional[tk.Event] = None) -> None:
        """React to agent and skill-level config changes in the sidebar controls."""
        self.black_level_var.set(self._normalise_level(self.black_level_var.get()))
        self.white_level_var.set(self._normalise_level(self.white_level_var.get()))
        self._refresh_ui()
        if self.computer_vs_computer.get() and self.cvc_autoplay.get():
            self.after(80, self._maybe_play_ai_turn)

    def _agent_for_config(self, agent_name: str, level: int) -> OthelloAgent:
        """Build an AI instance from a selected agent type and skill level.

        Levels 1-10 are mapped to concrete search budgets / heuristics.
        """
        lvl = self._normalise_level(level)
        agent_key = agent_name.strip().lower()

        if agent_key.startswith("rule-based"):
            if lvl <= 3:
                return GreedyAgent(difficulty="easy")
            if lvl <= 7:
                return GreedyAgent(difficulty="medium")

            if lvl <= 8:
                return GreedyAgent(difficulty="hard", heuristic=SimpleHeuristic())
            if lvl == 9:
                return GreedyAgent(difficulty="hard", heuristic=PositionalHeuristic())
            return GreedyAgent(difficulty="hard", heuristic=SmartHeuristic())

        if agent_key == "minimax":
            depth = 2 + ((lvl - 1) * 5) // 9  # 2..7
            time_limit = 1.0 + (lvl - 1) * 0.6
            heuristic = (
                SimpleHeuristic() if lvl <= 3 else PositionalHeuristic() if lvl <= 7 else SmartHeuristic()
            )
            return MinimaxAgent(max_depth=depth, heuristic=heuristic, time_limit=time_limit)

        if agent_key == "mcts":
            iterations = 60 + (lvl - 1) * 40
            rollout_depth = 8 + (lvl - 1) // 2
            return MCTSAgent(iterations=iterations, dept_roll=rollout_depth)

        return GreedyAgent(difficulty="medium")

    def _run_agent_evaluation(self) -> None:
        """Run agent-vs-agent benchmarking and emit results to terminal and text file."""
        agent_a_name = self.eval_search_agent_var.get()
        agent_a_level = self._normalise_level(self.eval_search_level_var.get())
        agent_b_name = self.eval_opponent_agent_var.get()
        agent_b_level = self._normalise_level(self.eval_opponent_level_var.get())
        matches = max(1, int(self.eval_matches_var.get()))

        self.eval_progress_var.set(0.0)
        self.eval_progress_text_var.set(f"Progress: 0 / {matches}")
        self.eval_game_progress_var.set(0.0)
        self.eval_game_progress_text_var.set("Current game: 0 / 60 plies")

        agent_a = self._agent_for_config(agent_a_name, agent_a_level)
        agent_b = self._agent_for_config(agent_b_name, agent_b_level)

        agent_a_wins = 0
        agent_b_wins = 0
        draws = 0
        agent_a_move_time = 0.0
        agent_b_move_time = 0.0
        agent_a_moves = 0
        agent_b_moves = 0
        agent_a_disc_diff_total = 0

        black_wins = 0
        white_wins = 0
        agent_a_points_by_game: List[float] = []
        agent_a_points_as_black = 0.0
        agent_a_points_as_white = 0.0
        games_agent_a_black = 0
        games_agent_a_white = 0
        game_lengths: List[int] = []
        game_durations: List[float] = []

        game_lines: List[str] = []
        game_move_logs: List[str] = []

        self.configure(cursor="watch")
        self.eval_button.configure(state="disabled")
        self.update_idletasks()

        try:
            for game_index in range(matches):
                agent_a_is_black = (game_index % 2 == 0)
                black_agent = agent_a if agent_a_is_black else agent_b
                white_agent = agent_b if agent_a_is_black else agent_a
                move_index = 0
                current_game_moves: List[str] = []
                game_start = time.perf_counter()

                self.eval_game_progress_var.set(0.0)
                self.eval_game_progress_text_var.set(
                    f"Current game {game_index + 1}/{matches}: 0 / 60 plies"
                )
                self.update_idletasks()

                black_agent.on_game_start()
                white_agent.on_game_start()
                game = OthelloGame()

                while not game.is_game_over():
                    legal = game.legal_moves()
                    if not legal:
                        passing_player = game.current_player
                        opponent = game.opponent(game.current_player)
                        if game.can_player_move(opponent):
                            current_game_moves.append(
                                f"pass: {PLAYER_NAMES[passing_player]} has no legal move"
                            )
                            game.current_player = opponent
                            continue
                        break

                    current_player = game.current_player
                    active = black_agent if game.current_player == BLACK else white_agent
                    start = time.perf_counter()
                    move = active.choose_move(game)
                    elapsed = time.perf_counter() - start

                    is_agent_a_turn = (
                        (agent_a_is_black and game.current_player == BLACK)
                        or ((not agent_a_is_black) and game.current_player == WHITE)
                    )
                    if is_agent_a_turn:
                        agent_a_move_time += elapsed
                        agent_a_moves += 1
                    else:
                        agent_b_move_time += elapsed
                        agent_b_moves += 1

                    if move is None or move not in legal:
                        fallback_move = next(iter(legal.keys()))
                        move_index += 1
                        current_game_moves.append(
                            (
                                f"{move_index:02d}. {PLAYER_NAMES[current_player]:5} -> "
                                f"{game.notation_for_move(fallback_move[0], fallback_move[1])} "
                                f"(fallback)"
                            )
                        )
                        game.apply_move(fallback_move[0], fallback_move[1])
                    else:
                        move_index += 1
                        current_game_moves.append(
                            (
                                f"{move_index:02d}. {PLAYER_NAMES[current_player]:5} -> "
                                f"{game.notation_for_move(move[0], move[1])}"
                            )
                        )
                        game.apply_move(move[0], move[1])

                    self.eval_game_progress_var.set(min(100.0, (move_index / 60.0) * 100.0))
                    self.eval_game_progress_text_var.set(
                        f"Current game {game_index + 1}/{matches}: {move_index} / 60 plies"
                    )
                    self.update_idletasks()

                black_agent.on_game_end()
                white_agent.on_game_end()

                scores = game.count_discs()
                agent_a_discs = scores[BLACK] if agent_a_is_black else scores[WHITE]
                agent_b_discs = scores[WHITE] if agent_a_is_black else scores[BLACK]
                diff = agent_a_discs - agent_b_discs
                agent_a_disc_diff_total += diff

                if diff > 0:
                    agent_a_wins += 1
                    agent_a_points = 1.0
                    result = "Agent A wins"
                elif diff < 0:
                    agent_b_wins += 1
                    agent_a_points = 0.0
                    result = "Agent B wins"
                else:
                    draws += 1
                    agent_a_points = 0.5
                    result = "Draw"

                if scores[BLACK] > scores[WHITE]:
                    black_wins += 1
                elif scores[WHITE] > scores[BLACK]:
                    white_wins += 1

                agent_a_points_by_game.append(agent_a_points)
                if agent_a_is_black:
                    games_agent_a_black += 1
                    agent_a_points_as_black += agent_a_points
                else:
                    games_agent_a_white += 1
                    agent_a_points_as_white += agent_a_points

                game_lengths.append(move_index)
                game_durations.append(time.perf_counter() - game_start)

                game_lines.append(
                    (
                        f"Game {game_index + 1:03d}: Agent A as {'Black' if agent_a_is_black else 'White'} | "
                        f"Score {agent_a_discs}-{agent_b_discs} | DiscDiff {diff:+d} | {result}"
                    )
                )
                game_move_logs.append(
                    "\n".join(
                        [
                            (
                                f"Game {game_index + 1:03d} Moves "
                                f"(Agent A as {'Black' if agent_a_is_black else 'White'}):"
                            ),
                            *(current_game_moves if current_game_moves else ["(no moves recorded)"]),
                        ]
                    )
                )

                self.eval_game_progress_var.set(100.0)
                self.eval_game_progress_text_var.set(
                    f"Current game {game_index + 1}/{matches}: complete ({move_index} plies)"
                )
                self.update_idletasks()

                completed = game_index + 1
                self.eval_progress_var.set((completed / matches) * 100.0)
                self.eval_progress_text_var.set(f"Progress: {completed} / {matches}")
                self.update_idletasks()

            total_games = matches
            agent_a_losses = agent_b_wins
            agent_b_losses = agent_a_wins
            agent_a_win_rate = (agent_a_wins / total_games) * 100.0
            agent_b_win_rate = (agent_b_wins / total_games) * 100.0
            agent_a_avg_diff = agent_a_disc_diff_total / total_games
            agent_b_avg_diff = -agent_a_avg_diff
            agent_a_avg_ms = (agent_a_move_time / agent_a_moves * 1000.0) if agent_a_moves else 0.0
            agent_b_avg_ms = (agent_b_move_time / agent_b_moves * 1000.0) if agent_b_moves else 0.0

            agent_a_score_rate = (agent_a_wins + 0.5 * draws) / total_games
            if total_games > 1:
                score_std = statistics.stdev(agent_a_points_by_game)
                score_margin = 1.96 * (score_std / math.sqrt(total_games))
            else:
                score_margin = 0.0
            score_ci_low = max(0.0, agent_a_score_rate - score_margin)
            score_ci_high = min(1.0, agent_a_score_rate + score_margin)

            decisive_games = agent_a_wins + agent_b_wins
            decisive_wr = (agent_a_wins / decisive_games) if decisive_games else 0.0
            wilson_low, wilson_high = self._wilson_interval(agent_a_wins, decisive_games)

            elo_est = self._elo_from_score(agent_a_score_rate)
            elo_low = self._elo_from_score(score_ci_low)
            elo_high = self._elo_from_score(score_ci_high)

            black_win_rate = black_wins / total_games
            white_win_rate = white_wins / total_games
            agent_a_black_score = (agent_a_points_as_black / games_agent_a_black) if games_agent_a_black else 0.0
            agent_a_white_score = (agent_a_points_as_white / games_agent_a_white) if games_agent_a_white else 0.0

            avg_game_len = statistics.mean(game_lengths) if game_lengths else 0.0
            std_game_len = statistics.pstdev(game_lengths) if len(game_lengths) > 1 else 0.0
            avg_game_sec = statistics.mean(game_durations) if game_durations else 0.0
            std_game_sec = statistics.pstdev(game_durations) if len(game_durations) > 1 else 0.0

            agent_a_label = f"{self._short_agent_name(agent_a_name)} L{agent_a_level} ({self._skill_band(agent_a_level)})"
            agent_b_label = f"{self._short_agent_name(agent_b_name)} L{agent_b_level} ({self._skill_band(agent_b_level)})"

            table_header = (
                f"{'Agent':<30} {'Wins':>4} {'Losses':>6} {'Draws':>5} {'Win%':>7} "
                f"{'AvgDiscDiff':>11} {'AvgMoveMs':>10}"
            )
            table_sep = "-" * len(table_header)
            table_rows = [
                f"{agent_a_label:<30} {agent_a_wins:>4} {agent_a_losses:>6} {draws:>5} {agent_a_win_rate:>6.1f}% {agent_a_avg_diff:>11.2f} {agent_a_avg_ms:>10.2f}",
                f"{agent_b_label:<30} {agent_b_wins:>4} {agent_b_losses:>6} {draws:>5} {agent_b_win_rate:>6.1f}% {agent_b_avg_diff:>11.2f} {agent_b_avg_ms:>10.2f}",
            ]

            methods_used = [
                "Evaluation methods:",
                "- Color-balanced win rate (alternating first player)",
                "- Average disc differential",
                "- Average decision time per move (ms)",
                "- Match score rate with 95% confidence interval",
                "- Decisive-game win rate with Wilson 95% interval",
                "- Elo difference estimate from score rate",
                "- Color-split performance (as Black vs as White)",
                "- Average game length (plies) and runtime (seconds)",
            ]

            advanced_metrics = [
                "Extended Metrics:",
                (
                    f"- Agent A score rate: {agent_a_score_rate * 100.0:.1f}% "
                    f"(95% CI: {score_ci_low * 100.0:.1f}% to {score_ci_high * 100.0:.1f}%)"
                ),
                (
                    f"- Agent A decisive win rate: {decisive_wr * 100.0:.1f}% "
                    f"(Wilson 95% CI: {wilson_low * 100.0:.1f}% to {wilson_high * 100.0:.1f}%) "
                    f"over {decisive_games} decisive games"
                ),
                (
                    f"- Elo estimate (Agent A - Agent B): {elo_est:+.1f} "
                    f"(from score-rate CI: {elo_low:+.1f} to {elo_high:+.1f})"
                ),
                (
                    f"- Baseline color bias: Black win rate {black_win_rate * 100.0:.1f}%, "
                    f"White win rate {white_win_rate * 100.0:.1f}%"
                ),
                (
                    f"- Agent A split score: as Black {agent_a_black_score * 100.0:.1f}% "
                    f"vs as White {agent_a_white_score * 100.0:.1f}%"
                ),
                (
                    f"- Average game length: {avg_game_len:.1f} plies (sd {std_game_len:.1f}); "
                    f"Average game time: {avg_game_sec:.2f}s (sd {std_game_sec:.2f}s)"
                ),
            ]

            table_text = "\n".join([table_header, table_sep, *table_rows])
            report_lines = [
                "Agent vs Agent Evaluation",
                f"Agent A: {agent_a_label}",
                f"Agent B: {agent_b_label}",
                f"Total games: {total_games}",
                "",
                *methods_used,
                "",
                "Summary Table:",
                table_text,
                "",
                *advanced_metrics,
                "",
                "Per-game Results:",
                *game_lines,
                "",
                "Per-game Move Logs:",
                *game_move_logs,
            ]
            report_text = "\n".join(report_lines)

            print("\n" + table_text)
            for metric_line in advanced_metrics:
                print(metric_line)
            print()

            with open(self.eval_output_path, "w", encoding="utf-8") as report_file:
                report_file.write(report_text)

            messagebox.showinfo(
                "Evaluation Complete",
                (
                    f"Completed {total_games} games.\n"
                    f"Summary table printed to terminal.\n"
                    f"Saved report to:\n{self.eval_output_path}"
                ),
            )
            self.eval_progress_var.set(100.0)
            self.eval_progress_text_var.set(f"Progress: {total_games} / {total_games} (done)")
            self.eval_game_progress_var.set(100.0)
            self.eval_game_progress_text_var.set("Current game: done")

        except Exception as exc:  # pragma: no cover - UI error path
            self.eval_progress_text_var.set("Progress: failed")
            self.eval_game_progress_text_var.set("Current game: failed")
            messagebox.showerror("Evaluation Error", f"Failed to run evaluation: {exc}")
        finally:
            self.eval_button.configure(state="normal")
            self.configure(cursor="")
            self.update_idletasks()

    def _is_ai_turn(self) -> bool:
        """Return `True` if the AI should make the next move in the current mode."""
        if self.game.is_game_over():
            return False
        if self.computer_vs_computer.get():
            return True
        return self.vs_computer.get() and self.game.current_player == WHITE

    def _new_game(self) -> None:
        """Reset the board to the opening position and clear the move log."""
        self._stop_flip_animation()
        self.game.reset()
        self.move_counter = 0
        self.log_box.delete(0, tk.END)
        self.hovered_cell = None
        self._refresh_ui()
        if self.computer_vs_computer.get() and self.cvc_autoplay.get():
            self.after(120, self._maybe_play_ai_turn)
            return
        self.after(120, self._maybe_play_ai_turn)

    def _undo(self) -> None:
        """Undo the last move(s) and redraw the board.

        In Human vs Computer mode both the human's move and the AI response
        are undone together so the human is always left with a choice to make.
        """
        self._stop_flip_animation()
        undone = False
        if self.vs_computer.get():
            # Undo both the human and AI move when possible.
            undone = self.game.undo()
            if undone:
                self.game.undo()
                if self.log_box.size() > 0:
                    self.log_box.delete(tk.END)
                if self.log_box.size() > 0:
                    self.log_box.delete(tk.END)
        else:
            undone = self.game.undo()
            if undone and self.log_box.size() > 0:
                self.log_box.delete(tk.END)

        if not undone:
            messagebox.showinfo("Undo", "No move available to undo yet.")
        self._refresh_ui()

    def _flash_hint(self) -> None:
        """Toggle legal-move hint dots on or off."""
        self.show_hints.set(not self.show_hints.get())
        self._refresh_ui()

    def _on_canvas_click(self, event: tk.Event) -> None:
        """Handle a mouse click on the board canvas.

        Ignores clicks when it is the AI's turn, a flip animation is running,
        or the clicked cell is not a legal move.
        """
        move = self._event_to_cell(event.x, event.y)
        if move is None or self.game.is_game_over():
            return
        if self.flip_animation_cells:
            return

        if self._is_ai_turn():
            return

        legal = self.game.legal_moves()
        if move not in legal:
            return

        self._play_move(move)

    def _on_canvas_motion(self, event: tk.Event) -> None:
        """Track the hovered cell to highlight it on the canvas."""
        self.hovered_cell = self._event_to_cell(event.x, event.y)
        self._draw_board()

    def _on_canvas_leave(self, _: tk.Event) -> None:
        """Clear the hover highlight when the cursor leaves the canvas."""
        self.hovered_cell = None
        self._draw_board()

    def _event_to_cell(self, x: int, y: int) -> Optional[Coord]:
        """Convert canvas pixel coordinates to a `(row, col)` board cell.

        Args:
            x: Canvas x-coordinate in pixels.
            y: Canvas y-coordinate in pixels.

        Returns:
            `(row, col)` if the point falls inside the board, else `None`.
        """
        left = self.BOARD_PADDING
        top = self.BOARD_PADDING
        right = left + self.BOARD_PIXELS
        bottom = top + self.BOARD_PIXELS

        if not (left <= x < right and top <= y < bottom):
            return None

        col = (x - left) // self.CELL_SIZE
        row = (y - top) // self.CELL_SIZE
        return int(row), int(col)

    def _play_move(self, move: Coord) -> None:
        """Apply a move to the game, animate disc flips, and log the move.

        Triggers :meth:`_maybe_play_ai_turn` after a short delay so the
        AI can respond if it is the next player.

        Args:
            move: `(row, col)` of the square to place a disc on.
        """
        row, col = move
        outcome = self.game.apply_move(row, col)
        self.move_counter += 1

        # Animate only the captured pieces; the newly placed piece stays static.
        self._start_flip_animation(list(outcome.flipped), outcome.player)

        summary = f"{self.move_counter:02d}. {PLAYER_NAMES[outcome.player]:5} → {self.game.notation_for_move(row, col)}"
        if outcome.passed_player:
            summary += f"   (pass: {PLAYER_NAMES[outcome.passed_player]})"
        self.log_box.insert(tk.END, summary)
        self.log_box.yview_moveto(1.0)

        self._refresh_ui()

        if outcome.game_over:
            winner = self.game.winner()
            if winner is None:
                messagebox.showinfo("Game Over", "The match ends in a draw.")
            else:
                messagebox.showinfo("Game Over", f"{PLAYER_NAMES[winner]} wins the match.")
            return

        self.after(180, self._maybe_play_ai_turn)

    def _maybe_play_ai_turn(self, force_once: bool = False) -> None:
        """Trigger one AI move if the current game state calls for it.

        Called after every human move and whenever Auto Play is active in
        Computer vs Computer mode. Guards against:

        - Game already over.
        - CvC mode with Auto Play off and no manual step request.
        - A flip animation still in progress (retries after 60 ms).
        - It not being an AI turn (e.g. human move expected).

        Args:
            force_once: When `True`, bypasses the Auto Play gate so that
                        the "Step" button can advance exactly one AI move.
        """
        if self.game.is_game_over():
            return

        # In CvC mode without Auto Play, only advance when explicitly stepped.
        if self.computer_vs_computer.get() and not self.cvc_autoplay.get() and not force_once:
            return

        # Delay until any running flip animation has finished.
        if self.flip_animation_cells:
            self.after(60, self._maybe_play_ai_turn)
            return

        if not self._is_ai_turn():
            return

        # Resolve the active side's selected AI type and level.
        if self.game.current_player == BLACK:
            agent_name = self.black_agent_var.get()
            level = self._normalise_level(self.black_level_var.get())
        else:
            agent_name = self.white_agent_var.get()
            level = self._normalise_level(self.white_level_var.get())

        agent = self._agent_for_config(agent_name, level)
        move  = agent.choose_move(self.game)

        if move is None:
            self._refresh_ui()
            return

        self._play_move(move)

    def _refresh_ui(self) -> None:
        """Synchronise all sidebar labels and button states with the current game state."""
        scores = self.game.count_discs()
        self._sync_mode_controls()
        black_level = self._normalise_level(self.black_level_var.get())
        white_level = self._normalise_level(self.white_level_var.get())
        self.black_level_var.set(black_level)
        self.white_level_var.set(white_level)
        self.black_skill_var.set(f"{self._skill_band(black_level)} skill (L{black_level})")
        self.white_skill_var.set(f"{self._skill_band(white_level)} skill (L{white_level})")

        self.status_var.set(self.game.status_text())
        self.turn_var.set("")
        if self.computer_vs_computer.get():
            self.mode_var.set(
                (
                    f"CvC - B:{self._short_agent_name(self.black_agent_var.get())} L{black_level} "
                    f"vs W:{self._short_agent_name(self.white_agent_var.get())} L{white_level}"
                )
            )
        elif self.vs_computer.get():
            self.mode_var.set(
                f"HvC - {self._short_agent_name(self.white_agent_var.get())} L{white_level}"
            )
        else:
            self.mode_var.set("Human vs Human")
        self.score_var.set(
            f"Black {scores[BLACK]}  •  White {scores[WHITE]}  •  Empty {scores[EMPTY]}"
        )
        self.hint_button.configure(text="Hint: On" if self.show_hints.get() else "Hint: Off")
        if self.computer_vs_computer.get():
            self.autoplay_button.configure(
                state="normal",
                text="Pause" if self.cvc_autoplay.get() else "Auto Play",
            )
            self.step_button.configure(state="normal")
        else:
            self.autoplay_button.configure(state="disabled", text="Auto Play")
            self.step_button.configure(state="disabled")
        self._draw_board()

    def _stop_flip_animation(self) -> None:
        """Cancel any running flip animation and reset animation state."""
        if self.flip_animation_job is not None:
            self.after_cancel(self.flip_animation_job)
            self.flip_animation_job = None

        self.flip_animation_cells = []
        self.flip_animation_step = 0
        self.flip_animation_to_player = None

    def _start_flip_animation(self, cells: List[Coord], to_player: str) -> None:
        """Begin the disc-flip animation for a set of captured squares.

        Args:
            cells:     Squares whose discs are being flipped.
            to_player: The player colour the flipped discs will become.
        """
        self._stop_flip_animation()

        self.flip_animation_cells = cells
        self.flip_animation_step = 0
        self.flip_animation_to_player = to_player

        if not cells:
            self._draw_board()
            return

        self._advance_flip_animation()

    def _advance_flip_animation(self) -> None:
        """Advance one frame of the flip animation and schedule the next via `after`."""
        if not self.flip_animation_cells:
            return

        if self.flip_animation_step >= self.FLIP_ANIMATION_STEPS:
            self.flip_animation_cells = []
            self.flip_animation_step = 0
            self.flip_animation_to_player = None
            self.flip_animation_job = None
            self._draw_board()
            return

        self.flip_animation_step += 1
        self._draw_board()
        self.flip_animation_job = self.after(
            self.FLIP_ANIMATION_FRAME_MS,
            self._advance_flip_animation,
        )

    def _draw_board(self) -> None:
        """Redraw the entire board canvas from scratch."""
        c = self.canvas
        c.delete("all")

        self._draw_board_shadow()
        self._draw_board_background()
        self._draw_labels()
        self._draw_hover()
        self._draw_grid()
        self._draw_hints()
        self._draw_discs()

    def _draw_board_shadow(self) -> None:
        """Draw the drop-shadow beneath the board rectangle."""
        left = self.BOARD_PADDING
        top = self.BOARD_PADDING
        right = left + self.BOARD_PIXELS
        bottom = top + self.BOARD_PIXELS
        self.canvas.create_rectangle(
            left + 12,
            top + 12,
            right + 12,
            bottom + 12,
            fill="#020617",
            outline="",
        )

    def _draw_board_background(self) -> None:
        """Draw the green board surface and border inset."""
        left = self.BOARD_PADDING
        top = self.BOARD_PADDING
        right = left + self.BOARD_PIXELS
        bottom = top + self.BOARD_PIXELS
        self.canvas.create_rectangle(
            left,
            top,
            right,
            bottom,
            fill=self.BOARD,
            outline="#14532D",
            width=4,
        )

        self.canvas.create_rectangle(
            left + 8,
            top + 8,
            right - 8,
            bottom - 8,
            outline="#86EFAC",
            width=1,
        )

    def _draw_labels(self) -> None:
        """Draw column letters (A–H) and row numbers (1–8) around the board edge."""
        for idx in range(BOARD_SIZE):
            x = self.BOARD_PADDING + idx * self.CELL_SIZE + self.CELL_SIZE / 2
            y = self.BOARD_PADDING + idx * self.CELL_SIZE + self.CELL_SIZE / 2
            self.canvas.create_text(
                x,
                self.BOARD_PADDING - 16,
                text=chr(ord("A") + idx),
                fill=self.TEXT,
                font=("Segoe UI", 11, "bold"),
            )
            self.canvas.create_text(
                self.BOARD_PADDING - 16,
                y,
                text=str(idx + 1),
                fill=self.TEXT,
                font=("Segoe UI", 11, "bold"),
            )

    def _draw_grid(self) -> None:
        """Draw the 8×8 grid lines and classic star-point markers."""
        left = self.BOARD_PADDING
        top = self.BOARD_PADDING

        for i in range(BOARD_SIZE + 1):
            offset = i * self.CELL_SIZE
            self.canvas.create_line(
                left + offset,
                top,
                left + offset,
                top + self.BOARD_PIXELS,
                fill=self.GRID,
                width=1,
            )
            self.canvas.create_line(
                left,
                top + offset,
                left + self.BOARD_PIXELS,
                top + offset,
                fill=self.GRID,
                width=1,
            )

        star_points = [(2, 2), (2, 7), (7, 2), (7, 7)]
        for row, col in star_points:
            cx, cy = self._cell_center(row - 1, col - 1)
            # Stacked circles emulate low opacity in Tkinter without RGBA support.
            self.canvas.create_oval(
                cx - 5,
                cy - 5,
                cx + 5,
                cy + 5,
                fill=self.STAR_POINT,
                outline="",
                stipple="gray25",
            )
            self.canvas.create_oval(
                cx - 2,
                cy - 2,
                cx + 2,
                cy + 2,
                fill=self.STAR_POINT,
                outline="",
                stipple="gray12",
            )

    def _draw_hover(self) -> None:
        """Highlight the cell under the cursor (yellow if legal, dim if not)."""
        if self.hovered_cell is None:
            return
        row, col = self.hovered_cell
        if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
            return

        x0, y0, x1, y1 = self._cell_bbox(row, col, padding=3)
        legal = self.game.legal_moves()
        outline = self.HOVER if (row, col) in legal else self.PANEL_SOFT
        self.canvas.create_rectangle(x0, y0, x1, y1, outline=outline, width=3)

    def _draw_hints(self) -> None:
        """Draw small green dots on every legal move square when hints are on."""
        if not self.show_hints.get() or self.game.is_game_over():
            return

        for row, col in self.game.legal_moves().keys():
            cx, cy = self._cell_center(row, col)
            self.canvas.create_oval(
                cx - 8,
                cy - 8,
                cx + 8,
                cy + 8,
                fill=self.HINT,
                outline="",
            )

    def _draw_discs(self) -> None:
        """Draw all discs, routing animating cells through the flip-transition renderer."""
        progress = 1.0
        if self.flip_animation_cells and self.FLIP_ANIMATION_STEPS > 1:
            progress = (self.flip_animation_step - 1) / (self.FLIP_ANIMATION_STEPS - 1)

        animating = set(self.flip_animation_cells)

        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = self.game.get_cell(row, col)
                if piece == EMPTY:
                    continue

                if (row, col) in animating and self.flip_animation_to_player is not None:
                    self._draw_flip_transition_disc(row, col, progress)
                    continue

                self._draw_standard_disc(row, col, piece)

    def _draw_standard_disc(self, row: int, col: int, piece: str) -> None:
        """Draw a static (non-animating) disc with a shadow and shine highlight.

        Args:
            row:   Board row index (0-based).
            col:   Board column index (0-based).
            piece: The player colour constant occupying this cell.
        """
        x0, y0, x1, y1 = self._cell_bbox(row, col, padding=10)
        self.canvas.create_oval(x0 + 4, y0 + 5, x1 + 4, y1 + 5, fill="#000000", outline="")
        self.canvas.create_oval(
            x0,
            y0,
            x1,
            y1,
            fill=PLAYER_COLORS[piece],
            outline="#CBD5E1" if piece == WHITE else "#020617",
            width=2,
        )

        if piece == WHITE:
            self.canvas.create_oval(
                x0 + 8,
                y0 + 8,
                x0 + 22,
                y0 + 22,
                fill="#FFFFFF",
                outline="",
            )
        else:
            self.canvas.create_oval(
                x0 + 8,
                y0 + 8,
                x0 + 22,
                y0 + 22,
                fill="#334155",
                outline="",
            )

    def _draw_flip_transition_disc(self, row: int, col: int, progress: float) -> None:
        """Draw a disc mid-flip using a horizontal-squeeze illusion.

        The horizontal radius is scaled from full → 0 → full across the
        animation, crossing through 0 at `progress=0.5` where the disc
        colour switches from the old player to the new player.

        Args:
            row:      Board row index (0-based).
            col:      Board column index (0-based).
            progress: Animation progress in `[0.0, 1.0]`.
        """
        to_piece = self.flip_animation_to_player
        if to_piece is None:
            self._draw_standard_disc(row, col, self.game.get_cell(row, col))
            return

        from_piece = WHITE if to_piece == BLACK else BLACK
        current_piece = from_piece if progress < 0.5 else to_piece
        # Squeeze width to zero then expand to emulate a 3D flip.
        width_scale = (1.0 - progress * 2.0) if progress < 0.5 else (progress - 0.5) * 2.0
        width_scale = max(0.1, width_scale)

        cx, cy = self._cell_center(row, col)
        x0, y0, x1, y1 = self._cell_bbox(row, col, padding=10)
        base_rx = (x1 - x0) / 2
        ry = (y1 - y0) / 2
        rx = base_rx * width_scale

        glow_radius = int(12 + 5 * (1.0 - abs(2.0 * progress - 1.0)))
        glow_color = "#FDE68A" if to_piece == BLACK else "#93C5FD"
        self.canvas.create_oval(
            cx - glow_radius,
            cy - glow_radius,
            cx + glow_radius,
            cy + glow_radius,
            fill=glow_color,
            outline="",
            stipple="gray50",
        )

        self.canvas.create_oval(
            cx - rx + 4,
            cy - ry + 5,
            cx + rx + 4,
            cy + ry + 5,
            fill="#000000",
            outline="",
        )
        self.canvas.create_oval(
            cx - rx,
            cy - ry,
            cx + rx,
            cy + ry,
            fill=PLAYER_COLORS[current_piece],
            outline="#CBD5E1" if current_piece == WHITE else "#020617",
            width=2,
        )

        shine_color = "#FFFFFF" if current_piece == WHITE else "#334155"
        shine_rx = max(2.0, 7.0 * width_scale)
        self.canvas.create_oval(
            cx - rx + 8,
            cy - ry + 8,
            cx - rx + 8 + shine_rx,
            cy - ry + 20,
            fill=shine_color,
            outline="",
        )

    def _cell_center(self, row: int, col: int) -> tuple[float, float]:
        """Return the canvas pixel coordinates of the centre of a board cell.

        Args:
            row: Board row index (0-based).
            col: Board column index (0-based).

        Returns:
            `(x, y)` canvas pixel coordinates.
        """
        x = self.BOARD_PADDING + col * self.CELL_SIZE + self.CELL_SIZE / 2
        y = self.BOARD_PADDING + row * self.CELL_SIZE + self.CELL_SIZE / 2
        return x, y

    def _cell_bbox(self, row: int, col: int, padding: int = 0) -> tuple[int, int, int, int]:
        """Return the bounding-box pixel coordinates for a board cell.

        Args:
            row:     Board row index (0-based).
            col:     Board column index (0-based).
            padding: Inset in pixels to shrink the bounding box on each side.

        Returns:
            `(x0, y0, x1, y1)` canvas pixel coordinates.
        """
        x0 = self.BOARD_PADDING + col * self.CELL_SIZE + padding
        y0 = self.BOARD_PADDING + row * self.CELL_SIZE + padding
        x1 = self.BOARD_PADDING + (col + 1) * self.CELL_SIZE - padding
        y1 = self.BOARD_PADDING + (row + 1) * self.CELL_SIZE - padding
        return x0, y0, x1, y1


def launch_app() -> None:
    """Create and run the Othello GUI application.

    Blocks until the window is closed.
    """
    app = OthelloUI()
    app.mainloop()
