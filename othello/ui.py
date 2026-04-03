from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional, Tuple, List

from .constants import BLACK, BOARD_SIZE, EMPTY, PLAYER_COLORS, PLAYER_NAMES, WHITE
from .engine import OthelloGame
from .greedy_ai import GreedyAI
from search.minimax import get_best_move
from tests.test_minimax import simple_heuristic
Coord = Tuple[int, int]


class OthelloUI(tk.Tk):
    CELL_SIZE = 72
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
        super().__init__()
        self.title("Othello — Python GUI")
        self.configure(bg=self.BG)
        self.geometry("1280x760")
        self.minsize(1180, 720)

        self.game = OthelloGame()
        self.ai = GreedyAI()
        self.vs_computer = tk.BooleanVar(value=False)
        self.computer_vs_computer = tk.BooleanVar(value=False)
        self.cvc_autoplay = tk.BooleanVar(value=False)
        self.white_difficulty_var = tk.StringVar(value="Medium")
        self.black_difficulty_var = tk.StringVar(value="Medium")
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

        self.black_difficulty_label = ttk.Label(
            controls_card,
            text="Black AI Difficulty",
            style="CardValue.TLabel",
        )
        self.black_difficulty_label.grid(row=3, column=0, sticky="w", pady=(2, 6))
        self.black_difficulty_combo = ttk.Combobox(
            controls_card,
            state="readonly",
            values=("Easy", "Medium", "Hard"),
            textvariable=self.black_difficulty_var,
            width=16,
        )
        self.black_difficulty_combo.grid(row=4, column=0, sticky="w", pady=(0, 8))
        self.black_difficulty_combo.bind("<<ComboboxSelected>>", self._on_difficulty_change)
        self.black_difficulty_combo.configure(state="disabled")

        self.white_difficulty_label = ttk.Label(
            controls_card,
            text="White AI Difficulty",
            style="CardValue.TLabel",
        )
        self.white_difficulty_label.grid(row=5, column=0, sticky="w", pady=(2, 6))
        self.white_difficulty_combo = ttk.Combobox(
            controls_card,
            state="readonly",
            values=("Easy", "Medium", "Hard"),
            textvariable=self.white_difficulty_var,
            width=16,
        )
        self.white_difficulty_combo.grid(row=6, column=0, sticky="w", pady=(0, 8))
        self.white_difficulty_combo.bind("<<ComboboxSelected>>", self._on_difficulty_change)
        self.white_difficulty_combo.configure(state="disabled")

        button_row = tk.Frame(controls_card, bg=self.PANEL_ALT)
        button_row.grid(row=7, column=0, sticky="ew")

        self._make_button(button_row, "New Game", self._new_game).grid(row=0, column=0, padx=(0, 8), pady=4)
        self._make_button(button_row, "Undo", self._undo).grid(row=0, column=1, padx=(0, 8), pady=4)
        self.hint_button = self._make_button(button_row, "Hint: Off", self._flash_hint)
        self.hint_button.configure(width=10)
        self.hint_button.grid(row=0, column=2, pady=4)

        self.cvc_button_row = tk.Frame(controls_card, bg=self.PANEL_ALT)
        self.cvc_button_row.grid(row=8, column=0, sticky="w", pady=(4, 0))

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

        log_card = self._make_card(sidebar, "Move Log")
        log_card.grid(row=5, column=0, sticky="nsew")
        sidebar.rowconfigure(5, weight=1)

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
        frame = ttk.Frame(parent, style="Card.TFrame", padding=16)
        frame.columnconfigure(0, weight=1)
        ttk.Label(frame, text=title, style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")
        return frame

    def _make_button(self, parent: tk.Widget, text: str, command) -> tk.Button:
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
        if self.vs_computer.get() and self.computer_vs_computer.get():
            self.computer_vs_computer.set(False)
        if not self.computer_vs_computer.get():
            self.cvc_autoplay.set(False)
        self._sync_mode_controls()
        self.mode_var.set("Human vs Computer" if self.vs_computer.get() else "Human vs Human")
        self._refresh_ui()
        self.after(120, self._maybe_play_ai_turn)

    def _toggle_computer_vs_computer(self) -> None:
        if self.computer_vs_computer.get() and self.vs_computer.get():
            self.vs_computer.set(False)
        if not self.computer_vs_computer.get():
            self.cvc_autoplay.set(False)
        self._sync_mode_controls()
        self._refresh_ui()
        self.after(120, self._maybe_play_ai_turn)

    def _toggle_cvc_autoplay(self) -> None:
        if not self.computer_vs_computer.get():
            return
        self.cvc_autoplay.set(not self.cvc_autoplay.get())
        self._refresh_ui()
        if self.cvc_autoplay.get():
            self.after(120, self._maybe_play_ai_turn)

    def _step_cvc_once(self) -> None:
        if not self.computer_vs_computer.get() or self.game.is_game_over():
            return
        if self.flip_animation_cells:
            return
        self.after(20, lambda: self._maybe_play_ai_turn(force_once=True))

    def _sync_mode_controls(self) -> None:
        show_hvc = self.vs_computer.get() and not self.computer_vs_computer.get()
        show_cvc = self.computer_vs_computer.get()

        if not show_cvc:
            self.cvc_autoplay.set(False)

        black_state = "readonly" if show_cvc else "disabled"
        if str(self.black_difficulty_combo.cget("state")) != black_state:
            self.black_difficulty_combo.configure(state=black_state)

        white_enabled = show_hvc or show_cvc
        white_state = "readonly" if white_enabled else "disabled"
        if str(self.white_difficulty_combo.cget("state")) != white_state:
            self.white_difficulty_combo.configure(state=white_state)

        if show_cvc:
            self.black_difficulty_label.grid()
            self.black_difficulty_combo.grid()
            self.white_difficulty_label.configure(text="White AI Difficulty")
            self.white_difficulty_label.grid()
            self.white_difficulty_combo.grid()
            self.cvc_button_row.grid()
        elif show_hvc:
            self.black_difficulty_label.grid_remove()
            self.black_difficulty_combo.grid_remove()
            self.white_difficulty_label.configure(text="AI Difficulty")
            self.white_difficulty_label.grid()
            self.white_difficulty_combo.grid()
            self.cvc_button_row.grid_remove()
        else:
            self.black_difficulty_label.grid_remove()
            self.black_difficulty_combo.grid_remove()
            self.white_difficulty_label.grid_remove()
            self.white_difficulty_combo.grid_remove()
            self.cvc_button_row.grid_remove()

    def _is_ai_turn(self) -> bool:
        if self.game.is_game_over():
            return False
        if self.computer_vs_computer.get():
            return True
        return self.vs_computer.get() and self.game.current_player == WHITE

    def _on_difficulty_change(self, _: tk.Event) -> None:
        self._refresh_ui()
        if self.computer_vs_computer.get() and self.cvc_autoplay.get():
            self.after(80, self._maybe_play_ai_turn)

    def _new_game(self) -> None:
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
        self.show_hints.set(not self.show_hints.get())
        self._refresh_ui()

    def _on_canvas_click(self, event: tk.Event) -> None:
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
        self.hovered_cell = self._event_to_cell(event.x, event.y)
        self._draw_board()

    def _on_canvas_leave(self, _: tk.Event) -> None:
        self.hovered_cell = None
        self._draw_board()

    def _event_to_cell(self, x: int, y: int) -> Optional[Coord]:
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
        if self.game.is_game_over():
            return
        if self.computer_vs_computer.get() and not self.cvc_autoplay.get() and not force_once:
            return
        if self.flip_animation_cells:
            self.after(60, self._maybe_play_ai_turn)
            return
        if not self._is_ai_turn():
            return

        difficulty = (
            self.black_difficulty_var.get().lower()
            if self.game.current_player == BLACK
            else self.white_difficulty_var.get().lower()
        )
        move = self.ai.choose_move(self.game, difficulty=difficulty)
        #move, score = get_best_move(self.game, depth=2, heuristic_func=simple_heuristic)
        if move is None:
            self._refresh_ui()
            return
        self._play_move(move)

    def _refresh_ui(self) -> None:
        scores = self.game.count_discs()
        self._sync_mode_controls()
        self.status_var.set(self.game.status_text())
        self.turn_var.set("")
        if self.computer_vs_computer.get():
            self.mode_var.set(
                f"Computer vs Computer - Black {self.black_difficulty_var.get()} / White {self.white_difficulty_var.get()}"
            )
        elif self.vs_computer.get():
            self.mode_var.set(f"Human vs Computer - AI {self.white_difficulty_var.get()}")
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
        if self.flip_animation_job is not None:
            self.after_cancel(self.flip_animation_job)
            self.flip_animation_job = None

        self.flip_animation_cells = []
        self.flip_animation_step = 0
        self.flip_animation_to_player = None

    def _start_flip_animation(self, cells: List[Coord], to_player: str) -> None:
        self._stop_flip_animation()

        self.flip_animation_cells = cells
        self.flip_animation_step = 0
        self.flip_animation_to_player = to_player

        if not cells:
            self._draw_board()
            return

        self._advance_flip_animation()

    def _advance_flip_animation(self) -> None:
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
        x = self.BOARD_PADDING + col * self.CELL_SIZE + self.CELL_SIZE / 2
        y = self.BOARD_PADDING + row * self.CELL_SIZE + self.CELL_SIZE / 2
        return x, y

    def _cell_bbox(self, row: int, col: int, padding: int = 0) -> tuple[int, int, int, int]:
        x0 = self.BOARD_PADDING + col * self.CELL_SIZE + padding
        y0 = self.BOARD_PADDING + row * self.CELL_SIZE + padding
        x1 = self.BOARD_PADDING + (col + 1) * self.CELL_SIZE - padding
        y1 = self.BOARD_PADDING + (row + 1) * self.CELL_SIZE - padding
        return x0, y0, x1, y1


def launch_app() -> None:
    app = OthelloUI()
    app.mainloop()
