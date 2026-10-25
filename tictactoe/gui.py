"""Tkinter desktop interface for the trained tic-tac-toe agent."""
from pathlib import Path
import tkinter as tk
from tkinter import messagebox
from .agent import QAgent
from .game import Position

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / 'models' / 'demo.json'


def launch(model_path: Path = DEFAULT_MODEL) -> None:
    agent = QAgent.load(model_path)
    root = tk.Tk()
    root.title('Game AI | Tic-Tac-Toe')
    root.resizable(False, False)
    position = Position()
    human = 1
    buttons: list[tk.Button] = []
    status = tk.StringVar(value='Your turn (X)')

    def draw():
        for i, button in enumerate(buttons):
            mark = position.board[i]
            button.configure(text='X' if mark == 1 else 'O' if mark == -1 else '',
                             state=tk.NORMAL if position.result() is None and
                             position.player == human and mark == 0 else tk.DISABLED,
                             disabledforeground='#152d4d' if mark == 1 else '#a34535')
        outcome = position.result()
        if outcome is not None:
            status.set('Draw!' if outcome == 0 else 'You win!' if outcome == human else 'AI wins!')
        else:
            status.set('Your turn (X)' if position.player == human else 'AI thinking (O)')

    def ai_move():
        nonlocal position
        if position.result() is None and position.player != human:
            position = position.play(agent.choose(position))
        draw()

    def human_move(index: int):
        nonlocal position
        if position.result() is not None or position.player != human or index not in position.legal_moves():
            return
        position = position.play(index)
        draw()
        if position.result() is None:
            root.after(160, ai_move)

    def reset():
        nonlocal position
        position = Position()
        draw()
        if human == -1:
            root.after(160, ai_move)

    def set_side(side: int):
        nonlocal human
        human = side
        reset()

    tk.Label(root, text='TIC-TAC-TOE · SELF-PLAY AI', font=('Segoe UI', 15, 'bold'), padx=15, pady=14).grid(row=0, column=0, columnspan=3)
    for i in range(9):
        button = tk.Button(root, text='', width=5, height=2, font=('Segoe UI', 23, 'bold'),
                           command=lambda i=i: human_move(i))
        button.grid(row=1 + i // 3, column=i % 3, padx=4, pady=4)
        buttons.append(button)
    tk.Label(root, textvariable=status, font=('Segoe UI', 12), pady=12).grid(row=4, column=0, columnspan=3)
    frame = tk.Frame(root)
    frame.grid(row=5, column=0, columnspan=3, pady=(0, 12))
    tk.Button(frame, text='Play as X', command=lambda: set_side(1)).pack(side='left', padx=4)
    tk.Button(frame, text='Play as O', command=lambda: set_side(-1)).pack(side='left', padx=4)
    tk.Button(frame, text='Restart', command=reset).pack(side='left', padx=4)
    draw()
    root.mainloop()


if __name__ == '__main__':
    try:
        launch()
    except (FileNotFoundError, ValueError) as exc:
        messagebox.showerror('Game AI', str(exc))
