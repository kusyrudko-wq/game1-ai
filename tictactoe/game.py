"""Game rules only: no learning and no strategy.

Board indices run from 0 to 8, row by row. X is 1, O is -1, empty is 0.
Create positions with Position() and play(); do not construct arbitrary boards.
"""
from dataclasses import dataclass
from functools import cache

WINNING_LINES = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
    (0, 4, 8), (2, 4, 6),             # diagonals
)


@dataclass(frozen=True)
class Position:
    board: tuple[int, ...] = (0,) * 9
    player: int = 1  # the player whose turn comes next

    @cache
    def result(self) -> int | None:
        """Return 1 for X's win, -1 for O's win, 0 for draw, None if ongoing."""
        for a, b, c in WINNING_LINES:
            if self.board[a] != 0 and self.board[a] == self.board[b] == self.board[c]:
                return self.board[a]
        return 0 if 0 not in self.board else None

    @cache
    def legal_moves(self) -> tuple[int, ...]:
        if self.result() is not None:
            return ()
        return tuple(i for i, cell in enumerate(self.board) if cell == 0)

    @cache
    def play(self, move: int) -> "Position":
        """Return a new position; never modify the old one."""
        if type(move) is not int or move not in self.legal_moves():
            raise ValueError(f"Illegal move: {move}. Legal moves: {self.legal_moves()}")
        board = list(self.board)
        board[move] = self.player
        return Position(tuple(board), -self.player)

    @cache
    def key(self) -> str:
        """Encode the board from the current player's perspective.

        0 = empty; 1 = my piece; 2 = the other player's piece.
        This lets the same table play both X and O. No rotation symmetry yet.
        """
        return "".join("0" if x == 0 else "1" if x == self.player else "2"
                       for x in self.board)

    def render(self) -> str:
        """Show move numbers 1..9 in empty cells for a human player."""
        cells = ["X" if x == 1 else "O" if x == -1 else str(i + 1)
                 for i, x in enumerate(self.board)]
        rows = [" | ".join(cells[i:i + 3]) for i in (0, 3, 6)]
        return "\n---------\n".join(rows)


def reachable_positions() -> set[Position]:
    """Enumerate legal positions, stopping immediately when a game ends."""
    seen: set[Position] = set()
    todo = [Position()]
    while todo:
        position = todo.pop()
        if position in seen:
            continue
        seen.add(position)
        todo.extend(position.play(move) for move in position.legal_moves())
    return seen
