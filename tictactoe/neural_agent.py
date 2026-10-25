"""Optional PyTorch agent. Training is provided separately by dqn_training."""
import torch
from torch import nn

from .game import Position


def encode_board(position: Position) -> torch.Tensor:
    """Return nine float32 values: own = 1, opponent = -1, empty = 0."""
    return torch.tensor(
        [cell * position.player for cell in position.board],
        dtype=torch.float32,
    )


class QNetwork(nn.Module):
    """Predict one Q-value per square, for a single board or a batch."""

    def __init__(self, width: int = 64) -> None:
        super().__init__()
        if type(width) is not int or width <= 0:
            raise ValueError('width must be a positive integer.')
        self.width = width
        self.layers = nn.Sequential(
            nn.Linear(9, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, 9),
        )

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        return self.layers(board)


class NeuralAgent:
    """Choose legal moves using a network, without exploration or learning.

    The default network is untrained: its predictions do not imply playing
    strength. Supply a QNetwork to use explicitly configured weights.
    """

    def __init__(self, network: QNetwork | None = None) -> None:
        self.network = network if network is not None else QNetwork()
        self.network.eval()

    def choose(self, position: Position) -> int:
        """Return the highest-valued legal move; ties use the lowest index."""
        moves = position.legal_moves()
        if not moves:
            raise ValueError("Cannot choose a move in a finished game.")
        parameter = next(self.network.parameters())
        board = encode_board(position).to(device=parameter.device,
                                          dtype=parameter.dtype)
        self.network.eval()
        with torch.inference_mode():
            values = self.network(board)
            # Index only legal squares, even when all their values are negative.
            best = values[list(moves)].argmax().item()
        return moves[best]


