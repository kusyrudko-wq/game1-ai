"""A trainable table of state-action values, not a neural network.

The playing policy uses table lookups only: it never calls minimax.
"""
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import random

from .game import Position


@dataclass
class QAgent:
    table: dict[str, list[float]] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def values(self, position: Position) -> list[float]:
        """Reading an unseen state does NOT change the model."""
        return self.table.get(position.key(), [0.0] * 9)

    def choose(self, position: Position, rng: random.Random | None = None,
               epsilon: float = 0.0) -> int:
        """Explore during training; otherwise choose the first best legal move.

        Training randomly breaks ties. Evaluation and human play use a fixed
        tie-break (lowest index) so the exhaustive certificate has a clear scope.
        """
        moves = position.legal_moves()
        if not moves:
            raise ValueError("Cannot choose a move in a finished game.")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between 0 and 1.")
        if epsilon > 0.0 and rng is None:
            raise ValueError("Exploration requires an explicit random generator.")
        if rng is not None and rng.random() < epsilon:
            return rng.choice(moves)
        values = self.values(position)
        best = max(values[m] for m in moves)
        ties = [m for m in moves if values[m] == best]
        return rng.choice(ties) if rng is not None else ties[0]

    def update(self, position: Position, move: int, next_position: Position,
               alpha: float = 0.2, gamma: float = 1.0) -> None:
        """One zero-sum, alternating-turn Q-learning update.

        The next position belongs to the OPPONENT. Their good outcome is bad
        for us, hence the minus sign in -gamma * max(next_values).
        """
        if not 0 < alpha <= 1 or not 0 <= gamma <= 1:
            raise ValueError("Require 0 < alpha <= 1 and 0 <= gamma <= 1.")
        if position.play(move) != next_position:
            raise ValueError("next_position does not match the supplied move.")
        result = next_position.result()
        if result is not None:
            target = float(result * position.player)  # win +1; draw 0
        else:
            next_values = self.values(next_position)
            target = -gamma * max(next_values[m] for m in next_position.legal_moves())
        values = self.table.setdefault(position.key(), [0.0] * 9)
        values[move] += alpha * (target - values[move])

    def save(self, path: Path) -> None:
        """JSON is readable and does not execute code when loaded."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format_version": 1, "algorithm": "zero-sum-tabular-q-learning",
                   "metadata": self.metadata, "q_values": self.table}
        path.write_text(json.dumps(payload, indent=2, sort_keys=True,
                                   allow_nan=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "QAgent":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("format_version") != 1:
            raise ValueError("Unsupported model format.")
        if payload.get("algorithm") != "zero-sum-tabular-q-learning":
            raise ValueError("Unsupported learning algorithm.")
        table = payload.get("q_values")
        metadata = payload.get("metadata", {})
        if not isinstance(table, dict) or not isinstance(metadata, dict):
            raise ValueError("Invalid model table or metadata.")
        for key, values in table.items():
            if (not isinstance(key, str) or len(key) != 9 or
                    any(c not in "012" for c in key)):
                raise ValueError("Invalid board key in model.")
            if (not isinstance(values, list) or len(values) != 9 or
                    any(type(x) not in (int, float) or not math.isfinite(x)
                        for x in values)):
                raise ValueError("Each state must have 9 finite numeric values.")
        return cls(table=table, metadata=metadata)
