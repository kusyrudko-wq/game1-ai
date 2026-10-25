"""Self-play training from an empty table. No expert moves or minimax labels."""
import random
from collections.abc import Callable

from .agent import QAgent
from .game import Position


def train(episodes: int = 200_000, seed: int = 42, alpha: float = 0.2,
          epsilon_start: float = 1.0, epsilon_end: float = 0.1,
          report_every: int = 10_000,
          callback: Callable[[int, QAgent, dict], None] | None = None
          ) -> tuple[QAgent, list[dict]]:
    """Both sides share a Q table and explore legal moves.

    epsilon decreases linearly over the first 80% of games. One episode is one
    whole game. Every call starts a NEW model; there is no implicit resume.
    """
    if episodes <= 0 or report_every <= 0:
        raise ValueError("episodes and report_every must be positive.")
    if not 0 < alpha <= 1:
        raise ValueError("Require 0 < alpha <= 1.")
    if not 0 <= epsilon_end <= epsilon_start <= 1:
        raise ValueError("Require 0 <= epsilon_end <= epsilon_start <= 1.")
    rng = random.Random(seed)
    agent = QAgent(metadata={
        "episodes": episodes, "seed": seed, "alpha": alpha, "gamma": 1.0,
        "epsilon_start": epsilon_start, "epsilon_end": epsilon_end,
        "epsilon_decay_fraction": 0.8,
        "training": "self-play from an empty Q table; no expert labels",
        "evaluation_tie_break": "lowest legal index among exact Q maxima",
    })
    history: list[dict] = []
    counts = {1: 0, -1: 0, 0: 0}
    for episode in range(1, episodes + 1):
        progress = min(1.0, (episode - 1) / max(1.0, 0.8 * episodes - 1))
        epsilon = epsilon_start + progress * (epsilon_end - epsilon_start)
        position = Position()
        while position.result() is None:
            move = agent.choose(position, rng=rng, epsilon=epsilon)
            next_position = position.play(move)
            agent.update(position, move, next_position, alpha=alpha)
            position = next_position
        counts[position.result()] += 1
        if episode % report_every == 0 or episode == episodes:
            row = {"episode": episode, "epsilon": round(epsilon, 6),
                   "states": len(agent.table), "x_wins": counts[1],
                   "o_wins": counts[-1], "draws": counts[0]}
            history.append(row)
            if callback is not None:
                callback(episode, agent, row)
            counts = {1: 0, -1: 0, 0: 0}
    return agent, history
