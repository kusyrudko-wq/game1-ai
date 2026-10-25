"""Independent evaluation. The exact solver is NEVER used by the learner.

Benchmarks include random and optimal opponents, an exhaustive best response
from the empty board, and a stronger audit of every reachable decision state.
"""
from collections.abc import Callable
from functools import cache
import random

from .agent import QAgent
from .game import Position, reachable_positions


@cache
def minimax_value(position: Position) -> int:
    """Exact win/draw/loss value from the CURRENT player's perspective."""
    result = position.result()
    if result is not None:
        return result * position.player
    return max(-minimax_value(position.play(m)) for m in position.legal_moves())


@cache
def optimal_moves(position: Position) -> tuple[int, ...]:
    moves = position.legal_moves()
    if not moves:
        return ()
    best = minimax_value(position)
    return tuple(m for m in moves if -minimax_value(position.play(m)) == best)


def benchmark(agent: QAgent, opponent: str = "random", games: int = 1000,
              seed: int = 7) -> dict:
    """Frozen deterministic learner; randomized opponent. Starts alternate."""
    if games <= 0 or opponent not in ("random", "minimax"):
        raise ValueError("Use positive games and opponent 'random' or 'minimax'.")
    rng = random.Random(seed)
    rows = {"X": {"wins": 0, "draws": 0, "losses": 0},
            "O": {"wins": 0, "draws": 0, "losses": 0}}
    for game in range(games):
        agent_player = 1 if game % 2 == 0 else -1
        position = Position()
        while position.result() is None:
            if position.player == agent_player:
                move = agent.choose(position)  # no exploration or updates
            else:
                moves = (position.legal_moves() if opponent == "random"
                         else optimal_moves(position))
                move = rng.choice(moves)
            position = position.play(move)
        score = position.result() * agent_player
        label = "wins" if score == 1 else "losses" if score == -1 else "draws"
        rows["X" if agent_player == 1 else "O"][label] += 1
    totals = {key: sum(row[key] for row in rows.values())
              for key in ("wins", "draws", "losses")}
    return {"opponent": opponent, "games": games, "seed": seed,
            "policy": "deterministic greedy; no learning", "by_side": rows,
            **totals,
            "score_rate": (totals["wins"] + 0.5 * totals["draws"]) / games}


def worst_case_policy(choose: Callable[[Position], int], agent_player: int) -> dict:
    """Explore EVERY opponent reply against a fixed deterministic policy.

    Values are from the agent's perspective, so the opponent minimizes them.
    This verifies play FROM THE EMPTY BOARD, not from all arbitrary positions.
    The guarantee assumes this game's rules and this evaluator are correct.
    """
    if agent_player not in (-1, 1):
        raise ValueError("agent_player must be X (1) or O (-1).")

    @cache
    def visit(position: Position) -> int:
        result = position.result()
        if result is not None:
            return result * agent_player
        if position.player == agent_player:
            return visit(position.play(choose(position)))
        return min(visit(position.play(m)) for m in position.legal_moves())

    worst = visit(Position())
    # Return one concrete worst-case line, useful when a model is exploitable.
    line: list[int] = []
    position = Position()
    while position.result() is None:
        if position.player == agent_player:
            move = choose(position)
        else:
            move = min(position.legal_moves(), key=lambda m: visit(position.play(m)))
        line.append(move + 1)  # human-readable squares 1..9
        position = position.play(move)
    return {"agent_side": "X" if agent_player == 1 else "O",
            "worst_case_score": worst, "cannot_lose_from_empty": worst >= 0,
            "positions_checked": visit.cache_info().currsize,
            "example_worst_case_line_1_to_9": line}


def audit_all_positions(agent: QAgent) -> dict:
    """Check whether the chosen action preserves minimax value at every state."""
    positions = sorted(reachable_positions(), key=lambda p: (p.board, p.player))
    decision_states = [p for p in positions if p.result() is None]
    mistakes = []
    for position in decision_states:
        move = agent.choose(position)
        value = minimax_value(position)
        chosen_value = -minimax_value(position.play(move))
        if chosen_value < value:
            mistakes.append({"board": list(position.board), "player": position.player,
                             "move_1_to_9": move + 1, "optimal_value": value,
                             "chosen_move_value": chosen_value})
    return {"reachable_positions": len(positions),
            "decision_states": len(decision_states),
            "suboptimal_decisions": len(mistakes),
            "optimal_at_every_reachable_state": not mistakes,
            "first_10_mistakes": mistakes[:10]}


def evaluate(agent: QAgent, games: int = 1000, seed: int = 7) -> dict:
    return {"model_metadata": agent.metadata, "states_in_table": len(agent.table),
            "random": benchmark(agent, "random", games, seed),
            "minimax": benchmark(agent, "minimax", games, seed),
            "exhaustive_from_empty": [worst_case_policy(agent.choose, 1),
                                      worst_case_policy(agent.choose, -1)],
            "all_positions_audit": audit_all_positions(agent)}
