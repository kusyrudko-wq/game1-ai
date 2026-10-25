"""Reproduce the demo and its learning curve without replacing bundled files.

Run from the project root: python experiments.py
All outputs go to runs/reproduction/. Evaluation never changes the model.
"""
import csv
import hashlib
import json
from pathlib import Path
import platform

from tictactoe.agent import QAgent
from tictactoe.evaluation import audit_all_positions, benchmark, evaluate
from tictactoe.training import train

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "runs" / "reproduction"


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit("runs/reproduction already exists. Move/rename it before reproducing again.")
    OUTPUT.mkdir(parents=True)
    curve = []

    def measure(episode: int, agent: QAgent, _row: dict) -> None:
        random_result = benchmark(agent, "random", games=1000, seed=7)
        exact_result = benchmark(agent, "minimax", games=1000, seed=7)
        audit = audit_all_positions(agent)
        row = {"episode": episode,
               "random_wins": random_result["wins"],
               "random_draws": random_result["draws"],
               "random_losses": random_result["losses"],
               "minimax_wins": exact_result["wins"],
               "minimax_draws": exact_result["draws"],
               "minimax_losses": exact_result["losses"],
               "suboptimal_decisions": audit["suboptimal_decisions"]}
        curve.append(row)
        print(row, flush=True)

    measure(0, QAgent(), {})
    agent, history = train(episodes=200_000, seed=42, report_every=10_000, callback=measure)
    agent.save(OUTPUT / "demo.json")
    for name, rows in (("learning-curve.csv", curve), ("selfplay-training.csv", history)):
        with (OUTPUT / name).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for name, model in (("demo-evaluation.json", agent), ("untrained-evaluation.json", QAgent())):
        (OUTPUT / name).write_text(json.dumps(evaluate(model, games=10_000, seed=7),
                                             indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256((OUTPUT / "demo.json").read_bytes()).hexdigest()
    bundled_digest = hashlib.sha256((ROOT / "models" / "demo.json").read_bytes()).hexdigest()
    provenance = {"python": platform.python_version(), "system": platform.system(),
                  "command": "python experiments.py", "model_sha256": digest,
                  "matches_bundled_model_bytes": digest == bundled_digest,
                  "curve_games_per_opponent": 1000, "final_games_per_opponent": 10_000,
                  "training_seed": 42, "evaluation_seed": 7}
    (OUTPUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Outputs: {OUTPUT}\nMatches bundled model: {digest == bundled_digest}")


if __name__ == "__main__":
    main()
