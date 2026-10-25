"""Command-line interface: python -m tictactoe {play,train,evaluate}."""
import argparse
import csv
import json
from pathlib import Path
import random
import sys

from .agent import QAgent
from .evaluation import evaluate, optimal_moves
from .game import Position
from .training import train

ROOT = Path(__file__).resolve().parents[1]
DEMO_MODEL = ROOT / "models" / "demo.json"


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def print_report(report: dict) -> None:
    for name in ("random", "minimax"):
        result = report[name]
        print(f"Against {name:7}: {result['wins']} wins, {result['draws']} draws, "
              f"{result['losses']} losses ({result['games']} games)")
        for side, counts in result["by_side"].items():
            print(f"  AI as {side}: {counts['wins']} W / {counts['draws']} D / "
                  f"{counts['losses']} L")
    print("Exhaustive check against all opponent replies, from the empty board:")
    for result in report["exhaustive_from_empty"]:
        status = "CANNOT LOSE" if result["cannot_lose_from_empty"] else "CAN LOSE"
        print(f"  AI as {result['agent_side']}: {status}; worst-case score "
              f"{result['worst_case_score']}")
        if not result["cannot_lose_from_empty"]:
            print("  Example losing line (squares 1..9):",
                  result["example_worst_case_line_1_to_9"])
    audit = report["all_positions_audit"]
    print(f"All-state audit: {audit['suboptimal_decisions']} suboptimal decisions / "
          f"{audit['decision_states']} reachable nonterminal positions.")
    print("Checks apply to this exact model and deterministic tie-break, not all training runs.")


def play_game(opponent: str, model: Path, human: str, seed: int) -> None:
    agent = QAgent.load(model) if opponent == "q" else None
    rng = random.Random(seed)
    human_player = 1 if human == "X" else -1
    position = Position()
    print(f"You are {human}. X moves first. Choose a square 1..9; q quits.")
    while position.result() is None:
        print("\n" + position.render() + "\n")
        if position.player == human_player:
            text = input("Your move: ").strip()
            if text.lower() in ("q", "quit", "exit"):
                return
            try:
                move = int(text) - 1
            except ValueError:
                print("Enter a number 1..9, or q to quit.")
                continue
            if move not in position.legal_moves():
                print("That square is unavailable. Choose an empty square.")
                continue
        else:
            if agent is not None:
                move = agent.choose(position)
            else:
                moves = position.legal_moves() if opponent == "random" else optimal_moves(position)
                move = rng.choice(moves)
            print(f"AI plays {move + 1}.")
        position = position.play(move)
    print("\n" + position.render())
    result = position.result()
    print("Draw." if result == 0 else "You win!" if result == human_player else "AI wins.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Learn a game through self-play, then test it.")
    commands = parser.add_subparsers(dest="command", required=True)

    play = commands.add_parser("play", help="Play a game against the AI.")
    play.add_argument("--opponent", choices=("q", "random", "minimax"), default="q")
    play.add_argument("--model", type=Path, default=DEMO_MODEL)
    play.add_argument("--human", type=str.upper, choices=("X", "O"), default="X")
    play.add_argument("--seed", type=int, default=7)

    training = commands.add_parser("train", help="Train a NEW Q table from scratch.")
    training.add_argument("--episodes", type=int, default=200_000)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--alpha", type=float, default=0.2)
    training.add_argument("--epsilon-start", type=float, default=1.0)
    training.add_argument("--epsilon-end", type=float, default=0.1)
    training.add_argument("--output", type=Path, default=ROOT / "models" / "my-agent.json")
    training.add_argument("--log", type=Path, default=ROOT / "runs" / "training.csv")
    training.add_argument("--overwrite", action="store_true", help="Allow replacing model/log files.")

    evaluation = commands.add_parser("evaluate", help="Benchmark a frozen model and audit it.")
    evaluation.add_argument("--model", type=Path, default=DEMO_MODEL)
    evaluation.add_argument("--games", type=int, default=1000, help="Games per opponent; starts alternate.")
    evaluation.add_argument("--seed", type=int, default=7)
    evaluation.add_argument("--output", type=Path)
    evaluation.add_argument("--untrained", action="store_true", help="Evaluate an empty table instead.")

    from .neural_cli import register
    register(commands)
    args = parser.parse_args()
    try:
        if args.command in ('train-neural', 'evaluate-neural'):
            from .neural_cli import execute
            execute(args, ROOT, print_report)
        elif args.command == "play":
            play_game(args.opponent, args.model, args.human, args.seed)
        elif args.command == "train":
            if args.output.resolve() == args.log.resolve():
                raise ValueError("Model and log must use different paths.")
            if not args.overwrite and (args.output.exists() or args.log.exists()):
                raise ValueError("Output model/log already exists. Use new paths or --overwrite.")

            def progress(episode: int, agent: QAgent, row: dict) -> None:
                print(f"Games: {episode:>7,} | states: {len(agent.table):>4} | "
                      f"exploration: {row['epsilon']:.3f}", flush=True)

            agent, history = train(args.episodes, args.seed, args.alpha,
                                   args.epsilon_start, args.epsilon_end, callback=progress)
            agent.save(args.output)
            args.log.parent.mkdir(parents=True, exist_ok=True)
            with args.log.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(history[0]))
                writer.writeheader()
                writer.writerows(history)
            print(f"Saved model: {args.output}\nSaved self-play log: {args.log}")
            print("Training completed. Run evaluate; training alone is not proof of strength.")
        else:
            agent = QAgent() if args.untrained else QAgent.load(args.model)
            report = evaluate(agent, args.games, args.seed)
            print_report(report)
            if args.output:
                if not args.untrained and args.output.resolve() == args.model.resolve():
                    raise ValueError("Evaluation output must not replace the model.")
                write_json(args.output, report)
                print(f"Saved evaluation: {args.output}")
    except FileNotFoundError as exc:
        print(f"File not found: {exc.filename}. Check the path or train a model first.", file=sys.stderr)
        return 1
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        print("\nStopped.")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

