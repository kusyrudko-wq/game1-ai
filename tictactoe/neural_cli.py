"""Optional neural CLI, imported without PyTorch until execution."""
import json
from pathlib import Path
import time


def register(commands):
    training = commands.add_parser('train-neural', help='Train a new self-play DQN (PyTorch).')
    integers = dict(episodes=20000, seed=42, epsilon_decay_episodes=16000,
                    replay_capacity=20000, batch_size=128, warmup_steps=512,
                    train_every=4, target_update=250, report_every=1000, threads=1, width=64)
    floats = dict(learning_rate=0.001, gamma=1.0, epsilon_start=1.0,
                  epsilon_end=0.1, gradient_clip=10.0)
    for name, default in integers.items():
        training.add_argument('--' + name.replace('_', '-'), type=int, default=default)
    for name, default in floats.items():
        training.add_argument('--' + name.replace('_', '-'), type=float, default=default)
    training.add_argument('--run-dir', type=Path, required=True,
                          help='NEW directory under runs/; never overwritten.')
    training.add_argument('--resume', type=Path, help='Resume an existing full checkpoint; saved config wins.')
    evaluation = commands.add_parser('evaluate-neural', help='Evaluate a frozen DQN (PyTorch).')
    evaluation.add_argument('--model', type=Path, required=True)
    evaluation.add_argument('--games', type=int, default=1000)
    evaluation.add_argument('--seed', type=int, default=7)
    evaluation.add_argument('--output', type=Path)


def write_new_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as file:
        json.dump(payload, file, indent=2, allow_nan=False)
        file.write('\n')


def evaluate_neural(agent, games=1000, seed=7):
    from .evaluation import audit_all_positions, benchmark, worst_case_policy
    return dict(algorithm='zero-sum-dqn',
                random=benchmark(agent, 'random', games, seed),
                minimax=benchmark(agent, 'minimax', games, seed),
                exhaustive_from_empty=[worst_case_policy(agent.choose, side) for side in (1, -1)],
                all_positions_audit=audit_all_positions(agent))


def execute(args, root, print_report):
    from dataclasses import asdict, fields
    import torch
    from .dqn_training import DQNConfig, DQNTrainer, load_agent, save_weights
    if args.command == 'train-neural':
        run_dir = args.run_dir.resolve()
        runs = (root / 'runs').resolve()
        if runs not in run_dir.parents:
            raise ValueError('--run-dir must be a new subdirectory of runs/.')
        trainer = (DQNTrainer.load_checkpoint(args.resume) if args.resume else
                   DQNTrainer(DQNConfig(**{f.name: getattr(args, f.name) for f in fields(DQNConfig)})))
        run_dir.mkdir(parents=True, exist_ok=False)
        started = time.perf_counter()
        write_new_json(run_dir / 'config.json', dict(config=asdict(trainer.config),
                       torch_version=str(torch.__version__), device='cpu',
                       resumed_from=str(args.resume) if args.resume else None,
                       design='self-play only; no expert labels; alternating-player zero-sum DQN'))
        def progress(current, row):
            current.save_checkpoint(run_dir / f'checkpoint-{current.episode:06d}.pt')
            with (run_dir / 'metrics.jsonl').open('a', encoding='utf-8') as file:
                file.write(json.dumps(row, allow_nan=False) + '\n')
            print(f"Games: {current.episode:,} | updates: {current.updates:,} | "
                  f"epsilon: {row['epsilon']:.3f} | loss: {row['mean_loss']}", flush=True)
        agent, history = trainer.run(callback=progress)
        trainer.save_checkpoint(run_dir / 'checkpoint-final.pt')
        save_weights(agent, run_dir / 'weights.pt', asdict(trainer.config))
        write_new_json(run_dir / 'training-summary.json',
                       dict(episodes=trainer.episode, steps=trainer.steps, updates=trainer.updates,
                            elapsed_seconds=time.perf_counter() - started, history=history))
        print(f'Saved weights, checkpoints and metrics: {run_dir}', flush=True)
    else:
        if args.games < 2:
            raise ValueError('Use at least two games to evaluate both sides.')
        if args.output and args.output.exists():
            raise ValueError('Evaluation output already exists; use a new path.')
        agent = load_agent(args.model)
        report = evaluate_neural(agent, args.games, args.seed)
        report['model_path'] = str(args.model)
        print_report(report)
        if args.output:
            write_new_json(args.output, report)

