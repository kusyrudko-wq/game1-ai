"""Reproducible CPU DQN search; solver scores select models, never train them.

Run: python -m tictactoe.dqn_experiments --output runs/dqn-search
Every candidate starts from scratch. Existing outputs are refused.
"""
import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import torch

from .dqn_training import DQNConfig, DQNTrainer, load_agent, save_weights
from .neural_cli import evaluate_neural, write_new_json

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / 'runs' / 'dqn-20000-20261024'


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def tree_hashes(directory):
    return {str(path.relative_to(directory)): file_hash(path)
            for path in sorted(Path(directory).rglob('*')) if path.is_file()}


def configurations():
    """Fixed screening plan: seed 42, single-factor controls plus combinations."""
    base = DQNConfig(episodes=10000, epsilon_decay_episodes=8000, report_every=2500)
    changes = [
        ('control', {}),
        ('lr-low', dict(learning_rate=.0003)),
        ('lr-high', dict(learning_rate=.003)),
        ('gamma-095', dict(gamma=.95)),
        ('gamma-099', dict(gamma=.99)),
        ('explore-more', dict(epsilon_end=.3, epsilon_decay_episodes=10000)),
        ('explore-fast', dict(epsilon_end=.05, epsilon_decay_episodes=4000)),
        ('target-fast', dict(target_update=100)),
        ('target-slow', dict(target_update=1000)),
        ('width-128-g095', dict(width=128, gamma=.95)),
        ('short-width-32', dict(width=32, episodes=5000, epsilon_decay_episodes=4000)),
        ('long-g095-explore', dict(episodes=20000, gamma=.95,
                                   epsilon_end=.2, epsilon_decay_episodes=15000)),
    ]
    return [(name, replace(base, **params)) for name, params in changes]


def rank_key(row):
    """Safety is mandatory; time is never a tie-break (keeps selection repeatable)."""
    return (not row['safe_from_empty'], row['audit_errors'], row['minimax_losses'],
            -row['random_score_rate'], row['name'], row['seed'])


def result_row(name, phase, config, report, training_seconds, evaluation_seconds, directory):
    row = dict(name=name, phase=phase, **asdict(config),
               audit_errors=report['all_positions_audit']['suboptimal_decisions'],
               safe_from_empty=all(r['cannot_lose_from_empty'] for r in report['exhaustive_from_empty']),
               training_seconds=training_seconds, evaluation_seconds=evaluation_seconds,
               directory=str(directory))
    for side, result in zip(('x', 'o'), report['exhaustive_from_empty']):
        row[f'worst_case_{side}'] = result['worst_case_score']
    for opponent in ('random', 'minimax'):
        for metric in ('wins', 'draws', 'losses', 'score_rate'):
            row[f'{opponent}_{metric}'] = report[opponent][metric]
        for side in ('X', 'O'):
            for metric in ('wins', 'draws', 'losses'):
                row[f'{opponent}_{side.lower()}_{metric}'] = report[opponent]['by_side'][side][metric]
    return row


def write_leaderboard(output, rows):
    # This one derived file is refreshed; candidate artifacts are immutable.
    with (output / 'leaderboard.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=rank_key))


def run_candidate(output, name, phase, config, games, evaluation_seed):
    directory = output / f'{name}-seed-{config.seed}'
    directory.mkdir(exist_ok=False)
    write_new_json(directory / 'config.json', asdict(config))
    start = time.perf_counter()
    trainer = DQNTrainer(config)
    def progress(current, row):
        with (directory / 'metrics.jsonl').open('a', encoding='utf-8') as file:
            file.write(json.dumps(row, allow_nan=False) + '\n')
        print(f"  {name} seed={config.seed}: {current.episode}/{config.episodes}", flush=True)
    agent, _ = trainer.run(callback=progress)
    trainer.save_checkpoint(directory / 'checkpoint.pt')
    save_weights(agent, directory / 'weights.pt', asdict(config))
    training_seconds = time.perf_counter() - start
    start = time.perf_counter()
    report = evaluate_neural(agent, games, evaluation_seed)
    evaluation_seconds = time.perf_counter() - start
    write_new_json(directory / 'evaluation.json', report)
    row = result_row(name, phase, config, report, training_seconds, evaluation_seconds,
                     directory.relative_to(output))
    write_new_json(directory / 'result.json', row)
    print(f"DONE {name} seed={config.seed}: errors={row['audit_errors']}, "
          f"safe={row['safe_from_empty']}, train={training_seconds:.2f}s", flush=True)
    return row


def run_suite(output, games=1000, evaluation_seed=7, budget_seconds=1800,
              baseline=BASELINE):
    if games < 2 or games % 2:
        raise ValueError('Use an even game count >= 2 for equal X/O samples.')
    if budget_seconds <= 0:
        raise ValueError('CPU wall-time budget must be positive.')
    output = Path(output).resolve()
    if (ROOT / 'runs').resolve() not in output.parents:
        raise ValueError('Output must be a new subdirectory of runs/.')
    baseline = Path(baseline).resolve()
    baseline_hashes = tree_hashes(baseline)
    if not (baseline / 'weights.pt').is_file():
        raise ValueError('Missing baseline weights.')
    plan = configurations()
    output.mkdir(parents=True, exist_ok=False)
    source_dir = output / 'source'
    source_dir.mkdir()
    source_hashes = {}
    for directory in ('tictactoe', 'tests'):
        for source in sorted((ROOT / directory).glob('*.py')):
            target = source_dir / directory / source.name
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(source, target)
            source_hashes[str(source.relative_to(ROOT))] = file_hash(source)
    write_new_json(output / 'manifest.json', dict(
        format_version=1, python=sys.version, torch=str(torch.__version__),
        platform=platform.platform(), device='cpu', threads=1,
        games_per_opponent=games, evaluation_seed=evaluation_seed,
        budget_seconds=budget_seconds, baseline=str(baseline),
        baseline_hashes=baseline_hashes, source_hashes=source_hashes,
        screening=[dict(name=name, config=asdict(config)) for name, config in plan],
        followup=dict(top_configurations=2, seeds=[43, 44]),
        selection='require no forced loss as X and O; minimize all-state errors; '
                  'ties: minimax losses, random score, name, seed; include baseline',
        design='all models start from scratch; self-play only; no expert targets; '
               'minimax audits used for model/hyperparameter selection only'))
    started = time.perf_counter()
    torch.set_num_threads(1)
    baseline_config = DQNConfig(**json.loads((baseline / 'config.json').read_text())['config'])
    before = time.perf_counter()
    baseline_report = evaluate_neural(load_agent(baseline / 'weights.pt'), games, evaluation_seed)
    baseline_eval_seconds = time.perf_counter() - before
    write_new_json(output / 'baseline-evaluation.json', baseline_report)
    old_time = json.loads((baseline / 'training-summary.json').read_text())['elapsed_seconds']
    rows = [result_row('baseline', 'baseline', baseline_config, baseline_report,
                       old_time, baseline_eval_seconds, baseline)]
    write_leaderboard(output, rows)
    def run_one(name, phase, config):
        if time.perf_counter() - started >= budget_seconds:
            return False
        row = run_candidate(output, name, phase, config, games, evaluation_seed)
        rows.append(row)
        write_leaderboard(output, rows)
        return True
    for name, config in plan:
        if not run_one(name, 'screen', config):
            break
    screening = [r for r in rows if r['phase'] == 'screen']
    promising = sorted([r for r in screening if r['safe_from_empty']], key=rank_key)[:2]
    followup_plan = [(r['name'], replace(dict(plan)[r['name']], seed=seed))
                     for r in promising for seed in (43, 44)]
    write_new_json(output / 'followup-plan.json',
                   [dict(name=name, config=asdict(config)) for name, config in followup_plan])
    for name, config in followup_plan:
        if not run_one(name, 'followup', config):
            break
    eligible = [r for r in rows if r['safe_from_empty']]
    best = min(eligible, key=rank_key) if eligible else None
    if best:
        origin = baseline if best['phase'] == 'baseline' else output / best['directory']
        checkpoint = origin / ('checkpoint-final.pt' if best['phase'] == 'baseline' else 'checkpoint.pt')
        shutil.copy2(checkpoint, output / 'best-checkpoint.pt')
        shutil.copy2(origin / 'weights.pt', output / 'best-weights.pt')
        write_new_json(output / 'best.json', dict(result=best,
                       weights_sha256=file_hash(output / 'best-weights.pt'),
                       checkpoint_sha256=file_hash(output / 'best-checkpoint.pt')))
    preserved = tree_hashes(baseline) == baseline_hashes
    elapsed = time.perf_counter() - started
    write_new_json(output / 'summary.json', dict(
        screening_completed=len(screening), followups_completed=sum(r['phase'] == 'followup' for r in rows),
        elapsed_seconds=elapsed, new_training_seconds=sum(r['training_seconds'] for r in rows if r['phase'] != 'baseline'),
        baseline_preserved=preserved, best=best,
        complete=len(screening) == len(plan) and sum(r['phase'] == 'followup' for r in rows) == len(followup_plan)))
    if not preserved:
        raise RuntimeError('Baseline content changed during the experiment.')
    print(f'Suite finished in {elapsed:.2f}s. Best: {best}', flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--games', type=int, default=1000)
    parser.add_argument('--evaluation-seed', type=int, default=7)
    parser.add_argument('--budget-seconds', type=float, default=1800)
    args = parser.parse_args()
    run_suite(args.output, args.games, args.evaluation_seed, args.budget_seconds)


if __name__ == '__main__':
    main()


