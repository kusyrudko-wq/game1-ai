import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from tictactoe.dqn_training import DQNConfig, DQNTrainer, load_agent, save_weights
from tictactoe.dqn_experiments import configurations, rank_key, run_suite, tree_hashes
from tictactoe.game import Position
from tictactoe.neural_agent import QNetwork

ROOT = Path(__file__).resolve().parents[1]


class ExperimentTests(unittest.TestCase):
    def test_plan_covers_requested_axes_and_is_reproducible(self):
        first = configurations()
        self.assertEqual(first, configurations())
        self.assertGreaterEqual(len(first), 12)
        self.assertEqual(len({name for name, _ in first}), len(first))
        for attribute in ('learning_rate', 'gamma', 'epsilon_end',
                          'epsilon_decay_episodes', 'target_update', 'width', 'episodes'):
            self.assertGreater(len({getattr(config, attribute) for _, config in first}), 1)

    def test_ranking_requires_safety_before_audit_score(self):
        safe = dict(safe_from_empty=True, audit_errors=50, minimax_losses=0,
                    random_score_rate=.9, name='safe', seed=42)
        unsafe = dict(safe, safe_from_empty=False, audit_errors=0, name='unsafe')
        self.assertLess(rank_key(safe), rank_key(unsafe))
        self.assertLess(rank_key(dict(safe, audit_errors=20)), rank_key(safe))

    def test_width_weights_checkpoint_and_legacy_loading(self):
        config = DQNConfig(episodes=4, width=32, batch_size=8, warmup_steps=8,
                           replay_capacity=64, report_every=4)
        trainer = DQNTrainer(config)
        trainer.run(until=2)
        with tempfile.TemporaryDirectory() as folder:
            weights, checkpoint = Path(folder) / 'weights.pt', Path(folder) / 'checkpoint.pt'
            save_weights(trainer.agent, weights)
            loaded = load_agent(weights)
            self.assertEqual(loaded.network.width, 32)
            trainer.save_checkpoint(checkpoint)
            resumed = DQNTrainer.load_checkpoint(checkpoint)
            trainer.run()
            resumed.run()
            for key, value in trainer.network.state_dict().items():
                self.assertTrue(torch.equal(value, resumed.network.state_dict()[key]))
            for key, value in loaded.network.state_dict().items():
                self.assertEqual(value.shape, trainer.network.state_dict()[key].shape)
        # Legacy weights/checkpoints omit width; tests need no ignored runs/ artifacts.
        with tempfile.TemporaryDirectory() as folder:
            legacy = DQNTrainer(replace(config, width=64))
            weights = Path(folder) / 'legacy-weights.pt'
            torch.save(dict(format_version=1, algorithm='zero-sum-dqn',
                            network=legacy.network.state_dict()), weights)
            baseline = load_agent(weights)
            self.assertEqual(baseline.network.width, 64)
            self.assertIn(baseline.choose(Position()), Position().legal_moves())
            checkpoint = Path(folder) / 'legacy-checkpoint.pt'
            legacy.save_checkpoint(checkpoint)
            payload = torch.load(checkpoint, weights_only=True)
            del payload['config']['width']
            old_checkpoint = Path(folder) / 'old.pt'
            torch.save(payload, old_checkpoint)
            self.assertEqual(DQNTrainer.load_checkpoint(old_checkpoint).config.width, 64)
        for bad in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                QNetwork(bad)

    def test_runner_saves_results_best_and_preserves_baseline(self):
        config = DQNConfig(episodes=3, batch_size=8, warmup_steps=8,
                           replay_capacity=64, report_every=3)
        counts = dict(wins=0, draws=2, losses=0)
        report = dict(all_positions_audit=dict(suboptimal_decisions=3),
                      exhaustive_from_empty=[dict(cannot_lose_from_empty=True,
                                                  worst_case_score=0)] * 2)
        for opponent in ('random', 'minimax'):
            report[opponent] = dict(counts, score_rate=.5,
                                   by_side={side: dict(wins=0, draws=1, losses=0)
                                            for side in ('X', 'O')})
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            baseline = root / 'baseline'
            baseline.mkdir()
            trainer = DQNTrainer(config)
            save_weights(trainer.agent, baseline / 'weights.pt')
            trainer.save_checkpoint(baseline / 'checkpoint-final.pt')
            (baseline / 'config.json').write_text(json.dumps(dict(config=asdict(config))))
            (baseline / 'training-summary.json').write_text(json.dumps(dict(elapsed_seconds=0)))
            before = tree_hashes(baseline)
            output = root / 'runs' / 'experiment'
            with mock.patch('tictactoe.dqn_experiments.ROOT', root), \
                 mock.patch('tictactoe.dqn_experiments.configurations',
                            return_value=[('a', config), ('b', replace(config, width=32))]), \
                 mock.patch('tictactoe.dqn_experiments.evaluate_neural', return_value=report):
                rows = run_suite(output, games=2, baseline=baseline)
                self.assertEqual(len(rows), 7)  # baseline + two screening + four followups
                with self.assertRaises(FileExistsError):
                    run_suite(output, games=2, baseline=baseline)
            self.assertEqual(tree_hashes(baseline), before)
            with (output / 'leaderboard.csv').open(newline='') as file:
                self.assertEqual(len(list(csv.DictReader(file))), 7)
            summary = json.loads((output / 'summary.json').read_text())
            self.assertTrue(summary['complete'])
            self.assertTrue(summary['baseline_preserved'])
            self.assertTrue((output / 'best-checkpoint.pt').is_file())
            self.assertIn(load_agent(output / 'best-weights.pt').choose(Position()), range(9))


if __name__ == '__main__':
    unittest.main()

