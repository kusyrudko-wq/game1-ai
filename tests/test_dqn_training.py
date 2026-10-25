from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from tictactoe.dqn_training import (DQNConfig, DQNTrainer, bellman_targets,
                                   load_agent, save_weights, transition)
from tictactoe.game import Position
from tictactoe.neural_cli import evaluate_neural


def position_after(*moves):
    p = Position()
    for move in moves:
        p = p.play(move)
    return p


class DQNTests(unittest.TestCase):
    def setUp(self):
        self.config = DQNConfig(episodes=12, seed=19, batch_size=8, warmup_steps=8,
                                replay_capacity=64, train_every=2, target_update=3,
                                report_every=6, epsilon_decay_episodes=10)
        self.trainer = DQNTrainer(self.config)

    def test_transition_perspectives_and_terminal_rewards(self):
        t = transition(Position(), 0)
        self.assertEqual(t[0], (0,) * 9)
        self.assertEqual(t[3], (-1,) + (0,) * 8)
        self.assertFalse(t[4])
        self.assertFalse(t[5][0])
        for moves, action in (((0, 3, 1, 4), 2), ((0, 3, 1, 4, 8), 5)):
            t = transition(position_after(*moves), action)
            self.assertEqual(t[2], 1.0)
            self.assertTrue(t[4])
            self.assertFalse(any(t[5]))
        t = transition(position_after(0, 1, 2, 4, 3, 5, 7, 6), 8)
        self.assertEqual(t[2], 0.0)
        self.assertTrue(t[4])

    def test_target_sign_mask_and_terminal_no_bootstrap(self):
        network = mock.Mock(return_value=torch.tensor([[999., .7, -1., 0., 0., 0., 0., 0., 0.]]))
        legal = torch.zeros(3, 9, dtype=torch.bool)
        legal[0, 1:3] = True
        targets = bellman_targets(network, torch.zeros(3, 9),
                                  torch.tensor([0., 1., 0.]),
                                  torch.tensor([False, True, True]), legal, .5)
        torch.testing.assert_close(targets, torch.tensor([-.35, 1., 0.]))
        self.assertEqual(network.call_args.args[0].shape, (1, 9))
        self.assertFalse(targets.requires_grad)
        # All-negative legal values must not be replaced by illegal zeros.
        network.return_value = torch.tensor([[0., -.7, -1., 0., 0., 0., 0., 0., 0.]])
        targets = bellman_targets(network, torch.zeros(1, 9), torch.zeros(1),
                                  torch.tensor([False]), legal[:1], 1.)
        torch.testing.assert_close(targets, torch.tensor([.7]))

    def test_all_terminal_batch_never_calls_target(self):
        network = mock.Mock(side_effect=AssertionError('Terminal bootstrap'))
        result = bellman_targets(network, torch.zeros(2, 9), torch.tensor([1., 0.]),
                                 torch.ones(2, dtype=torch.bool),
                                 torch.zeros(2, 9, dtype=torch.bool), 1.)
        self.assertEqual(result.tolist(), [1., 0.])

    def test_exploration_and_greedy_are_legal_for_both_sides(self):
        for p in (position_after(0), position_after(0, 4)):
            for epsilon in (0., 1.):
                for _ in range(30):
                    self.assertIn(self.trainer.choose(p, epsilon), p.legal_moves())
        with self.assertRaises(ValueError):
            self.trainer.choose(position_after(0, 3, 1, 4, 2), 1.)

    def test_update_changes_online_only_until_target_sync(self):
        self.trainer.replay.extend([transition(Position(), 0)] * 8)
        before = {k: v.clone() for k, v in self.trainer.network.state_dict().items()}
        self.assertIsNotNone(self.trainer.update())
        self.assertTrue(any(not torch.equal(v, before[k])
                            for k, v in self.trainer.network.state_dict().items()))
        for key, value in self.trainer.target.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))
        self.trainer.update()
        self.trainer.update()
        for key, value in self.trainer.target.state_dict().items():
            self.assertTrue(torch.equal(value, self.trainer.network.state_dict()[key]))
        self.assertTrue(all(p.grad is None for p in self.trainer.target.parameters()))

    def test_warmup_and_replay_capacity(self):
        self.assertIsNone(self.trainer.update())
        self.trainer.run()
        self.assertLessEqual(len(self.trainer.replay), self.config.replay_capacity)
        self.assertGreater(self.trainer.updates, 0)
        self.assertEqual(sum(row['x_wins'] + row['o_wins'] + row['draws']
                             for row in self.trainer.history), self.config.episodes)

    def test_checkpoint_resume_is_identical_to_uninterrupted_training(self):
        self.trainer.run(until=5)
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / 'checkpoint.pt'
            self.trainer.save_checkpoint(checkpoint)
            loaded = DQNTrainer.load_checkpoint(checkpoint)
            self.assertEqual(list(loaded.replay), list(self.trainer.replay))
            self.assertEqual(loaded.rng.getstate(), self.trainer.rng.getstate())
            loaded.run()
            with self.assertRaises(FileExistsError):
                loaded.save_checkpoint(checkpoint)
        full = DQNTrainer(self.config)
        full.run()
        self.assertEqual(loaded.history, full.history)
        for key, value in loaded.network.state_dict().items():
            self.assertTrue(torch.equal(value, full.network.state_dict()[key]))
        for key, value in loaded.target.state_dict().items():
            self.assertTrue(torch.equal(value, full.target.state_dict()[key]))

    def test_weights_round_trip_and_frozen_evaluation(self):
        self.trainer.run()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'weights.pt'
            save_weights(self.trainer.agent, path)
            agent = load_agent(path)
            with self.assertRaises(FileExistsError):
                save_weights(agent, path)
        before = {k: v.clone() for k, v in agent.network.state_dict().items()}
        report = evaluate_neural(agent, games=10, seed=7)
        self.assertEqual(report['all_positions_audit']['decision_states'], 4520)
        for opponent in ('random', 'minimax'):
            for side in ('X', 'O'):
                self.assertEqual(sum(report[opponent]['by_side'][side].values()), 5)
        for key, value in agent.network.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))
            self.assertTrue(torch.equal(value, self.trainer.network.state_dict()[key]))

    def test_invalid_config_rejected(self):
        for change in (dict(gamma=-1), dict(batch_size=0), dict(learning_rate=float('nan')),
                       dict(epsilon_end=2), dict(target_update=0), dict(replay_capacity=1)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(self.config, **change)


if __name__ == '__main__':
    unittest.main()

class NeuralCLITests(unittest.TestCase):
    def test_tabular_cli_works_without_importing_torch(self):
        import subprocess
        import sys
        code = '''
import builtins
original_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise AssertionError("Tabular command must not import PyTorch")
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded
import sys
sys.argv = ["tictactoe", "evaluate", "--games", "2", "--seed", "7"]
from tictactoe.__main__ import main
raise SystemExit(main())
'''
        result = subprocess.run([sys.executable, '-c', code], capture_output=True,
                                text=True, cwd=Path(__file__).resolve().parents[1], timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('CANNOT LOSE', result.stdout)

    def test_cli_rejects_existing_run_and_paths_outside_runs(self):
        import argparse
        from tictactoe.neural_cli import execute, register
        parser = argparse.ArgumentParser()
        register(parser.add_subparsers(dest='command'))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for destination in (root / 'models' / 'demo.json', root / 'runs' / 'existing'):
                if destination.name == 'existing':
                    destination.mkdir(parents=True)
                args = parser.parse_args(['train-neural', '--episodes', '1',
                                          '--run-dir', str(destination)])
                with self.subTest(destination=destination), self.assertRaises((ValueError, FileExistsError)):
                    execute(args, root, mock.Mock())
            self.assertFalse((root / 'models').exists())

    def test_evaluation_refuses_to_overwrite_model(self):
        import argparse
        from tictactoe.neural_cli import execute, register
        parser = argparse.ArgumentParser()
        register(parser.add_subparsers(dest='command'))
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / 'model.pt'
            model.write_bytes(b'preserve this file')
            args = parser.parse_args(['evaluate-neural', '--model', str(model),
                                      '--output', str(model)])
            with self.assertRaises(ValueError):
                execute(args, Path(folder), mock.Mock())
            self.assertEqual(model.read_bytes(), b'preserve this file')


if __name__ == '__main__':
    unittest.main()
