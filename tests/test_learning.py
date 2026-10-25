import copy
import json
from pathlib import Path
import random
import tempfile
import unittest

from tictactoe.agent import QAgent
from tictactoe.evaluation import audit_all_positions, benchmark, worst_case_policy
from tictactoe.game import Position
from tictactoe.training import train

ROOT = Path(__file__).resolve().parents[1]


def position_after(*moves):
    position = Position()
    for move in moves:
        position = position.play(move)
    return position


class LearningTests(unittest.TestCase):
    def test_unseen_state_is_not_inserted_during_play(self):
        agent = QAgent()
        agent.choose(Position())
        self.assertEqual(agent.table, {})

    def test_legal_move_mask(self):
        p = Position().play(0)
        agent = QAgent({p.key(): [999.0] + [0.0] * 8})
        self.assertEqual(agent.choose(p), 1)
        for seed in range(30):
            self.assertIn(agent.choose(p, random.Random(seed), 1.0), p.legal_moves())

    def test_deterministic_tie_break(self):
        agent = QAgent()
        self.assertEqual(agent.choose(Position()), 0)
        self.assertEqual(agent.choose(Position()), 0)

    def test_terminal_win_reward_for_x_and_o(self):
        for moves, winning_move in (((0, 3, 1, 4), 2), ((0, 3, 1, 4, 8), 5)):
            p = position_after(*moves)
            agent = QAgent()
            agent.update(p, winning_move, p.play(winning_move), alpha=1)
            self.assertEqual(agent.values(p)[winning_move], 1.0)

    def test_terminal_draw_reward(self):
        p = position_after(0, 1, 2, 4, 3, 5, 7, 6)
        agent = QAgent({p.key(): [1.0] * 9})
        agent.update(p, 8, p.play(8), alpha=1)
        self.assertEqual(agent.values(p)[8], 0.0)

    def test_opponent_value_must_be_negated(self):
        p = Position()
        next_p = p.play(0)
        agent = QAgent({next_p.key(): [999.0] + [0.7] * 8})
        agent.update(p, 0, next_p, alpha=1)
        self.assertAlmostEqual(agent.values(p)[0], -0.7)

    def test_learning_rate(self):
        p = position_after(0, 3, 1, 4)
        agent = QAgent()
        agent.update(p, 2, p.play(2), alpha=0.2)
        self.assertAlmostEqual(agent.values(p)[2], 0.2)

    def test_bad_training_parameters_rejected(self):
        for params in ({"episodes": 0}, {"alpha": 0},
                       {"epsilon_start": 0.2, "epsilon_end": 0.9}):
            with self.assertRaises(ValueError):
                train(**params)

    def test_training_is_reproducible(self):
        first, history1 = train(1000, seed=123)
        second, history2 = train(1000, seed=123)
        self.assertEqual(first.table, second.table)
        self.assertEqual(history1, history2)
        self.assertTrue(first.table)

    def test_evaluation_does_not_train(self):
        agent, _ = train(200, seed=2)
        before = copy.deepcopy(agent.table)
        result = benchmark(agent, games=20)
        worst_case_policy(agent.choose, 1)
        audit_all_positions(agent)
        self.assertEqual(agent.table, before)
        self.assertEqual(result["wins"] + result["draws"] + result["losses"], 20)
        for side in ("X", "O"):
            self.assertEqual(sum(result["by_side"][side].values()), 10)

    def test_json_round_trip(self):
        agent, _ = train(100, seed=4)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            agent.save(path)
            loaded = QAgent.load(path)
        self.assertEqual(agent.table, loaded.table)
        self.assertEqual(agent.metadata, loaded.metadata)

    def test_invalid_model_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            path.write_text(json.dumps({"format_version": 1,
                "algorithm": "zero-sum-tabular-q-learning", "q_values": {"bad": [0]}}))
            with self.assertRaises(ValueError):
                QAgent.load(path)


class BundledModelTests(unittest.TestCase):
    def test_demo_optimal_at_every_reachable_state(self):
        agent = QAgent.load(ROOT / "models" / "demo.json")
        self.assertEqual(audit_all_positions(agent)["suboptimal_decisions"], 0)

    def test_demo_cannot_lose_from_empty_board(self):
        # Only certify the bundled checkpoint, NOT arbitrary new training runs.
        agent = QAgent.load(ROOT / "models" / "demo.json")
        for side in (1, -1):
            with self.subTest(side=side):
                self.assertTrue(worst_case_policy(agent.choose, side)["cannot_lose_from_empty"])


if __name__ == "__main__":
    unittest.main()
