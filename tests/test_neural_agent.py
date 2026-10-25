import unittest
from unittest import mock

import torch
from torch import nn

from tictactoe.game import Position
from tictactoe.neural_agent import NeuralAgent, QNetwork, encode_board


def position_after(*moves):
    position = Position()
    for move in moves:
        position = position.play(move)
    return position


class NeuralAgentTests(unittest.TestCase):
    def setUp(self):
        # Keep network initialization reproducible without changing other tests.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            self.network = QNetwork()
        self.agent = NeuralAgent(self.network)

    def set_values(self, values):
        with torch.no_grad():
            for parameter in self.network.parameters():
                parameter.zero_()
            self.network.layers[-1].bias.copy_(torch.tensor(values))

    def test_board_encoding_for_both_players(self):
        cases = (
            (Position(), [0] * 9),
            (position_after(0), [-1, 0, 0, 0, 0, 0, 0, 0, 0]),
            (position_after(0, 4), [1, 0, 0, 0, -1, 0, 0, 0, 0]),
            (position_after(0, 4, 8), [-1, 0, 0, 0, 1, 0, 0, 0, -1]),
        )
        for position, expected in cases:
            with self.subTest(position=position):
                encoded = encode_board(position)
                self.assertEqual(encoded.tolist(), expected)
                self.assertEqual(encoded.dtype, torch.float32)
                self.assertEqual(encoded.shape, (9,))

    def test_architecture_and_output_shapes(self):
        layers = list(self.network.layers)
        self.assertEqual([type(layer) for layer in layers],
                         [nn.Linear, nn.ReLU, nn.Linear, nn.ReLU, nn.Linear])
        self.assertEqual([(layer.in_features, layer.out_features)
                          for layer in layers if isinstance(layer, nn.Linear)],
                         [(9, 64), (64, 64), (64, 9)])
        self.assertEqual(self.network(torch.zeros(9)).shape, (9,))
        self.assertEqual(self.network(torch.zeros(4, 9)).shape, (4, 9))

    def test_best_legal_move_for_x_and_o_with_negative_values(self):
        self.set_values([100, -9, -8, -7, 200, -6, -1, -4, -3])
        for position in (position_after(0, 4), position_after(0, 4, 8)):
            with self.subTest(player=position.player):
                self.assertEqual(self.agent.choose(position), 6)
                self.assertIn(self.agent.choose(position), position.legal_moves())

    def test_ties_choose_lowest_legal_index(self):
        self.set_values([0] * 9)
        self.assertEqual(self.agent.choose(Position()), 0)
        self.assertEqual(self.agent.choose(position_after(0)), 1)

    def test_finished_games_rejected(self):
        for position in (position_after(0, 3, 1, 4, 2),
                         position_after(0, 1, 2, 4, 3, 5, 7, 6, 8)):
            with self.subTest(position=position):
                with self.assertRaisesRegex(ValueError, "finished game"):
                    self.agent.choose(position)

    def test_inference_does_not_change_weights_or_compute_gradients(self):
        before = {key: value.clone()
                  for key, value in self.network.state_dict().items()}
        with mock.patch.object(self.network, 'forward',
                                       wraps=self.network.forward) as forward:
            self.agent.choose(Position())
            self.assertEqual(forward.call_count, 1)
        for key, value in self.network.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))
        self.assertTrue(all(p.grad is None for p in self.network.parameters()))
        self.assertFalse(self.network.training)


if __name__ == '__main__':
    unittest.main()

