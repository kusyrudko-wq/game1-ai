import unittest
from tictactoe.game import Position, reachable_positions
from tictactoe.evaluation import minimax_value, optimal_moves, worst_case_policy


def position_after(*moves: int) -> Position:
    position = Position()
    for move in moves:
        position = position.play(move)
    return position


class GameTests(unittest.TestCase):
    def test_new_board(self):
        p = Position()
        self.assertIsNone(p.result())
        self.assertEqual(p.player, 1)
        self.assertEqual(p.legal_moves(), tuple(range(9)))

    def test_play_does_not_mutate(self):
        p = Position()
        q = p.play(4)
        self.assertEqual(p.board, (0,) * 9)
        self.assertEqual(q.board[4], 1)
        self.assertEqual(q.player, -1)

    def test_rows_columns_diagonals(self):
        for moves in ((0, 3, 1, 4, 2), (0, 1, 3, 2, 6),
                      (0, 1, 4, 2, 8), (2, 0, 4, 1, 6)):
            with self.subTest(moves=moves):
                p = position_after(*moves)
                self.assertEqual(p.result(), 1)
                self.assertEqual(p.legal_moves(), ())

    def test_o_can_win(self):
        self.assertEqual(position_after(0, 3, 1, 4, 8, 5).result(), -1)

    def test_draw(self):
        p = position_after(0, 1, 2, 4, 3, 5, 7, 6, 8)
        self.assertEqual(p.result(), 0)
        self.assertEqual(p.legal_moves(), ())

    def test_illegal_move(self):
        for move in (-1, 9, "1"):
            with self.assertRaises(ValueError):
                Position().play(move)
        with self.assertRaises(ValueError):
            Position().play(0).play(0)
        with self.assertRaises(ValueError):
            position_after(0, 3, 1, 4, 2).play(8)

    def test_key_uses_current_player_perspective(self):
        self.assertEqual(Position().key(), "000000000")
        self.assertEqual(Position().play(0).key(), "200000000")
        self.assertEqual(position_after(0, 1).key(), "120000000")

    def test_all_reachable_position_invariants(self):
        positions = reachable_positions()
        self.assertEqual(len(positions), 5478)
        for p in positions:
            x, o = p.board.count(1), p.board.count(-1)
            self.assertIn(x - o, (0, 1))
            self.assertEqual(p.player, 1 if x == o else -1)
            if p.result() is None:
                self.assertEqual(len(p.legal_moves()), p.board.count(0))
            else:
                self.assertFalse(p.legal_moves())


class SolverTests(unittest.TestCase):
    def test_perfect_play_draws(self):
        self.assertEqual(minimax_value(Position()), 0)

    def test_immediate_win(self):
        p = position_after(0, 3, 1, 4)
        self.assertIn(2, optimal_moves(p))
        self.assertEqual(minimax_value(p), 1)

    def test_forced_block(self):
        p = position_after(0, 4, 1)
        self.assertEqual(optimal_moves(p), (2,))

    def test_perfect_policy_cannot_lose_either_side(self):
        for side in (1, -1):
            report = worst_case_policy(lambda p: optimal_moves(p)[0], side)
            self.assertEqual(report["worst_case_score"], 0)
            self.assertTrue(report["cannot_lose_from_empty"])

    def test_bad_policy_is_exploitable(self):
        report = worst_case_policy(lambda p: p.legal_moves()[0], -1)
        self.assertEqual(report["worst_case_score"], -1)


if __name__ == "__main__":
    unittest.main()
