import unittest

import numpy as np
from gym.spaces import MultiDiscrete
from hexarena.arena import Arena
from hexarena.monkey import Monkey

rng = np.random.default_rng()


class TestMonkey(unittest.TestCase):

    def test_basics(self):
        monkey = Monkey()
        monkey = Monkey({'resol': 4})
        arena = Arena(resol=2)
        monkey = Monkey(arena=arena)
        self.assertIsInstance(monkey.state_space, MultiDiscrete)
        self.assertEqual(
            tuple(monkey.state_space.nvec), tuple([arena.num_tiles]*2),
        )

    def test_reset(self):
        monkey = Monkey()
        monkey.reset()
        for x in [monkey.pos, monkey.gaze]:
            self.assertGreaterEqual(x, 0)
            self.assertLess(x, monkey.arena.num_tiles)

    def test_state(self):
        monkey = Monkey()
        monkey.reset()
        pos, gaze = monkey.get_state()
        for x in [pos, gaze]:
            self.assertTrue(0<=x<monkey.arena.num_tiles)
        pos, gaze = rng.choice(monkey.arena.num_tiles, size=2)
        monkey.set_state((pos, gaze))
        self.assertEqual(monkey.pos, pos)
        self.assertEqual(monkey.gaze, gaze)

    def test_param(self):
        monkey = Monkey()
        push_cost, move_price, look_price = monkey.get_param()
        for x in [push_cost, move_price, look_price]:
            self.assertGreaterEqual(x, 0)
        push_cost, move_price, look_price = rng.uniform(0, 1, 3)
        monkey.set_param((push_cost, move_price, look_price))
        self.assertEqual(monkey.push_cost, push_cost)
        self.assertEqual(monkey.move_price, move_price)
        self.assertEqual(monkey.look_price, look_price)


if __name__=='__main__':
    unittest.main()
