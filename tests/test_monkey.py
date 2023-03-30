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
        state = monkey.get_state()
        self.assertEqual(len(state), 2)
        for i in range(2):
            self.assertTrue(0<=state[i]<monkey.arena.num_tiles)
        pos, gaze = rng.choice(monkey.arena.num_tiles, size=2)
        monkey.set_state((pos, gaze))
        self.assertEqual(monkey.pos, pos)
        self.assertEqual(monkey.gaze, gaze)


if __name__=='__main__':
    unittest.main()
