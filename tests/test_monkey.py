import unittest

from gym.spaces import MultiDiscrete
from hexarena.arena import Arena
from hexarena.monkey import Monkey


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

if __name__=='__main__':
    unittest.main()
