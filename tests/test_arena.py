import unittest

import numpy as np
from hexarena.arena import Arena

rng = np.random.default_rng()


class TestArena(unittest.TestCase):

    def test_basics(self):
        with self.assertRaises(Exception):
            arena = Arena(1)
        for resol in [2, 4, 6]:
            arena = Arena(resol=resol)

    def test_is_inside(self):
        arena = Arena()
        for x, y in [(0.5, 0), (0, 0.85)]:
            self.assertTrue(arena.is_inside((x, y)))
        for x, y in [(0.5**0.5, 0.5**0.5), (0, -0.87)]:
            self.assertFalse(arena.is_inside((x, y)))

    def test_get_state_index(self):
        arena = Arena()
        for _ in range(5):
            r = rng.uniform(0, 1)
            theta = rng.uniform(0, 2*np.pi)
            x, y = r*np.cos(theta), r*np.sin(theta)
            if arena.is_inside((x, y)):
                s_idx = arena.get_state_index((x, y))
                self.assertLess(s_idx, arena.num_states)
            else:
                with self.assertRaises(Exception):
                    arena.get_state_index((x, y))


if __name__=='__main__':
    unittest.main()
