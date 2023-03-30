import unittest

import numpy as np
from hexarena.arena import Arena


class TestArena(unittest.TestCase):

    def test_basics(self):
        rng = np.random.default_rng()
        with self.assertRaises(Exception):
            arena = Arena(1)
        for resol in [2, 4, 6]:
            arena = Arena(resol=resol)

            for x, y in [(0.5, 0), (0, 0.85)]:
                self.assertTrue(arena.is_inside((x, y)))
            for x, y in [(0.5**0.5, 0.5**0.5), (0, -0.87)]:
                self.assertFalse(arena.is_inside((x, y)))

            r = rng.uniform(0, 3**0.5/2)
            theta = rng.uniform(0, 2*np.pi)
            x, y = r*np.cos(theta), r*np.sin(theta)
            s_idx = arena.get_state_index((x, y))
            self.assertLess(s_idx, arena.num_states)


if __name__=='__main__':
    unittest.main()
