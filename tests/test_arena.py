import unittest

import numpy as np
from hexarena.arena import Arena


class TestArena(unittest.TestCase):

    def test_basics(self):
        rng = np.random.default_rng()
        with self.assertRaises(Exception):
            arena = Arena(2)
        for resol in [1, 3, 5]:
            arena = Arena(resol=resol)

            r = rng.uniform(0, 3**0.5/2)
            theta = rng.uniform(0, 2*np.pi)
            x, y = r*np.cos(theta), r*np.sin(theta)
            s_idx = arena.get_state_index((x, y))
            self.assertLess(s_idx, 3*resol**2+3*resol+1)


if __name__=='__main__':
    unittest.main()
