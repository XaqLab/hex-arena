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
            self.assertEqual(len(arena.corners), 6)
            self.assertEqual(len(arena.boxes), 3)
            self.assertEqual(arena.center, 0)
            self.assertEqual(len(arena.inners), 3*resol**2-3*resol)
            self.assertEqual(len(arena.outers), 6*resol)

    def test_is_inside(self):
        arena = Arena()
        for x, y in [(0.5, 0), (0, 0.85)]:
            self.assertTrue(arena.is_inside((x, y)))
        for x, y in [(0.5**0.5, 0.5**0.5), (0, -0.87)]:
            self.assertFalse(arena.is_inside((x, y)))

    def test_get_tile_index(self):
        arena = Arena()
        r = rng.uniform(0, 1)
        theta = rng.uniform(0, 2*np.pi)
        x, y = r*np.cos(theta), r*np.sin(theta)
        s_idx = arena.nearest_tile((x, y))
        self.assertLess(s_idx, arena.num_tiles)


if __name__=='__main__':
    unittest.main()
