import unittest

from hexarena.arena import Arena


class TestArena(unittest.TestCase):

    def test_basics(self):
        for resol in [1, 2, 4]:
            arena = Arena(resol=resol)


if __name__=='__main__':
    unittest.main()
