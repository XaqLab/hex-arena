import unittest

from hexarena.box import Box


class TestBox(unittest.TestCase):

    def test_basics(self):
        box = Box()
        box = Box(rate=15)
        box = Box(step_size=0.5)
        box = Box(sigma_c=0.)
        box = Box(num_grades=6)
        box = Box(resol=6)


if __name__=='__main__':
    unittest.main()
