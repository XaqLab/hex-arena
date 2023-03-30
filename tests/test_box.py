import unittest

from hexarena.box import FoodBox
from hexarena.alias import Array


class TestFoodBox(unittest.TestCase):

    def test_basics(self):
        box = FoodBox()
        box = FoodBox(rate=15)
        box = FoodBox(step_size=0.5)
        box = FoodBox(sigma_c=0.)
        box = FoodBox(num_grades=6)
        box = FoodBox(resol=6)

    def test_reset(self):
        num_grades, resol = 3, 1
        box = FoodBox(num_grades=num_grades, resol=resol)
        observation, info = box.reset()
        self.assertIsInstance(observation, Array)
        self.assertEqual(observation.shape, (resol, resol))
        _o = observation.reshape(-1)
        for i in range(resol**2):
            self.assertGreaterEqual(_o[i], 0)
            self.assertLess(_o[i], num_grades)
        self.assertIsInstance(info, dict)

    def test_step(self):
        ...


if __name__=='__main__':
    unittest.main()
