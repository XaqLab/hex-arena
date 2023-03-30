import unittest

import numpy as np
from gym.spaces import MultiDiscrete
from hexarena.box import FoodBox
from hexarena.alias import Array


class TestFoodBox(unittest.TestCase):

    def test_basics(self):
        box = FoodBox()
        num_grades, array_size, prob_resol = 6, 4, 10
        box = FoodBox(
            rate=0.1, step_size=0.5, sigma_c=0., food_reward=10.,
            num_grades=6, array_size=4, prob_resol=prob_resol,
        )
        self.assertIsInstance(box.state_space, MultiDiscrete)
        self.assertTrue(tuple(box.state_space.nvec)==(2, prob_resol))
        self.assertIsInstance(box.observation_space, MultiDiscrete)
        self.assertTrue(tuple(box.observation_space.nvec)==tuple([num_grades]*array_size**2))

    def test_reset(self):
        num_grades, resol = 3, 1
        box = FoodBox(num_grades=num_grades, array_size=resol)
        observation, info = box.reset()
        self.assertIsInstance(observation, Array)
        self.assertEqual(observation.shape, (resol, resol))
        _o = observation.reshape(-1)
        for i in range(resol**2):
            self.assertGreaterEqual(_o[i], 0)
            self.assertLess(_o[i], num_grades)
        self.assertIsInstance(info, dict)

        box = FoodBox(num_grades=100, array_size=100, sigma_c=2)
        o_0, _ = box.reset(seed=0)
        o_1, _ = box.reset(seed=0)
        o_2, _ = box.reset(seed=1)
        self.assertTrue(np.all(o_0==o_1))
        self.assertFalse(np.all(o_0==o_2))

    def test_step(self):
        resol = 2
        box = FoodBox(array_size=resol)
        box.reset()
        for action in range(2):
            observation, reward, _, _, _ = box.step(action)
            self.assertIsInstance(observation, Array)
            self.assertEqual(observation.shape, (resol, resol))
            self.assertIsInstance(reward, float)
        with self.assertRaises(Exception):
            box.step(2)


if __name__=='__main__':
    unittest.main()
