import unittest

import numpy as np
from gym.spaces import MultiDiscrete
from hexarena.box import FoodBox
from hexarena.alias import Array


class TestFoodBox(unittest.TestCase):

    def test_basics(self):
        box = FoodBox()
        num_grades, num_patches = 6, 16
        box = FoodBox(
            rate=0.1, dt=0.5, reward=10.,
            num_grades=num_grades, num_patches=num_patches,
            sigma=0., eps=0.01,
        )
        self.assertIsInstance(box.state_space, MultiDiscrete)
        self.assertEqual(tuple(box.state_space.nvec), (2, num_grades))
        self.assertIsInstance(box.observation_space, MultiDiscrete)
        self.assertEqual(tuple(box.observation_space.nvec), tuple([num_grades]*num_patches))

    def test_reset(self):
        num_grades, num_patches = 3, 1
        box = FoodBox(num_grades=num_grades, num_patches=num_patches)
        box.reset()
        self.assertIsInstance(box.food, bool)
        self.assertIsInstance(box.cue, float)
        self.assertTrue(0<box.cue<1)
        self.assertIsInstance(box.colors, Array)
        self.assertEqual(box.colors.shape, (box.mat_size, box.mat_size))
        self.assertTrue(0<=box.colors.min()<=box.colors.max()<box.num_grades)

        # test random seed
        box = FoodBox(num_grades=100, num_patches=2500, sigma=2)
        box.reset(seed=0)
        c_0 = box.colors
        box.reset(seed=0)
        c_1 = box.colors
        box.reset(seed=1)
        c_2 = box.colors
        self.assertTrue(np.all(c_0==c_1))
        self.assertFalse(np.all(c_0==c_2))

    def test_step(self):
        box = FoodBox(num_patches=4)
        box.reset()
        for action in range(2):
            reward = box.step(action)
            self.assertIsInstance(box.colors, Array)
            self.assertEqual(box.colors.shape, (box.mat_size, box.mat_size))
            self.assertIsInstance(reward, float)
        with self.assertRaises(Exception):
            box.step(2)

    def test_state(self):
        num_grades = 10
        box = FoodBox(num_grades=num_grades)
        box.reset()
        state = box.get_state()
        self.assertEqual(len(state), 2)
        self.assertTrue(0<=state[0]<2)
        self.assertTrue(0<=state[1]<box.num_grades)
        cue = 8
        state = (0, cue)
        box.set_state(state)
        self.assertFalse(box.food)
        self.assertTrue(cue/num_grades<=box.cue<(cue+1)/num_grades)


if __name__=='__main__':
    unittest.main()
