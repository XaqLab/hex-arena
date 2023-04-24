import unittest

import numpy as np
from hexarena.box import StationaryBox
from hexarena.alias import Array


class TestBaseBox(unittest.TestCase):
    box_classes = [StationaryBox]

    def test_basics(self):
        for Box in self.box_classes:
            box = Box(
                dt=1., reward=20., num_grades=8, num_patches=9,
            )

            param = box.get_param()
            self.assertTrue(np.all(np.array(param)>np.array(box.param_low)))
            self.assertTrue(np.all(np.array(param)<np.array(box.param_high)))
            box.set_param(param)

            box.reset()
            state = box.get_state()
            self.assertEqual(len(state), len(box.state_space.nvec))
            for i in range(len(state)):
                self.assertIsInstance(state[i], int)
                self.assertGreaterEqual(state[i], 0)
                self.assertLess(state[i], box.state_space.nvec[i])
            box.set_state(state)

    def test_reset(self):
        num_grades = 100
        num_patches = 2500
        sigma = 2
        for Box in self.box_classes:
            box = Box(num_grades=num_grades, num_patches=num_patches, sigma=sigma)
            box.reset()
            self.assertIsInstance(box.food, bool)
            self.assertIsInstance(box.cue, float)
            self.assertTrue(0<box.cue<1)
            self.assertIsInstance(box.colors, Array)
            self.assertEqual(box.colors.shape, (box.mat_size, box.mat_size))
            self.assertTrue(0<=box.colors.min()<=box.colors.max()<box.num_grades)

            # test random seed
            box.reset(seed=0)
            c_0 = box.colors
            box.reset(seed=0)
            c_1 = box.colors
            box.reset(seed=1)
            c_2 = box.colors
            self.assertTrue(np.all(c_0==c_1))
            self.assertFalse(np.all(c_0==c_2))

    def test_step(self):
        reward = 8.765
        for Box in self.box_classes:
            box = Box(reward=reward)
            box.reset()
            for action in range(2):
                self.assertIsInstance(box.step(action), float)
                self.assertIsInstance(box.colors, Array)
                self.assertEqual(box.colors.shape, (box.mat_size, box.mat_size))
            with self.assertRaises(Exception):
                box.step(2)
            box.food = True
            self.assertEqual(box.step(1), reward)
            box.food = False
            self.assertEqual(box.step(1), 0)


if __name__=='__main__':
    unittest.main()
