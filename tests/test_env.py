import unittest

import numpy as np
from gym.spaces import MultiDiscrete
from irc.utils import check_env
from hexarena.env import ForagingEnv


class TestForagingEnv(unittest.TestCase):

    def test_basics(self):
        env = ForagingEnv()
        check_env(env)
        self.assertIsInstance(env.state_space, MultiDiscrete)
        self.assertIsInstance(env.observation_space, MultiDiscrete)

        env = ForagingEnv(
            arena={'resol': 4},
            boxes=[
                {'rate': 1/15, 'num_grades': 6, 'num_patches': 9, 'sigma': 0.01},
                {'rate': 1/21}, {'rate': 1/35},
            ],
            monkey={'push_cost': 0., 'turn_price': 0.02},
            dt=2.,
        )

    def test_reset(self):
        env = ForagingEnv()
        observation, info = env.reset()
        self.assertEqual(len(observation), len(env.observation_space.nvec))
        for i in range(len(observation)):
            self.assertTrue(0<=observation[i]<env.observation_space.nvec[i])
        self.assertIsInstance(info, dict)
        o_0, _ = env.reset(seed=0)
        o_1, _ = env.reset(seed=0)
        o_2, _ = env.reset(seed=1)
        self.assertEqual(tuple(o_0), tuple(o_1))
        self.assertNotEqual(tuple(o_0), tuple(o_2))

    def test_step(self):
        env = ForagingEnv()
        env.reset()
        for _ in range(10):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            self.assertEqual(len(observation), len(env.observation_space.nvec))
            self.assertTrue(np.isscalar(reward))
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertIsInstance(info, dict)


if __name__=='__main__':
    unittest.main()
