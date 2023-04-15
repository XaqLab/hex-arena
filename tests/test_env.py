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

        dt = 2.
        env = ForagingEnv(
            arena={'resol': 4},
            monkey={'push_cost': 0., 'turn_price': 0.02},
            boxes=[
                {'rate': 1/15, 'num_grades': 6, 'num_patches': 9, 'sigma': 0.01},
                {'rate': 1/21}, {'rate': 1/35},
            ],
            dt=dt,
        )
        for box in env.boxes:
            self.assertEqual(box.dt, dt)

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

    def test_param(self):
        monkey = {
            'push_cost': 0.012,
            'turn_price': 0.023,
            'move_price': 0.034,
            'look_price': 0.045,
        }
        boxes = [
            {'rate': 0.015, 'sigma': 0.01},
            {'rate': 0.025, 'sigma': 0.02},
            {'rate': 0.035, 'sigma': 0.03},
        ]
        env = ForagingEnv(
            monkey=monkey, boxes=boxes,
        )
        param = (
            monkey['push_cost'], monkey['turn_price'],
            monkey['move_price'], monkey['look_price'],
            boxes[0]['rate'], boxes[0]['sigma'],
            boxes[1]['rate'], boxes[1]['sigma'],
            boxes[2]['rate'], boxes[2]['sigma'],
        )
        self.assertEqual(param, tuple(env.get_param()))

        param = (
            0.021, 0.032, 0.043, 0.054,
            0.035, 0.03, 0.025, 0.02, 0.015, 0.01,
        )
        env.set_param(param)
        self.assertEqual(tuple(env.monkey.get_param()), param[:4])
        for i in range(3):
            self.assertEqual(tuple(env.boxes[i].get_param()), param[2*i+4:2*i+6])

        self.assertTrue(np.all(np.array(param)>np.array(env.param_low)))
        self.assertTrue(np.all(np.array(param)<np.array(env.param_high)))

if __name__=='__main__':
    unittest.main()
