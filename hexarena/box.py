import numpy as np
from gym.spaces import MultiDiscrete
from jarvis.config import Config

from typing import Optional

from . import rcParams
from .alias import BoxObservation

class FoodBox:
    r"""Class for a food box with 2D color cue."""

    def __init__(self,
        *,
        rate: Optional[float] = None,
        step_size: Optional[float] = None,
        sigma_c: Optional[float] = None,
        food_reward: Optional[float] = None,
        num_grades: Optional[int] = None,
        array_size: Optional[int] = None,
        prob_resol: Optional[int] = None,
        eps_prob: Optional[int] = None,
    ):
        r"""
        Args
        ----
        rate:
            Poisson rate of food appears, in counts per second.
        step_size:
            Time step size for temporal discretization, in seconds.
        sigma_c:
            Parameter governing the noise of color cues, should be non-negative.
        num_grades:
            Number of distinct colors on a color map, used for discretizing the
            color cue.
        array_size:
            Characterizes spatial resolution of color cues on a screen. For
            example, if `array_size=4`, a 4*4 grid of integers will be used to
            represent the colored pattern on the screen.
        prob_resol:
            Resolution of `prob`, evenly dividing [0, 1).
        eps_prob:
            A small postive number for `prob` when there is no food.

        """
        _rcParams = Config(rcParams.get('box.Box._init_'))
        self.rate = rate or _rcParams.rate
        self.step_size = step_size or _rcParams.step_size
        self.sigma_c = sigma_c or _rcParams.sigma_c
        self.food_reward = food_reward or _rcParams.food_reward
        self.num_grades = num_grades or _rcParams.num_grades
        self.array_size = array_size or _rcParams.array_size
        self.prob_resol = prob_resol or _rcParams.prob_resol
        self.eps_prob = eps_prob or _rcParams.eps_prob

        self.state_space = MultiDiscrete([2, self.prob_resol])
        self.observation_space = MultiDiscrete([self.num_grades]*self.array_size**2)

        self.rng = np.random.default_rng()

    def _observe(self) -> BoxObservation:
        r"""Returns color cues.

        Returns
        -------
        colors: (resol*resol)
            A 2D int array containing color cues.

        """
        p = np.full((self.array_size, self.array_size), fill_value=self.prob)
        z = np.arctanh(p*2-1)
        z += self.rng.normal(0, self.sigma_c, (self.array_size, self.array_size))
        p = (np.tanh(z)+1)/2
        colors = np.floor(p*self.num_grades).astype(int)
        return colors

    def reset(self, seed: Optional[int] = None) -> tuple[BoxObservation, dict]:
        r"""Resets box state.

        Args
        ----
        seed:
            If provided, reset the random number generator.

        Returns
        -------
        observation:
            Color cue on the screen, see `_observe` for more details.
        info:
            Additional information.

        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.food = False
        self.prob = self.eps_prob
        observation = self._observe()
        info = {'food': self.food, 'prob': self.prob}
        return observation, info

    def step(self, action: int) -> tuple[BoxObservation, float, bool, bool, dict]:
        r"""Updates box state.

        Args
        ----
        action:
            A binary action for 'no push' (0) and 'push' (1).

        Returns
        -------
        observation:
            Color cue on the screen, see `_observe` for more details.
        reward:
            Food reward from the box. Action costs are not considered here.
        terminated, truncated:
            Termination flags.
        info:
            Additional information.

        """
        assert action==0 or action==1
        reward = 0.
        if action==0: # no push
            p = 1-np.exp(-self.rate*self.step_size)
            if self.rng.random()<p:
                self.food = True
            self.prob = 1-(1-self.prob)*(1-p)
        else: # push
            if self.food:
                reward += self.food_reward
            self.food = False
            self.prob = self.eps_prob
        observation = self._observe()
        info = {'food': self.food, 'prob': self.prob}
        return observation, reward, False, False, info
