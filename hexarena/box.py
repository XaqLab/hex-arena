import numpy as np
from gym.spaces import MultiDiscrete
from jarvis.config import Config

from typing import Optional

from . import rcParams
from .alias import BoxState, EnvParam

class FoodBox:
    r"""Class for a food box with 2D color cue."""

    def __init__(self,
        *,
        rate: Optional[float] = None,
        dt: Optional[float] = None,
        reward: Optional[float] = None,
        num_grades: Optional[int] = None,
        num_patches: Optional[int] = None,
        sigma: Optional[float] = None,
        eps: Optional[int] = None,
    ):
        r"""
        Args
        ----
        rate:
            Poisson rate of food appears, in counts per second.
        dt:
            Time step size for temporal discretization, in seconds.
        reward:
            Reward value of food.
        num_grades:
            Number of distinct colors on a color map, used for discretizing
            `cue` and `colors`.
        num_patches:
            Number of colored patches on the screen, must be a square of an
            integer to represent a square matrix. For example, if
            `num_patches=16`, a 4*4 grid of integers will be used to represent
            the color pattern on the screen.
        sigma:
            Parameter governing the noise of color cues, should be non-negative.
        eps:
            A small postive number for `cue` when there is no food.

        """
        _rcParams = Config(rcParams.get('box.Box._init_'))
        self.rate = _rcParams.rate if rate is None else rate
        self.dt = _rcParams.dt if dt is None else dt
        self.reward = _rcParams.reward if reward is None else reward
        self.num_grades = _rcParams.num_grades if num_grades is None else num_grades
        self.num_patches = _rcParams.num_patches if num_patches is None else num_patches
        self.sigma = _rcParams.sigma if sigma is None else sigma
        self.eps = _rcParams.eps if eps is None else eps

        self.mat_size = int(self.num_patches**0.5)
        assert self.mat_size**2==self.num_patches, (
            f"`num_patches` ({self.num_patches}) must be a squre of an integer."
        )

        # state: (food, cue)
        self.state_space = MultiDiscrete([2, self.num_grades])
        # param: (rate, sigma)
        self.param_low = [0, 0]
        self.param_high = [np.inf, np.inf]

        self.rng = np.random.default_rng()

    def render(self) -> None:
        r"""Returns color cues.

        Returns
        -------
        colors: (resol*resol)
            A 2D int array containing color cues.

        """
        p = np.full((self.mat_size, self.mat_size), fill_value=self.cue)
        z = np.arctanh(p*2-1)
        z += self.rng.normal(0, self.sigma, z.shape)
        p = (np.tanh(z)+1)/2
        self.colors = np.floor(p*self.num_grades).astype(int)

    def reset(self, seed: Optional[int] = None) -> None:
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
        self.cue = self.eps
        self.render()

    def step(self, action: int) -> float:
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
            p = 1-np.exp(-self.rate*self.dt)
            if self.rng.random()<p:
                self.food = True
            self.cue = 1-(1-self.cue)*(1-p)
        else: # push
            if self.food:
                reward += self.reward
            self.food = False
            self.cue = self.eps
        self.render()
        return reward

    def get_param(self) -> EnvParam:
        r"""Returns box parameters."""
        param = (self.rate, self.sigma)
        return param

    def set_param(self, param: EnvParam) -> None:
        r"""Sets box parameters."""
        self.rate, self.sigma = param

    def get_state(self) -> BoxState:
        r"""Returns box state."""
        state = (int(self.food), int(self.cue*self.num_grades))
        return state

    def set_state(self, state: BoxState) -> None:
        r"""Sets box state."""
        self.food = bool(state[0])
        self.cue = (float(state[1])+0.5)/self.num_grades
