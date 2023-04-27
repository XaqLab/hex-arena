import numpy as np
from scipy import stats
from gym.spaces import MultiDiscrete
from jarvis.config import Config

from typing import Optional

from . import rcParams
from .alias import Array, BoxState, EnvParam


class BaseFoodBox:
    r"""Base class for a food box with 2D color cue."""

    def __init__(self,
        *,
        dt: Optional[float] = None,
        reward: Optional[float] = None,
        num_grades: Optional[int] = None,
        num_patches: Optional[int] = None,
        sigma: Optional[float] = None,
    ):
        _rcParams = Config(rcParams.get('box.BaseFoodBox._init_'))
        self.dt = _rcParams.dt if dt is None else dt
        self.reward = _rcParams.reward if reward is None else reward
        self.num_grades = _rcParams.num_grades if num_grades is None else num_grades
        self.num_patches = _rcParams.num_patches if num_patches is None else num_patches
        self.sigma = _rcParams.sigma if sigma is None else sigma

        self.mat_size = int(self.num_patches**0.5)
        assert self.mat_size**2==self.num_patches, (
            f"`num_patches` ({self.num_patches}) must be a squre of an integer."
        )

        self.state_space: MultiDiscrete = None
        self.param_low: EnvParam = None
        self.param_high: EnvParam = None

        self.food: bool = None # food availability
        self.cue: float = None # a number in (0, 1) for color cue
        self.colors: Array = None # a 2D int array of shape (mat_size, mat_size)

        self.rng = np.random.default_rng()

    def get_param(self) -> EnvParam:
        r"""Returns box parameters."""
        raise NotImplementedError

    def set_param(self, param: EnvParam) -> None:
        r"""Sets box parameters."""
        raise NotImplementedError

    def get_state(self) -> BoxState:
        r"""Returns box state."""
        raise NotImplementedError

    def set_state(self, state: BoxState) -> None:
        r"""Sets box state."""
        raise NotImplementedError

    def render(self) -> None:
        r"""Renders color cues."""
        p = np.full((self.mat_size, self.mat_size), fill_value=self.cue)
        z = np.arctanh((p-0.5)*1.99)
        z += self.rng.normal(0, self.sigma, z.shape)
        p = (np.tanh(z)+1)/2
        self.colors = np.floor(p*self.num_grades).astype(int)

    def _reset(self) -> None:
        r"""Resets food and cue."""
        raise NotImplementedError

    def reset(self, seed: Optional[int] = None) -> None:
        r"""Resets box state.

        Args
        ----
        seed:
            If provided, reset the random number generator.

        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._reset()
        self.render()

    def _step(self, action: int) -> None:
        r"""Updates food and cue."""
        raise NotImplementedError

    def step(self, action: int) -> float:
        r"""Updates box state.

        Args
        ----
        action:
            A binary action for 'no push' (0) and 'push' (1).

        Returns
        -------
        reward:
            Food reward from the box. Action costs are not considered here.

        """
        assert action==0 or action==1
        if action==1 and self.food:
            reward = self.reward
        else:
            reward = 0.
        self._step(action)
        self.render()
        return reward


class StationaryBox(BaseFoodBox):

    def __init__(self,
        *,
        tau: Optional[float] = None,
        eps: Optional[float] = None,
        **kwargs,
    ):
        _rcParams = Config(rcParams.get('box.StationaryBox._init_'))
        super().__init__(**kwargs)
        self.tau = _rcParams.tau if tau is None else tau
        self.eps = _rcParams.eps if eps is None else eps

        # state: (food, cue)
        self.state_space = MultiDiscrete([2, self.num_grades])
        # param: (tau, sigma)
        self.param_low = [0, 0]
        self.param_high = [np.inf, np.inf]

    def get_param(self) -> EnvParam:
        r"""Returns box parameters."""
        param = (self.tau, self.sigma)
        return param

    def set_param(self, param: EnvParam) -> None:
        r"""Sets box parameters."""
        self.tau, self.sigma = param

    def get_state(self) -> BoxState:
        r"""Returns box state."""
        state = (int(self.food), int(self.cue*self.num_grades))
        return state

    def set_state(self, state: BoxState) -> None:
        r"""Sets box state."""
        self.food = bool(state[0])
        self.cue = (float(state[1])+self.rng.random())/self.num_grades

    def _reset(self):
        self.food = False
        self.cue = self.eps

    def _step(self, action: int):
        if action==0: # no push
            p = 1-np.exp(-self.dt/self.tau)
            if self.rng.random()<p:
                self.food = True
            self.cue = 1-(1-self.cue)*(1-p)
        else: # push
            self.food = False
            self.cue = self.eps


class RestorableBox(BaseFoodBox):

    def __init__(self,
        *,
        k_tau: Optional[float] = None,
        theta_tau: Optional[float] = None,
        change_rate: Optional[float] = None,
        jump_ratio: Optional[float] = None,
        restore_ratio: Optional[float] = None,
        **kwargs,
    ):
        _rcParams = Config(rcParams.get('box.RestorableBox._init_'))
        super().__init__(**kwargs)
        self.k_tau = _rcParams.k_tau if k_tau is None else k_tau
        self.theta_tau = _rcParams.theta_tau if theta_tau is None else theta_tau
        self.change_rate = _rcParams.change_rate if change_rate is None else change_rate
        self.jump_ratio = _rcParams.jump_ratio if jump_ratio is None else jump_ratio
        self.restore_ratio = _rcParams.restore_ratio if restore_ratio is None else restore_ratio

        self.tau: float = None

        # state: (food, tau)
        self.state_space = MultiDiscrete([2, self.num_grades])
        # param: (theta_tau, change_rate, jump_ratio, restore_ratio, sigma)
        self.param_low = [0, 0, 0, 0, 0]
        self.param_high = [np.inf, np.inf, np.inf, np.inf, np.inf]

    def get_param(self) -> EnvParam:
        r"""Returns box parameters."""
        param = (self.theta_tau, self.change_rate, self.jump_ratio, self.restore_ratio, self.sigma)
        return param

    def set_param(self, param: EnvParam) -> None:
        r"""Sets box parameters."""
        self.theta_tau, self.change_rate, self.jump_ratio, self.restore_ratio, self.sigma = param

    def get_state(self) -> BoxState:
        r"""Returns box state."""
        state = (int(self.food), int(self.cue*self.num_grades*0.999))
        return state

    def _tau2cue(self, tau):
        cue = 1-stats.gamma.cdf(tau, a=self.k_tau, scale=self.theta_tau)
        return cue

    def _cue2tau(self, cue):
        tau = stats.gamma.ppf(1-cue, a=self.k_tau, scale=self.theta_tau)
        return tau

    def set_state(self, state: BoxState) -> None:
        r"""Sets box state."""
        self.food = bool(state[0])
        self.cue = (float(state[1])+self.rng.random())/self.num_grades
        self.tau = self._cue2tau(self.cue)

    def _reset(self):
        self.food = False
        self.cue = self.rng.random()
        self.tau = self._cue2tau(self.cue)

    def _step(self, action: int):
        if action==0: # no push
            p = 1-np.exp(-self.dt/self.tau)
            if self.rng.random()<p:
                self.food = True
            p = 1-np.exp(-self.dt*self.change_rate)
            if self.rng.random()<p:
                new_tau = self.rng.gamma(self.k_tau, self.theta_tau)
                self.tau = np.exp(
                    (1-self.restore_ratio)*np.log(self.tau)
                    +self.restore_ratio*np.log(new_tau)
                )
        else: # push
            if not self.food:
                self.tau *= self.jump_ratio
            self.food = False
        self.cue = self._tau2cue(self.tau)


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
            `cue` and `colors`. TODO distinguish between `cue` and `colors`.
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
        self.cue = (float(state[1])+self.rng.random())/self.num_grades
