import numpy as np
from scipy import stats
from gym.spaces import MultiDiscrete
from jarvis.config import Config

from typing import Optional

from . import rcParams
from .alias import Array, BoxState, EnvParam


class BaseFoodBox:
    r"""Base class for a food box with 2D color cue."""

    food: bool # food availability
    cue: float # a number in (0, 1)
    colors: Array # a 2D int array of shape (mat_size, mat_size)

    param_low: EnvParam
    param_high: EnvParam

    def __init__(self,
        *,
        dt: Optional[float] = None,
        reward: Optional[float] = None,
        num_levels: Optional[int] = None,
        num_grades: Optional[int] = None,
        num_patches: Optional[int] = None,
        sigma: Optional[float] = None,
    ):
        r"""
        Args
        ----
        dt:
            Time step size for temporal discretization, in seconds.
        reward:
            Reward value of food.
        num_levels:
            Number of levels to discretize `cue`.
        num_grades:
            Number of distinct colors on a color map for discrete values of
            `colors`.
        num_patches:
            Number of colored patches on the screen, must be a square of an
            integer to represent a square matrix. For example, if
            `num_patches=16`, a 4*4 grid of integers will be used to represent
            the color pattern on the screen.
        sigma:
            Parameter governing the noise of color cues, should be non-negative.

        """
        _rcParams = rcParams.get('box.BaseFoodBox._init_', {})
        self.dt = _rcParams.dt if dt is None else dt
        self.reward = _rcParams.reward if reward is None else reward
        self.num_levels = _rcParams.num_levels if num_levels is None else num_levels
        self.num_grades = _rcParams.num_grades if num_grades is None else num_grades
        self.num_patches = _rcParams.num_patches if num_patches is None else num_patches
        self.sigma = _rcParams.sigma if sigma is None else sigma

        self.mat_size = int(self.num_patches**0.5)
        assert self.mat_size**2==self.num_patches, (
            f"`num_patches` ({self.num_patches}) must be a squre of an integer."
        )

        self.state_space = MultiDiscrete([2, self.num_levels]) # state: (food, cue)
        self.rng = np.random.default_rng()

    def get_param(self) -> EnvParam:
        r"""Returns box parameters."""
        raise NotImplementedError

    def set_param(self, param: EnvParam) -> None:
        r"""Sets box parameters."""
        raise NotImplementedError

    def get_state(self) -> BoxState:
        r"""Returns box state."""
        state = (int(self.food), int(self.cue*self.num_levels))
        return state

    def set_state(self, state: BoxState) -> None:
        r"""Sets box state."""
        self.food = bool(state[0])
        self.cue = (float(state[1])+self.rng.random())/self.num_levels

    def render(self) -> None:
        r"""Renders color cues."""
        p = np.full((self.mat_size, self.mat_size), fill_value=self.cue)
        z = np.arctanh((p-0.5)*1.99)
        z += self.rng.normal(0, self.sigma, z.shape)
        p = (np.tanh(z)+1)/2
        self.colors = np.floor(p*self.num_grades).astype(int)

    def _reset(self) -> None:
        r"""Resets food and cue."""
        self.food = False
        self.cue = 0.

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

    def _step(self, push: bool) -> None:
        r"""Updates food and cue."""
        raise NotImplementedError

    def step(self, push: bool) -> float:
        r"""Updates box state.

        Args
        ----
        push:
            Whether to push the button of food box.

        Returns
        -------
        reward:
            Food reward from the box. Action costs are not considered here.

        """
        if push and self.food:
            reward = self.reward
        else:
            reward = 0.
        self._step(push)
        self.render()
        return reward


class StationaryBox(BaseFoodBox):
    r"""Food box with a fixed appear rate.

    Food appears according to a Poisson process with fixed rate. The color cue
    represents the cumulative probability of food being available, essentially
    a timer.

    """

    def __init__(self,
        *,
        tau: Optional[float] = None,
        **kwargs,
    ):
        r"""
        Args
        ----
        tau:
            Time constant of Poisson process. An time interval is drawn from the
            exponential distribution parameterized by `tau` after each push.
        kwargs:
            Additional keyword arguments for `BaseFoodBox`.

        """
        _rcParams = rcParams.get('box.StationaryBox._init_', {})
        super().__init__(**kwargs)
        self.tau = _rcParams.tau if tau is None else tau

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

    def _step(self, push: bool):
        if push:
            self.food = False
            self.cue = 0.
        else:
            p_appear = 1-np.exp(-self.dt/self.tau)
            if self.rng.random()<p_appear:
                self.food = True
            self.cue = 1-(1-self.cue)*(1-p_appear)


class RestorableBox(BaseFoodBox):
    r"""Food box with time-varying appear rate.

    Food appears according to a Poisson process with temporal parameter `tau`,
    but `tau` itself can change over time. Controlled by another Poisson
    process, a new `tau` value will be drawn from a default distribution, and
    averaged with current value in a weighted manner. In addition, `tau` will
    increase when a incorret push is made.

    Color cue now explicitly encodes `tau` with noise controlled by `sigma`.

    """

    tau: float

    def __init__(self,
        *,
        k_tau: Optional[float] = None,
        theta_tau: Optional[float] = None,
        change_rate: Optional[float] = None,
        restore_ratio: Optional[float] = None,
        jump_ratio: Optional[float] = None,
        **kwargs,
    ):
        r"""
        Args
        ----
        k_tau, theta_tau:
            Parameters of a gamma distribution, serving as the default
            distribution of redrawing `tau`.
        change_rate:
            Temporal parameter controlling the update of `tau`, in the same way
            as `tau` controlling the update of `food`.
        restore_ratio:
            Parameter controlling the average of current `tau` and the newly
            drawn `tau`. 0 means stationary food rate, and 1 means no temporal
            smoothing.
        jump_ratio:
            Parameter controlling the penalty of incorrect push, `tau` will
            increase by `jump_ratio` in such cases.

        """
        _rcParams = Config(rcParams.get('box.RestorableBox._init_'))
        super().__init__(**kwargs)
        self.k_tau = _rcParams.k_tau if k_tau is None else k_tau
        self.theta_tau = _rcParams.theta_tau if theta_tau is None else theta_tau
        self.change_rate = _rcParams.change_rate if change_rate is None else change_rate
        self.restore_ratio = _rcParams.restore_ratio if restore_ratio is None else restore_ratio
        self.jump_ratio = _rcParams.jump_ratio if jump_ratio is None else jump_ratio

        # param: (theta_tau, change_rate, restore_ratio, jump_ratio, sigma)
        self.param_low = [0, 0, 0, 1, 0]
        self.param_high = [np.inf, np.inf, 1, np.inf, np.inf]

    def get_param(self) -> EnvParam:
        r"""Returns box parameters."""
        param = (self.theta_tau, self.change_rate, self.jump_ratio, self.restore_ratio, self.sigma)
        return param

    def set_param(self, param: EnvParam) -> None:
        r"""Sets box parameters."""
        self.theta_tau, self.change_rate, self.jump_ratio, self.restore_ratio, self.sigma = param

    def _tau2cue(self, tau):
        cue = 1-stats.gamma.cdf(tau, a=self.k_tau, scale=self.theta_tau)
        return cue

    def _cue2tau(self, cue):
        tau = stats.gamma.ppf(1-cue, a=self.k_tau, scale=self.theta_tau)
        return tau

    def set_state(self, state: BoxState) -> None:
        r"""Sets box state."""
        super().set_state(state)
        self.tau = self._cue2tau(self.cue)

    def _reset(self):
        self.food = False
        self.tau = self.rng.gamma(self.k_tau, self.theta_tau)
        self.cue = self._tau2cue(self.tau)

    def _step(self, push: bool):
        if push:
            if not self.food:
                self.tau *= self.jump_ratio
            self.food = False
        else:
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
        self.cue = self._tau2cue(self.tau)
