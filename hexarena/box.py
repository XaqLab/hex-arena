import numpy as np
from scipy import stats
from gymnasium.spaces import MultiDiscrete

from typing import Optional
from collections.abc import Collection

from hexarena.alias import EnvParam

from . import rcParams
from .alias import Array, BoxState, EnvParam


class BaseFoodBox:
    r"""Base class for a food box with 2D color cue."""

    food: bool # food availability
    level: int # a number in [0, num_levels)
    colors: Array # a 2D int array of shape (mat_size, mat_size)

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
            Reward value of food, not considered as a parameter.
        num_levels:
            Number of box quality levels.
        num_grades:
            Number of distinct colors on a color map for discrete values of
            `colors`.
        num_patches:
            Number of colored patches on the screen, must be a square of an
            integer to represent a square matrix. For example, if
            `num_patches=16`, a 4*4 grid of integers will be used to represent
            the color pattern on the screen.
        sigma:
            Parameter governing the noise of color cues, should be in (0, 0.5).
            Color cues are drawn from a beta distribution of which the variance
            is determined by sigma. See `render` for more details.

        """
        _rcParams = rcParams.get('box.BaseFoodBox._init_', {})
        self.dt: float = _rcParams.dt if dt is None else dt
        self.reward: float = _rcParams.reward if reward is None else reward
        self.num_levels: int = _rcParams.num_levels if num_levels is None else num_levels
        self.num_grades: int = _rcParams.num_grades if num_grades is None else num_grades
        self.num_patches: int = _rcParams.num_patches if num_patches is None else num_patches
        self.sigma: float = _rcParams.sigma if sigma is None else sigma

        self.mat_size = int(self.num_patches**0.5)
        assert self.mat_size**2==self.num_patches, (
            f"`num_patches` ({self.num_patches}) must be a squre of an integer."
        )

        self.state_space = MultiDiscrete([2, self.num_levels]) # state: (food, level)
        self.rng = np.random.default_rng()
        self.param_names = ['sigma']

    def __repr__(self) -> str:
        return f"Box with {self.num_levels} levels and {self.num_grades} color grades"

    def _get_param(self, name: str) -> tuple[EnvParam, EnvParam, EnvParam]:
        r"""Returns value and bounds of a named parameter.

        Args
        ----
        name:
            Parameter name.

        Returns
        -------
        val, low, high:
            List of floats representing the current value, lower and upper
            bounds of the parameter.

        """
        if name=='sigma':
            val, low, high = [self.sigma], [0], [0.5]
        return val, low, high

    def _set_param(self, name: str, val: EnvParam) -> None:
        r"""Sets value of a named parameter.

        Args
        ----
        name:
            Parameter name.
        val:
            Parameter values.

        """
        if name=='sigma':
            assert len(val)==1
            self.sigma = val[0]

    def get_param(self) -> EnvParam:
        r"""Returns box parameters."""
        param = []
        for name in self.param_names:
            val, *_ = self._get_param(name)
            param += val
        return param

    def set_param(self, param: EnvParam) -> None:
        r"""Sets box parameters."""
        c = 0
        for name in self.param_names:
            val, low, high = self._get_param(name)
            n = len(val)
            val = param[c:c+n]
            for i in range(n):
                assert low[i]<val[i]<high[i]
            self._set_param(name, val)
            c += n

    def param_bounds(self) -> tuple[EnvParam, EnvParam]:
        r"""Returns lower and upper bound of box parameters."""
        low, high = [], []
        for name in self.param_names:
            _, _low, _high = self._get_param(name)
            low += [*_low]
            high += [*_high]
        return low, high

    def get_state(self) -> BoxState:
        r"""Returns box state."""
        state = (int(self.food), self.level)
        return state

    def set_state(self, state: BoxState) -> None:
        r"""Sets box state."""
        self.food = bool(state[0])
        self.level = state[1]

    def render(self) -> None:
        r"""Renders color cues.

        Colors are drawn from a discretized beta distribution, of which the mean
        is determined by `level`.

        """
        cue = (self.level+0.5)/self.num_levels
        rv = stats.beta(a=cue/self.sigma, b=(1-cue)/self.sigma)
        self.colors = self.rng.choice(
            self.num_grades, size=(self.mat_size, self.mat_size),
            p=np.diff(rv.cdf(np.linspace(0, 1, self.num_grades+1))),
        )

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
        self.level = self.rng.choice(self.num_levels)
        self.render()

    def _step(self, push: bool) -> None:
        r"""Updates food and level."""
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


class PoissonBox(BaseFoodBox):
    r"""Box with food following a Poisson process.

    Different quality levels are characterized by different parameters of
    Poisson process.

    """

    def __init__(self,
        *,
        taus: Optional[Collection[float]] = None,
        **kwargs,
    ):
        r"""
        Args
        ----
        taus:
            Time constant for Poisson process corresponding to each level.
        kwargs:
            Keyword arguments for `BaseFoodBox`.

        """
        _rcParams = rcParams.get('box.PoissonBox._init_', {})
        if taus is not None:
            kwargs['num_levels'] = len(taus)
        super().__init__(**kwargs)
        if taus is None:
            self.taus = (np.arange(self.num_levels)+1)/self.num_levels*_rcParams.max_tau
        else:
            assert len(taus)==self.num_levels
            self.taus = np.array([*taus])
        self.taus: Array

        self.param_names += ['taus']

    def __repr__(self) -> str:
        return "Box with taus: ({})".format(', '.join([
            '{:.2g}'.format(tau) for tau in self.taus
        ]))

    def _get_param(self, name: str) -> tuple[EnvParam, EnvParam, EnvParam]:
        if name=='taus':
            val = [*self.taus]
            low = [0]*self.num_levels
            high = [np.inf]*self.num_levels
        else:
            val, low, high = super()._get_param(name)
        return val, low, high

    def _set_param(self, name: str, val: EnvParam) -> None:
        if name=='taus':
            assert len(val)==self.num_levels
            self.taus = np.array([*val])
        else:
            super()._set_param(name, val)

    def _step(self, push: bool) -> None:
        if push:
            self.food = False
        else:
            p_appear = 1-np.exp(-self.dt/self.taus[self.level])
            if self.rng.random()<p_appear:
                self.food = True


class StationaryBox(PoissonBox):
    r"""Box with a fixed quality.

    Box level represents the probability of food presence instead of food appear
    rate.

    """

    def __init__(self,
        tau: Optional[float] = None,
        num_levels: Optional[int] = None,
        **kwargs,
    ):
        r"""
        Args
        ----
        tau:
            Time constant for every level.
        num_levels:
            Number of cue levels instead of quality levels.
        kwargs:
            Keyword arguments for `PoissonBox`.

        """
        _rcParams = rcParams.get('box.StationaryBox._init_', {})
        tau = _rcParams.tau if tau is None else tau
        if num_levels is None:
            num_levels = _rcParams.num_levels
        super().__init__(taus=[tau]*num_levels, **kwargs)

    def __repr__(self) -> str:
        return "Box with tau: ({})".format(self.taus[0])

    def _get_param(self, name: str) -> tuple[EnvParam, EnvParam, EnvParam]:
        if name=='taus':
            val = [self.taus[0]]
            low = [0]
            high = [np.inf]
        else:
            val, low, high = super()._get_param(name)
        return val, low, high

    def _set_param(self, name: str, val: EnvParam) -> None:
        if name=='taus':
            assert len(val)==1
            super()._set_param('taus', [val[0]]*self.num_levels)
        else:
            super()._set_param(name, val)

    def _step(self, push: bool) -> None:
        super()._step(push)
        if push: # reset to lowest level
            self.level = 0
        else: # level increase approximately according to cumulative probability
            p = self.rng.uniform(self.level/self.num_levels, (self.level+1)/self.num_levels)
            p = 1-(1-p)*np.exp(-self.dt/self.taus[0])
            self.level = min(int(np.floor(p*self.num_levels)), self.num_levels-1)


class VolatileBox(PoissonBox):
    r"""Box with volatile qualities.

    Box level will change to a random value, with the change happening according
    to a Poisson process. Incorrect push will lead to a level decrease if
    possible.

    """

    def __init__(self,
        *,
        volatility: Optional[float] = None,
        **kwargs,
    ):
        r"""
        Args
        ----
        volatility:
            A positive number describing the level change rate.
        kwargs:
            Keyword arguments for `PoissonBox`.

        """
        _rcParams = rcParams.get('box.VolatileBox._init_', {})
        super().__init__(**kwargs)
        assert np.all(np.diff(self.taus)<=0), "Box qualities need to be set in increasing order."
        self.volatility = _rcParams.volatility if volatility is None else volatility

        self.param_names += ['volatility']

    def _get_param(self, name: str) -> tuple[EnvParam, EnvParam, EnvParam]:
        if name=='volatility':
            val, low, high = [self.volatility], [0], [np.inf]
        else:
            val, low, high = super()._get_param(name)
        return val, low, high

    def _set_param(self, name: str, val: EnvParam) -> None:
        if name=='volatility':
            assert len(val)==1
            self.volatility = val[0]
        else:
            super()._set_param(name, val)

    def _step(self, push: bool) -> None:
        if push: # penalty for incorrect push
            if not self.food and self.level>0:
                self.level -= 1
        else: # box level changes randomly
            p_change = 1-np.exp(-self.dt*self.volatility)
            if self.rng.random()<p_change:
                self.level = self.rng.choice(self.num_levels)
        super()._step(push)
