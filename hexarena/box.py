import numpy as np
import torch
from scipy import stats
from gymnasium.spaces import MultiDiscrete
from irc.dist.space import DiscreteVarSpace
from irc.dist.embedder import BaseEmbedder
from irc.model import SamplingBeliefModel

from collections.abc import Sequence

from .alias import Array, Tensor, BoxState, EnvParam
from .color import get_cue_array


class BaseFoodBox:
    r"""Base class for a food box with 2D color cue.

    Args
    ----
    dt:
        Time step size for temporal discretization, in seconds.
    reward:
        Reward value of food, not considered as a parameter.
    num_levels:
        Number of box quality levels.
    kappa:
        Non-negative float for stimulus reliability, see `.color.get_cue_movie`
        for more details.
    resol:
        Color cue array resolution, `(height, width)`.

    """

    food: bool # food availability
    level: int # a number in [0, num_levels)
    colors: Array # a 2D float array of shape (height, width)

    def __init__(self,
        *,
        dt: float = 1.,
        reward: float = 10.,
        num_levels: int = 3,
        kappa: float = 0.1,
        resol: tuple[int, int] = (128, 128),
    ):
        self.dt = dt
        self.reward = reward
        self.num_levels = num_levels
        self.kappa = kappa
        self.resol = resol

        self.state_space = MultiDiscrete([2, self.num_levels]) # state: (food, level)

        self.rng = np.random.default_rng()
        self.param_names = ['kappa']

    def __repr__(self) -> str:
        return f"Box with {self.num_levels} levels"

    @property
    def spec(self) -> dict:
        return {
            'dt': self.dt, 'reward': self.reward,
            'num_levels': self.num_levels,
            'kappa': self.kappa, 'resol': self.resol,
        }

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
        if name=='kappa':
            val, low, high = [self.kappa], [0], [np.inf]
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
        if name=='kappa':
            assert len(val)==1
            self.kappa = val[0]

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

    def get_colors(self, cue: float) -> Array:
        r"""Returns the color cue array.

        Args
        ----
        cue:
            A real number in (0, 1) that determines the mean color (blue vs red)
            of the cue array.

        Returns
        -------
        colors: (height, width)
            2D float array with values on a periodic range [0, 1).

        """
        colors = get_cue_array(cue, size=self.resol, kappa=self.kappa, rng=self.rng)
        return colors

    def render(self) -> None:
        r"""Renders color cues.

        The cue value is determined based on current box state, and the cue
        array `self.colors` is updated.

        """
        cue = self.level/(self.num_levels-1)
        self.colors = self.get_colors(cue)

    def _reset(self) -> None:
        raise NotImplementedError

    def reset(self, seed: int|None = None) -> None:
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
        reward = self.reward if push and self.food else 0.
        self._step(push)
        self.render()
        return reward


class PoissonBox(BaseFoodBox):
    r"""Box with food following a Poisson process.

    Different quality levels are characterized by different parameters of
    Poisson process.

    Args
    ----
    taus:
        Time constant for Poisson process corresponding to each level.
    kwargs:
        Keyword arguments for `BaseFoodBox`.

    """

    def __init__(self,
        *,
        taus: Sequence[float]|None = None,
        **kwargs,
    ):
        if taus is not None:
            kwargs['num_levels'] = len(taus)
        super().__init__(**kwargs)
        if taus is None:
            max_tau = 36. # sec
            self.taus = np.arange(self.num_levels, 0, -1)/self.num_levels*max_tau
        else:
            assert len(taus)==self.num_levels
            self.taus = np.array([*taus])
        self.taus: Array

        self.param_names += ['taus']

    def __repr__(self) -> str:
        return "Box with taus: ({})".format(', '.join([
            '{:.2g}'.format(tau) for tau in self.taus
        ]))

    @property
    def spec(self) -> dict:
        spec = super().spec
        spec.update({
            '_target_': 'hexarena.box.PoissonBox',
            'taus': list(self.taus),
        })
        return spec

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

    def _reset(self) -> None:
        self.food = False
        self.level = 0

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

    Args
    ----
    tau:
        Time constant for every level.
    num_levels:
        Number of cue levels instead of quality levels.
    kwargs:
        Keyword arguments for `PoissonBox`.

    """

    def __init__(self,
        tau: float = 24.,
        num_levels: int = 8,
        **kwargs,
    ):
        super().__init__(taus=[tau]*num_levels, **kwargs)

    def __repr__(self) -> str:
        return "Box with tau: ({})".format(self.taus[0])

    @property
    def tau(self) -> float:
        return self.taus[0]
    @tau.setter
    def tau(self, val) -> float:
        self.taus = np.full_like(self.taus, fill_value=val)

    @property
    def spec(self) -> dict:
        spec = super().spec
        tau = float(spec.pop('taus')[0])
        spec.update({
            '_target_': 'hexarena.box.StationaryBox',
            'tau': tau,
        })
        return spec

    def _get_param(self, name: str) -> tuple[EnvParam, EnvParam, EnvParam]:
        if name=='taus':
            val = [self.tau]
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

    Args
    ----
    volatility:
        A positive number describing the level change rate.
    kwargs:
        Keyword arguments for `PoissonBox`.

    """

    def __init__(self,
        *,
        volatility: float = 0.05,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert np.all(np.diff(self.taus)<=0), "Box qualities need to be set in increasing order."
        self.volatility = volatility

        self.param_names += ['volatility']

    @property
    def spec(self) -> dict:
        spec = super().spec
        spec.update({
            '_target_': 'hexarena.box.VolatileBox',
            'volatility': self.volatility,
        })
        return spec

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

    def draw_level(self) -> None:
        self.level = self.rng.choice(self.num_levels)

    def _reset(self) -> None:
        self.food = False
        self.draw_level()

    def _step(self, push: bool) -> None:
        if push: # penalty for incorrect push
            if not self.food and self.level>0:
                self.level -= 1
        else: # box level changes randomly
            p_change = 1-np.exp(-self.dt*self.volatility)
            if self.rng.random()<p_change:
                self.draw_level()
        super()._step(push)


class GammaLinearBox(BaseFoodBox):
    r"""Box updated by Gamma schedule and uses a linear cue.

    After each button push, a reward time interval is drawn from a Gamma
    distribution, then the color cue increases linearly from 0 to 1 until the
    reward is delivered.

    Args
    ----
    tau:
        Expectation of drawn reward intervals.
    shape:
        Shape parameter of the Gamma distribution to draw reward intervals.
    max_interval:
        The upper bound of drawn interval is `max_interval*dt`.

    """

    timer: int # time since last push, [0, interval]

    def __init__(self,
        tau: float = 14.,
        shape: float = 10.,
        max_interval: int = 50,
        **kwargs,
    ):
        super().__init__(num_levels=max_interval, **kwargs)
        self.scale = tau/shape
        self.shape = shape
        assert max_interval>=stats.gamma.ppf(0.99, self.shape, scale=self.scale)/self.dt, (
            f"'max_interval' {max_interval} too small for "
            f"the Gamma distribution ({self.shape}, {self.scale})"
        )
        self.state_space = MultiDiscrete([self._state_count(max_interval)])
        self.param_names += ['tau']

    def __repr__(self) -> str:
        return f"Box with Gamma ({self.shape}, {self.scale}) schedule"

    @property
    def spec(self) -> dict:
        spec = super().spec
        spec.update({
            '_target_': 'hexarena.box.GammaLinearBox',
            'shape': self.shape, 'scale': self.scale,
        })
        return spec

    @property # max interval
    def max_interval(self) -> int:
        return self.num_levels

    @property # drawn interval, [1, max_interval]
    def interval(self) -> int:
        return self.level
    @interval.setter
    def interval(self, val):
        self.level = val

    @property
    def food(self) -> bool:
        r"""Returns food state based on current timer and level."""
        return self.timer==self.level

    def _get_param(self, name: str) -> tuple[EnvParam, EnvParam, EnvParam]:
        if name=='tau':
            val, low, high = [self.scale*self.shape], [0], [np.inf]
        else:
            val, low, high = super()._get_param(name)
        return val, low, high

    def _set_param(self, name: str, val: EnvParam) -> None:
        if name=='tau':
            assert len(val)==1
            self.scale = val[0]/self.shape
        else:
            super()._set_param(name, val)

    @staticmethod
    def _state_count(max_interval: int) -> int:
        r"""Returns total number of states up to given level.

        `timer` ranges in `[0, interval]` and `interval` ranges in `[1, max_interval]`.

        """
        return max_interval*(max_interval+3)//2

    @staticmethod
    def _sub2idx(interval: int, timer: int) -> int:
        r"""Converts tuple state to index.

        Args
        ----
        interval:
            Current drawn interval, in `[1, max_interval]`.
        timer:
            Current timer, ranging in `[0, level]`.

        """
        return GammaLinearBox._state_count(interval-1)+timer

    @staticmethod
    def _idx2sub(state_idx: int) -> tuple[int, int]:
        r"""Converts state index to tuple.

        Args
        ----
        state_idx:
            An integer in `[0, state_count(num_levels))`.

        Returns
        -------
        interval, timer:
            Current drawn interval and timer corresponding to `state_idx`.

        """
        interval = 1
        while GammaLinearBox._state_count(interval)<=state_idx:
            interval += 1
        timer = state_idx-GammaLinearBox._state_count(interval-1)
        return interval, timer

    def get_state(self) -> BoxState:
        r"""Returns box state.

        Box state is the index of the ordered sequence of all possible
        `(interval, timer)` tuples.

        """
        state = (self._sub2idx(self.interval, self.timer),)
        return state

    def set_state(self, state: BoxState) -> None:
        r"""Sets box state."""
        self.interval, self.timer = self._idx2sub(state[0])

    def render(self) -> None:
        cue = self.timer/self.interval # color cue marks the progress towards reward
        self.colors = self.get_colors(cue)

    def _reset(self) -> None:
        r"""Draws new reward interval from a Gamma distribution."""
        interval = int(np.ceil(self.rng.gamma(self.shape, self.scale)/self.dt))
        self.interval = min(interval, self.max_interval)
        self.timer = 0

    def _step(self, push: bool) -> None:
        if push:
            self._reset()
        else:
            self.timer = min(self.timer+1, self.interval)


class LinearBoxStateEmbedder(BaseEmbedder):
    r"""State embedder for linear color box state.

    The box state is two integers `(interval, timer)`. Instead of encoding them
    by huge one-hot vector(s), the embedder directly uses two float numbers as
    feature, the scaled interval `interval/20` and progress `timer/interval`.

    Args
    ----
    spaces:
        A list containing only one `DiscreteVarSpace`, which is derived from
        `GammaLinearBox` state space.

    """

    def __init__(self, spaces: list[DiscreteVarSpace]):
        assert len(spaces)==1 and isinstance(spaces[0], DiscreteVarSpace)

        max_interval = 0
        while GammaLinearBox._state_count(max_interval)<spaces[0].n:
            max_interval += 1
        assert GammaLinearBox._state_count(max_interval)==spaces[0].n, (
            f"Invalid space dimension ({spaces[0].n})"
        )
        self.max_interval = max_interval
        super().__init__(spaces)
        self.feat_dim = 2 # (level, timer)

    def __repr__(self) -> str:
        return f"Embedder for linear box with max interval {self.max_interval}."

    @property
    def spec(self) -> dict:
        spec = super().spec
        spec.update({
            '_target_': 'hexarena.box.LinearBoxStateEmbedder',
        })
        return spec

    def embed(self, xs: Tensor) -> Tensor:
        feats = []
        for x in xs:
            interval, timer = GammaLinearBox._idx2sub(int(x.item()))
            feats.append((interval/20, timer/interval))
        feats = torch.tensor(feats, dtype=torch.float, device=xs.device)
        return feats

def belief2probs(
    model: SamplingBeliefModel, beliefs: Tensor,
):
    r"""Converts belief vector to state probabilities of each box."""
    n_boxes = model.env.num_boxes
    n_levels = None
    for i in range(n_boxes):
        if n_levels is None:
            n_levels = model.env.boxes[i].num_levels
        else:
            assert n_levels==model.env.boxes[i].num_levels
    assert isinstance(model.env.boxes[0], GammaLinearBox)
    n_samples, belief_dim = beliefs.shape
    param_dim = belief_dim//n_boxes
    p_boxes = np.zeros((n_samples, n_boxes, n_levels, n_levels+1)) # P(interval, timer) for all boxes
    for i in range(n_boxes):
        p_s = model.p_s.s_dists[i]
        all_xs = p_s.all_xs
        for t in range(n_samples):
            param_vec = beliefs[t, i*param_dim:(i+1)*param_dim]
            logps, _ = p_s.loglikelihoods(all_xs, param_vec)
            for k in range(len(all_xs)):
                interval, timer = model.env.boxes[i]._idx2sub(k)
                p_boxes[t, i, interval-1, timer] = logps[k].exp().item()
    return p_boxes
