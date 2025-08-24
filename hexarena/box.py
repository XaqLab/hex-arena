import numpy as np
from gymnasium.spaces import Discrete, Box, Dict
from jarvis.utils import cls_name

from .alias import Array, EnvParam
from .color import get_cue_array


class BaseFoodBox:
    r"""Base class for a food box with 2D color cue.

    Args
    ----
    dt:
        Time step size for temporal discretization, in seconds.
    reward:
        Reward value of food, not considered as a parameter.
    kappa:
        Non-negative float for stimulus reliability, see `.color.get_cue_movie`
        for more details.
    resol:
        Color cue array resolution, `(height, width)`.

    """

    food: bool # food availability
    cue: float # scalar in [0, 1] to generate color array
    # colors: Array # a 2D float array of shape (height, width)
    # param_names: list[str] # list of parameter names

    pos: int # position on arena

    def __init__(self,
        tau: float,
        *,
        cue_in_state: bool = True,
        tau_in_state: bool = False,
        dt: float = 1.,
        reward: float = 10.,
        kappa: float = 0.1,
        resol: tuple[int, int] = (128, 96),
    ):
        self.tau = tau
        self.cue_in_state = cue_in_state
        self.tau_in_state = tau_in_state
        self.dt = dt
        self.reward = reward

        self.kappa = kappa
        self.resol = resol

        if self.tau_in_state:
            self.state_space = Dict({'tau': Box(0, np.inf)})
        else:
            self.state_space = Dict({})

        self.rng = np.random.default_rng()
        self.param_names = []
        if self.cue_in_state:
            self.param_names.append('kappa')
        if self.tau_in_state:
            self.param_names.append('tau')

    def __repr__(self) -> str:
        r_strs = [f'{key}={val}' for key, val in self.spec.items() if key!='_target_']
        return "{}(\n{},\n)".format(self.__class__.__name__, ',\n'.join(r_strs))

    @property
    def spec(self) -> dict:
        return {
            '_target_': cls_name(self),
            'cue_in_state': self.cue_in_state, 'tau_in_state': self.tau_in_state,
            'tau': self.tau, 'dt': self.dt, 'reward': self.reward,
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
        if name=='tau':
            val, low, high = [self.tau], [0], [np.inf]
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
        if name=='tau':
            assert len(val)==1
            self.tau = val[0]

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

    def get_state(self) -> dict[str, int|Array]:
        r"""Returns box state."""
        raise NotImplementedError

    def set_state(self, state: dict[str, int|Array]) -> None:
        r"""Sets box state."""
        raise NotImplementedError

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
        r"""Renders color cues."""
        self.colors = self.get_colors(self.cue)

    def _reset(self) -> None:
        r"""Resets box state with existing random number generator."""
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
        r"""Updates box state after reward outcome is dealt with.

        Args
        ----
        push:
            Whether to push the button to open the box.

        """
        raise NotImplementedError

    def step(self, push: bool) -> float:
        r"""Updates box state.

        Args
        ----
        push:
            Whether to push the button to open the box.

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
    r"""Box with food appearance following a Poisson process.

    Args
    ----
    tau:
        Time constant of the Poisson process.
    kwargs:
        Keyword arguments for `BaseFoodBox`.

    """

    def __init__(self,
        *,
        tau: float = 15.,
        **kwargs,
    ):
        super().__init__(tau, **kwargs)
        self.state_space.spaces['food'] = Discrete(2)
        if self.cue_in_state:
            self.state_space.spaces['cue'] = Box(0, 1)

    def __str__(self) -> str:
        return "Poisson food box with tau={:.2g} (kappa={:.3g})".format(self.tau, self.kappa)

    def _reset(self) -> None:
        self.food = False
        self.cue = 0.

    def _step(self, push: bool) -> None:
        if push:
            self._reset()
        else:
            gamma = np.exp(-self.dt/self.tau).item()
            self.cue = 1-(1-self.cue)*gamma
            p_appear = 1-gamma
            if self.rng.random()<p_appear:
                self.food = True


class GammaBox(BaseFoodBox):
    r"""Box updated by Gamma schedule and uses a linear cue.

    After each button push, a reward time interval is drawn from a Gamma
    distribution, then the color cue increases linearly from 0 to 1 until the
    reward is delivered.

    Args
    ----
    tau:
        Expectation of reward intervals from a Gamma distribution.
    shape:
        Shape parameter of the Gamma distribution.
    kwargs:
        Keyword arguments for `BaseFoodBox`.

    """

    drawn: float # drawn interval from Gamma distribution
    timer: float # time since last push, [0, inf)

    def __init__(self,
        tau: float = 14.,
        shape: float = 10.,
        **kwargs,
    ):
        super().__init__(tau, **kwargs)
        self.scale = tau/shape
        self.shape = shape

        if self.cue_in_state:
            self.state_space.spaces.update({
                'drawn': Box(0, np.inf), 'clock': Box(0, np.inf),
            })
        else:
            self.state_space['countdown'] = Box(0, np.inf)

    def __str__(self) -> str:
        return f"Box with Gamma ({self.shape}, {self.scale}) schedule"

    @property
    def food(self) -> bool:
        r"""Returns food state based on current timer and level."""
        if self.cue_in_state:
            return self.clock>=self.drawn
        return self.timer>=self.drawn

    @property
    def cue(self) -> float:
        r"""Returns cue value based on current timer and level."""
        return min(self.timer/self.drawn, 1.)

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

    def get_state(self) -> dict[str, float]:
        r"""Returns box state."""
        state = {'drawn': self.drawn, 'timer': self.timer}
        return state

    def set_state(self, state: dict[str, float]) -> None:
        r"""Sets box state."""
        self.drawn = state['drawn']
        self.timer = state['timer']
        self.render()

    def render(self) -> None:
        self.colors = self.get_colors(self.cue)

    def _reset(self) -> None:
        r"""Draws new reward interval from a Gamma distribution."""
        self.drawn = self.rng.gamma(self.shape, self.scale)
        self.timer = 0

    def _step(self, push: bool) -> None:
        if push:
            self._reset()
        else:
            self.timer += self.dt
