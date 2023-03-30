import numpy as np
from gym.spaces import MultiDiscrete
from jarvis.config import Config

from typing import Optional, Union

from . import rcParams
from .arena import Arena
from .alias import EnvParam, MonkeyState

class Monkey:
    r"""Class for the monkey in an arena."""

    def __init__(self,
        arena: Union[Arena, dict, None] = None,
        push_cost: Optional[float] = None,
        move_price: Optional[float] = None,
        look_price: Optional[float] = None,
    ):
        r"""
        Args
        ----
        arena:
            The arena in which the monkey plays in.

        """
        _rcParams = Config(rcParams.get('monkey.Monkey._init_'))
        if arena is None or isinstance(arena, dict):
            arena = Config(arena)
            arena._target_ = 'hexarena.arena.Arena'
            arena: Arena = arena.instantiate()
        self.arena = arena
        self.push_cost = push_cost or _rcParams.push_cost
        assert self.push_cost>=0, (
            f"Push cost `push_cost` ({self.push_cost}) must be non-negative."
        )
        self.move_price = move_price or _rcParams.move_price
        assert self.move_price>=0, (
            f"Move cost per distance `move_price` ({self.move_price}) must be non-negative."
        )
        self.look_price = look_price or _rcParams.look_price
        assert self.look_price>=0, (
            f"Look cost per degree `look_price` ({self.look_price}) must be non-negative."
        )

        # state: (pos, gaze)
        self.state_space = MultiDiscrete([self.arena.num_tiles]*2)
        # param: (push_cost, move_price, look_price)
        self.param_low = [0, 0, 0]
        self.param_high = [np.inf, np.inf, np.inf]

        self.rng = np.random.default_rng()

    def reset(self, seed: Optional[int] = None) -> None:
        r"""Resets the monkey state.

        Put the monkey on the outer region randomly, and sets up the gaze
        position to a random tile in inner region.

        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.pos = self.rng.choice(self.arena.outers)
        self.gaze = self.rng.choice(self.arena.inners)

    def get_param(self) -> EnvParam:
        r"""Returns monkey parameters."""
        param = (self.push_cost, self.move_price, self.look_price)
        return param

    def set_param(self, param) -> None:
        r"""Sets monkey parameters."""
        self.push_cost, self.move_price, self.look_price = param

    def get_state(self) -> MonkeyState:
        r"""Returns monkey state."""
        state = (self.pos, self.gaze)
        return state

    def set_state(self, state: MonkeyState) -> None:
        r"""Sets the monkey state."""
        self.pos, self.gaze = state
