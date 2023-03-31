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
        turn_price: Optional[float] = None,
        move_price: Optional[float] = None,
        look_price: Optional[float] = None,
        velocity: Optional[float] = None,
        k_gamma: Optional[float] = None,
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
        self.push_cost = _rcParams.push_cost if push_cost is None else push_cost
        assert self.push_cost>=0, (
            f"Push cost `push_cost` ({self.push_cost}) must be non-negative."
        )
        self.turn_price = _rcParams.turn_price if turn_price is None else turn_price
        assert self.turn_price>=0, (
            f"Turn cost per degree `turn_price` ({self.turn_price}) must be non-negative."
        )
        self.move_price = _rcParams.move_price if move_price is None else move_price
        assert self.move_price>=0, (
            f"Move cost per distance `move_price` ({self.move_price}) must be non-negative."
        )
        self.look_price = _rcParams.look_price if look_price is None else look_price
        assert self.look_price>=0, (
            f"Look cost per degree `look_price` ({self.look_price}) must be non-negative."
        )
        self.velocity = _rcParams.velocity if velocity is None else velocity
        self.k_gamma = _rcParams.k_gamma if k_gamma is None else k_gamma

        # state: (pos, gaze)
        self.state_space = MultiDiscrete([self.arena.num_tiles]*2)
        # param: (push_cost, turn_price, move_price, look_price)
        self.param_low = [0, 0, 0, 0]
        self.param_high = [np.inf, np.inf, np.inf, np.inf]
        # action: (move, look)
        self._num_moves = 7 # stay still and six hexagonal directions
        self._push = self._num_moves*self.arena.num_tiles # action index for push

        self.rng = np.random.default_rng()

    def reset(self, seed: Optional[int] = None) -> None:
        r"""Resets the monkey state.

        Put the monkey on the outer region randomly, and sets up the gaze
        location to a random tile in inner region.

        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.pos = self.rng.choice(self.arena.outers)
        self.gaze = self.rng.choice(self.arena.inners)

    def get_param(self) -> EnvParam:
        r"""Returns monkey parameters."""
        param = (self.push_cost, self.turn_price, self.move_price, self.look_price)
        return param

    def set_param(self, param) -> None:
        r"""Sets monkey parameters."""
        self.push_cost, self.turn_price, self.move_price, self.look_price = param

    def get_state(self) -> MonkeyState:
        r"""Returns monkey state."""
        state = (self.pos, self.gaze)
        return state

    def set_state(self, state: MonkeyState) -> None:
        r"""Sets the monkey state."""
        self.pos, self.gaze = state

    def _direction(self, end: int, start: int):
        if end==start:
            theta = None
        else:
            dx, dy = np.array(self.arena.anchors[end])-np.array(self.arena.anchors[start])
            theta = np.arctan2(dy, dx)
        return theta

    @staticmethod
    def _delta_deg(theta_0, theta_1):
        delta = np.mod(theta_0-theta_1+np.pi, 2*np.pi)-np.pi
        return np.abs(delta)/np.pi*180

    def step(self, move: int, look: int, eps: float = 1e-4) -> float:
        r"""Monkey acts for one step.

        Turn cost, move cost and look cost will be separately computed.

        Args
        ----
        move:
            A integer in [0, 6]. `move=0` means staying still, while the other
            values means moving along one hexagonal direction. Moving distance
            is draw from a fixed distribution.
        look:
            A integer in [0, `num_tiles`), for the next gaze location.
        eps:
            A small positive number used for out of boundary detection.

        Returns
        -------
        reward:
            Negative value for action cost.

        """
        reward = 0.
        phi = self._direction(self.gaze, self.pos) # face direction
        if move==0:
            dxy = np.array([0., 0.])
        else:
            theta = move/6*(2*np.pi)
            dxy = np.array([np.cos(theta), np.sin(theta)])/self.arena.resol
        d = self.rng.gamma(self.k_gamma, self.velocity/self.k_gamma)
        xy = np.array(self.arena.anchors[self.pos])
        while d>0.5 and self.arena.is_inside(xy*(1-eps)):
            xy += dxy
            d -= 1
        pos = self.arena.nearest_tile(xy) # new position
        theta = self._direction(pos, self.pos) # moving direction
        if not(phi is None or theta is None):
            reward -= self.turn_price*self._delta_deg(theta, phi)
        dxy = np.array(self.arena.anchors[pos])-np.array(self.arena.anchors[self.pos])
        d = (dxy**2).sum()**0.5 # moving distance
        reward -= d*self.move_price
        self.pos = pos
        self.gaze = look
        phi = self._direction(self.gaze, self.pos) # new face direction
        if not(phi is None or theta is None):
            reward -= self.look_price*self._delta_deg(theta, phi)
        assert self.pos%1==0, f"'pos' {self.pos} is not int."
        assert self.gaze%1==0, f"'gaze' {self.gaze} is not int."
        return reward
