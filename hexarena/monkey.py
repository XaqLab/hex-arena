import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete
from jarvis.config import Config

from .arena import Arena
from .alias import EnvParam, MonkeyState

class Monkey:
    r"""Class for the monkey in an arena."""

    def __init__(self,
        arena: Arena|dict|None = None,
        push_cost: float = 1.,
        turn_price: float = 0.001,
        move_price: float = 0.,
        look_price: float = 0.001,
    ):
        r"""
        Args
        ----
        arena:
            The arena in which the monkey plays in.
        push_cost:
            Cost of pushing the button to open the food box.
        turn_price:
            Price of turning, in units of 1/deg. It will be multiplied by the
            turning angle before moving to get turning cost.
        move_price:
            Price of moving, in units of 1/(1^2). It will be multiplied by the
            square of distance to get moving cost.
        look_price:
            Price of looking, in units of 1/deg. It will be multiplied by the
            turning angle after moving to get looking cost.

        """
        if arena is None or isinstance(arena, dict):
            arena = Config(arena)
            arena._target_ = 'hexarena.arena.Arena'
            arena = arena.instantiate()
        self.arena: Arena = arena
        self.push_cost = push_cost
        self.turn_price = turn_price
        self.move_price = move_price
        self.look_price = look_price

        # state: (pos, gaze)
        self.state_space = MultiDiscrete([self.arena.num_tiles]*2)
        # action: (push, move, look)
        self.action_space = Discrete(self.arena.num_boxes*self.arena.num_tiles+self.arena.num_tiles**2)

        self.rng = np.random.default_rng()

    def __repr__(self) -> str:
        return "A monkey with push and moving cost"

    @property
    def spec(self) -> dict:
        return {
            '_target_': 'hexarena.monkey.Monkey',
            'push_cost': self.push_cost,
            'turn_price': self.turn_price,
            'move_price': self.move_price,
            'look_price': self.look_price,
        }

    def reset(self, seed: int|None = None) -> None:
        r"""Resets the monkey state.

        Put the monkey on the outer region randomly, and sets up the gaze
        location to a random tile in inner region.

        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.pos: int = self.rng.choice(self.arena.outers)
        self.gaze: int = self.rng.choice(self.arena.inners)

    def get_param(self) -> EnvParam:
        r"""Returns monkey parameters."""
        param = [self.push_cost, self.turn_price, self.move_price, self.look_price]
        return param

    def set_param(self, param) -> None:
        r"""Sets monkey parameters."""
        self.push_cost, self.turn_price, self.move_price, self.look_price = param

    def param_bounds(self) -> tuple[EnvParam, EnvParam]:
        # param: (push_cost, turn_price, move_price, look_price)
        param_low = [-np.inf, -np.inf, -np.inf, -np.inf]
        param_high = [np.inf, np.inf, np.inf, np.inf]
        return param_low, param_high

    def get_state(self) -> MonkeyState:
        r"""Returns monkey state."""
        state = (self.pos, self.gaze)
        return state

    def set_state(self, state: MonkeyState) -> None:
        r"""Sets the monkey state."""
        self.pos, self.gaze = state

    def _direction(self, end: int, start: int) -> float|None:
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

    def convert_action(self, action: int) -> tuple[bool, int, int]:
        r"""Converts action integer to interpretable variables.

        Args
        ----
        action:
            An integer in [0, num_tiles^2+num_boxes*num_tiles). `action` in
            [0, num_tiles^2) means moving and looking only. `action` in
            [num_tiles^2, num_tiles^2+num_boxes*num_tiles) means moving to a box
            and push, combined with looking.

        Returns
        -------
        push:
            Whether to push the button of food box.
        move:
            An integer in [0, num_tiles) for the desired tile of moving to.
        look:
            An integer in [0, num_tiles) for the desired tile of looking at.

        """
        if action<self.arena.num_tiles**2:
            push = False
            move = action//self.arena.num_tiles
            look = action%self.arena.num_tiles
        else:
            push = True
            move = self.arena.boxes[(action-self.arena.num_tiles**2)//self.arena.num_tiles]
            look = (action-self.arena.num_tiles**2)%self.arena.num_tiles
        return push, move, look

    def index_action(self, push: bool, move: int, look: int) -> int:
        r"""Returns action index given interpretable variables.

        Args
        ----
        push, move, look:
            See `convert_action` for more details.

        Returns
        -------
        action:
            See `convert_action` for more details.

        """
        if push:
            b_idx = self.arena.boxes.index(move)
            action = self.arena.num_tiles**2+b_idx*self.arena.num_tiles+look
        else:
            action = move*self.arena.num_tiles+look
        return action

    def step(self, push: bool, move: int, look: int) -> float:
        r"""Monkey acts for one step.

        Args
        ----
        push, move, look:
            See `convert_action` for more details.

        Returns
        -------
        reward:
            Endogenous reward to the monkey, i.e. the summation of turn cost,
            move cost, look cost and push cost.

        """
        reward = 0.
        # turn cost
        phi = self._direction(self.gaze, self.pos) # face direction
        theta = self._direction(move, self.pos) # moving direction
        if not(phi is None or theta is None):
            reward -= self.turn_price*self._delta_deg(theta, phi)
        else:
            theta = theta or phi # for look cost later
        # move cost
        dxy = np.array(self.arena.anchors[move])-np.array(self.arena.anchors[self.pos])
        d2 = (dxy**2).sum()
        reward -= self.move_price*d2
        self.pos = move
        self.gaze = look
        # look cost
        phi = self._direction(self.gaze, self.pos) # new face direction
        if not(phi is None or theta is None):
            reward -= self.look_price*self._delta_deg(theta, phi)
        # push cost
        if push:
            reward -= self.push_cost
        return reward
