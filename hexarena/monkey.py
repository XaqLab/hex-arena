import numpy as np
from collections.abc import Iterable
from gymnasium.spaces import Discrete, MultiDiscrete, Dict
from jarvis.config import Config

from .arena import Arena
from .alias import Array, EnvParam


class BaseMonkey:
    r"""Base class for the monkey.

    The monkey can push the box for food and look at color cue.

    Args
    ----
    time_cost:
        Cost of each time step. Usually non-negative to encourage engagement.
    push_cost:
        Cost of pushing the button to open the food box.
    vis_field:
        A number in (0, 1] specifying the size of visual field for integrating
        color cues. For example `integrate_area=0.5` means a random patch of
        size 0.5x0.5 on the monitor is integrated to get the mean color.

    """

    def __init__(self,
        *,
        time_cost: float = 0.,
        push_cost: float = 0.,
        vis_field: float = 0.8,
    ):
        self.time_cost = time_cost
        self.push_cost = push_cost
        self.vis_field = vis_field

        self.state_space = Dict({})

    def __str__(self) -> str:
        return "Monkey with push cost {:g}".format(self.push_cost)

    def __repr__(self) -> str:
        return str(self.spec)

    @property
    def spec(self) -> dict:
        return {
            '_target_': 'hexarena.monkey.BaseMonkey',
            'time_cost': self.time_cost,
            'push_cost': self.push_cost,
            'vis_field': self.vis_field,
        }

    def reset(self) -> None:
        ...

    def step(self, push: bool) -> float:
        r"""Monkey acts for one step.

        Args
        ----
        push:
            Whether the monkey pushes the food box.

        Returns
        -------
        reward:
            Endogenous reward to the monkey, i.e. the summation of time cost and
            push cost.

        """
        reward = -self.time_cost
        if push:
            reward -= self.push_cost
        return reward

    def look(self, colors: Array) -> tuple[float, float]:
        r"""Returns the observation when looking at a color cue array.

        Args
        ----
        colors: (height, width)
            The color cue array with values in a periodic range [0, 1).

        Returns
        -------
        x, y:
            2D representation of circular mean of a random crop of `colors`, in
            range [-1, 1].

        """
        H, W = colors.shape
        h = int(np.ceil(self.vis_field*H))
        w = int(np.ceil(self.vis_field*W))
        i = self.rng.choice(H-h)
        j = self.rng.choice(W-w)
        vals = colors[i:i+h, j:j+w]
        x = np.cos(2*np.pi*vals).mean().item()
        y = np.sin(2*np.pi*vals).mean().item()
        return x, y

    def get_param(self) -> EnvParam:
        r"""Returns monkey parameters."""
        param = [self.time_cost, self.push_cost]
        return param

    def set_param(self, param) -> None:
        r"""Sets monkey parameters."""
        self.time_cost, self.push_cost = param

    def param_bounds(self) -> tuple[EnvParam, EnvParam]:
        # param: (time_cost, push_cost)
        param_low = [-np.inf, -np.inf]
        param_high = [np.inf, np.inf]
        return param_low, param_high

    def get_state(self) -> dict:
        return {}

    def set_state(self, state: dict) -> None:
        assert state=={}


class ArenaMonkey(BaseMonkey):
    r"""Class for the monkey in an arena.

    Args
    ----
    arena:
        The arena in which the monkey plays in.
    turn_price:
        Price of turning, in units of 1/deg. It will be multiplied by the
        turning angle before/after moving to get turning cost.
    move_price:
        Price of moving, in units of 1/(1^2). It will be multiplied by the
        square of distance to get moving cost.
    center_cost:
        Cost for staying near the center. Staying cost of the tiles decreases
        linearly to 0 towards arena corner.

    """

    def __init__(self,
        arena: Arena|dict|None = None,
        *,
        turn_price: float = 0.001,
        move_price: float = 0.,
        center_cost: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if arena is None or isinstance(arena, dict):
            arena = Config(arena)
            arena._target_ = 'hexarena.arena.Arena'
            arena = arena.instantiate()
        self.arena: Arena = arena
        self.turn_price = turn_price
        self.move_price = move_price
        self.center_cost = center_cost

        # state: (pos, gaze)
        self.state_space = Dict({
            'pos': Discrete(self.arena.n_tiles),
            'gaze': Discrete(self.arena.n_tiles),
        })
        # action: (push, move, look)
        self.action_space = Discrete(self.arena.n_boxes*self.arena.n_tiles+self.arena.n_tiles**2)

        rs = (np.array(self.arena.anchors)**2).sum(axis=1)**0.5
        self.stay_costs = self.center_cost*(1-rs)
        self.rng = np.random.default_rng()

    def __str__(self) -> str:
        arena_str = str(self.arena)
        arena_str = arena_str[0].lower()+arena_str[1:]
        return f"Monkey in {arena_str}"

    @property
    def spec(self) -> dict:
        spec = super().spec
        spec.update({
            '_target_': 'hexarena.monkey.ArenaMonkey',
            'turn_price': self.turn_price,
            'move_price': self.move_price,
            'center_cost': self.center_cost,
        })
        return spec

    def reset(self, seed: int|None = None) -> None:
        r"""Resets the monkey state.

        Put the monkey on the outer region randomly, and sets up the gaze
        location to a random tile in inner region.

        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.pos: int = self.rng.choice(self.arena.n_tiles)
        self.gaze: int = self.rng.choice(self.arena.n_tiles)

    def get_param(self) -> EnvParam:
        r"""Returns monkey parameters."""
        param = super().get_param()+[
            self.turn_price, self.move_price, self.center_cost,
        ]
        return param

    def set_param(self, param) -> None:
        r"""Sets monkey parameters."""
        self.turn_price, self.move_price, self.center_cost = param[-3:]
        super().set_param(param[:-3])

    def param_bounds(self) -> tuple[EnvParam, EnvParam]:
        # param: (*, turn_price, move_price, center_cost)
        param_low, param_high = super().param_bounds()
        param_low += [-np.inf, -np.inf, -np.inf]
        param_high += [np.inf, np.inf, np.inf]
        return param_low, param_high

    def get_state(self) -> dict[str, int]:
        r"""Returns monkey state."""
        state = {'pos': self.pos, 'gaze': self.gaze}
        return state

    def set_state(self, state: dict[str, int]) -> None:
        r"""Sets the monkey state."""
        self.pos, self.gaze = state['pos'], state['gaze']

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
        action = int(action)
        if action<self.arena.n_tiles**2:
            push = False
            move = action//self.arena.n_tiles
            look = action%self.arena.n_tiles
        else:
            push = True
            move = self.arena.boxes[(action-self.arena.n_tiles**2)//self.arena.n_tiles]
            look = (action-self.arena.n_tiles**2)%self.arena.n_tiles
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
            action = self.arena.n_tiles**2+b_idx*self.arena.n_tiles+look
        else:
            action = move*self.arena.n_tiles+look
        return action

    def merge_actions(self,
        actions: Iterable[int],
        num_macros: int = 10,
    ) -> list[int]:
        r"""Merges primitive actions to macro actions.

        Designed for three-box arena, so that primitive actions are grouped into
        smaller set of macro actions.

        Args
        ----
        actions: (num_steps,)
            Sequence of primitive actions.
        num_actions:
            Macro action space size. 'num_macros=10' involves 3 push actions and
            7 move actions. See comments for more details. 'num_macros=22'
            involves 3 push actions and 19 move actions, for arena size of 2.

        Returns
        -------
        macros: (num_steps,)
            Sequence of macro actions.

        """
        if not (self.arena.n_tiles==19 and self.arena.n_boxes==3):
            raise NotImplementedError("Only works on the arena with 'resol=2'")
        if num_macros not in [10, 22]:
            raise NotImplementedError(f"`num_macros={num_macros}` is not supported")
        macros = []
        for action in actions:
            push, move, _ = self.convert_action(action) # `look` is ignored
            if num_macros==10:
                if push:
                    macro = self.arena.boxes.index(move) # [0, 3) for push actions
                else:
                    if move in [1, 2, 7, 8, 9]: # near box 0
                        macro = 3
                    if move in [3, 4, 11, 12, 13]: # near box 1
                        macro = 4
                    if move in [5, 6, 15, 16, 17]: # near box 2
                        macro = 5
                    if move==0: # center
                        macro = 6
                    if move==10: # between box 0-1
                        macro = 7
                    if move==14: # between box 1-2
                        macro = 8
                    if move==18: # between box 2-0
                        macro = 9
            if num_macros==22:
                if push:
                    macro = self.arena.boxes.index(move)
                else:
                    macro = 3+move # move to each tile is considered separately
            macros.append(macro)
        return macros

    def step(self, push: bool, move: int, look: int) -> float:
        r"""Monkey acts for one step.

        Args
        ----
        push, move, look:
            See `convert_action` for more details.

        Returns
        -------
        reward:
            Endogenous reward to the monkey, i.e. the summation of time cost,
            push cost, turning cost, moving cost and staying cost.

        """
        reward = super().step(push)
        # turning cost before moving
        phi = self._direction(self.gaze, self.pos) # face direction
        theta = self._direction(move, self.pos) # moving direction
        if not(phi is None or theta is None):
            reward -= self.turn_price*self._delta_deg(theta, phi)
        else:
            theta = theta or phi # for look cost later
        # moving cost
        dxy = np.array(self.arena.anchors[move])-np.array(self.arena.anchors[self.pos])
        d2 = (dxy**2).sum()
        reward -= self.move_price*d2
        self.pos = move
        self.gaze = look
        # turning cost after moving
        phi = self._direction(self.gaze, self.pos) # new face direction
        if not(phi is None or theta is None):
            reward -= self.turn_price*self._delta_deg(theta, phi)
        # staying cost
        reward -= self.stay_costs[move]
        return reward
