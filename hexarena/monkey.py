import numpy as np
from collections.abc import Iterable
from gymnasium.spaces import Discrete, MultiDiscrete
from jarvis.config import Config

from .arena import Arena
from .alias import EnvParam, MonkeyState, Array

class Monkey:
    r"""Class for the monkey in an arena.

    Args
    ----
    arena:
        The arena in which the monkey plays in.
    num_grades:
        Number of distinct colors the agent perceives.
    integrate_area:
        A number in (0, 1] specifying the size of visual field for integrating
        color cues. For example `integrate_area=0.5` means a random patch of
        size 0.5x0.5 on the monitor is integrated to get the mean color.
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
    center_cost:
        Cost for staying at the center. Stay cost of othe tiles decreases
        linearly to 0 towards arena corner.

    """

    def __init__(self,
        arena: Arena|dict|None = None,
        num_grades: int = 8,
        integrate_area: float = 0.8,
        push_cost: float = 1.,
        turn_price: float = 0.001,
        move_price: float = 0.,
        look_price: float = 0.001,
        center_cost: float = 0.1,
    ):
        if arena is None or isinstance(arena, dict):
            arena = Config(arena)
            arena._target_ = 'hexarena.arena.Arena'
            arena = arena.instantiate()
        self.arena: Arena = arena
        self.num_grades = num_grades
        self.integrate_area = integrate_area
        self.push_cost = push_cost
        self.turn_price = turn_price
        self.move_price = move_price
        self.look_price = look_price
        self.center_cost = center_cost

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
            'num_grades': self.num_grades,
            'integrate_area': self.integrate_area,
            'push_cost': self.push_cost,
            'turn_price': self.turn_price,
            'move_price': self.move_price,
            'look_price': self.look_price,
            'center_cost': self.center_cost,
        }

    def reset(self, seed: int|None = None) -> None:
        r"""Resets the monkey state.

        Put the monkey on the outer region randomly, and sets up the gaze
        location to a random tile in inner region.

        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.pos: int = self.rng.choice(self.arena.num_tiles)
        self.gaze: int = self.rng.choice(self.arena.num_tiles)

    def get_param(self) -> EnvParam:
        r"""Returns monkey parameters."""
        param = [self.push_cost, self.turn_price, self.move_price, self.look_price, self.center_cost]
        return param

    def set_param(self, param) -> None:
        r"""Sets monkey parameters."""
        self.push_cost, self.turn_price, self.move_price, self.look_price, self.center_cost = param

    def param_bounds(self) -> tuple[EnvParam, EnvParam]:
        # param: (push_cost, turn_price, move_price, look_price, center_cost)
        param_low = [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf]
        param_high = [np.inf, np.inf, np.inf, np.inf, np.inf]
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
        assert self.arena.num_boxes==3 and num_macros in [10, 22], (
            f"`num_macros={num_macros}` is not supported"
        )
        if num_macros==22:
            assert self.arena.num_tiles==19
        macros = []
        for action in actions:
            push, move, _ = self.convert_action(action) # `look` is ignored
            if num_macros==10:
                if push:
                    macro = self.arena.boxes.index(move) # [0, 3) for push actions
                else:
                    assert self.arena.num_tiles==19, "Only 19-tile environment is supported"
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
                    macro = 3+move
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
        # stay cost
        d = (np.array(self.arena.anchors[move])**2).sum()**0.5
        reward -= self.center_cost*(1-d)
        # push cost
        if push:
            reward -= self.push_cost
        return reward

    def look(self, colors: Array) -> int:
        r"""Returns the observation when looking at a color cue array.

        Args
        ----
        colors: (height, width)
            The color cue array with values in a periodic range [0, 1).

        Returns
        -------
        observation:
            An integer in [0, num_grades), indicating the circular mean color on
            a random patch.

        """
        H, W = colors.shape
        h = int(np.ceil(self.integrate_area*H))
        w = int(np.ceil(self.integrate_area*W))
        i = self.rng.choice(H-h)
        j = self.rng.choice(W-h)
        vals = colors[i:i+h, j:j+w]
        xs, ys = np.cos(2*np.pi*vals), np.sin(2*np.pi*vals)
        val = np.mod(np.arctan2(ys.mean(), xs.mean())/(2*np.pi), 1)
        observation = int(np.floor(val*self.num_grades))
        return observation
