import numpy as np
from gym import Env
from gym.spaces import Discrete, MultiDiscrete
from jarvis.config import Config
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation

from typing import Optional
from irc.buffer import Episode

from . import rcParams
from .arena import Arena
from .box import FoodBox
from .monkey import Monkey
from .alias import EnvParam, Observation, State, Axes

class ForagingEnv(Env):
    r"""Foraging environment with food boxes in a hexagonal arena."""

    def __init__(self,
        arena: Optional[dict] = None,
        boxes: Optional[list[Optional[dict]]] = None,
        monkey: Optional[dict] = None,
        dt: Optional[float] = None,
    ):
        _rcParams = Config(rcParams.get('env.ForagingEnv._init_'))
        dt = _rcParams.dt if dt is None else dt
        arena = Config(arena)
        arena._target_ = 'hexarena.arena.Arena'
        self.arena: Arena = arena.instantiate()
        num_boxes = 3 # fixed three boxes
        if boxes is None:
            boxes = [None]*num_boxes
        else:
            assert len(boxes)==num_boxes
        self.boxes: list[FoodBox] = []
        for i in range(num_boxes):
            box = Config(boxes[i])
            box._target_ = 'hexarena.box.FoodBox'
            box.dt = dt
            box: FoodBox = box.instantiate()
            box.pos = self.arena.boxes[i]
            self.boxes.append(box)
        monkey = Config(monkey)
        monkey._target_ = 'hexarena.monkey.Monkey'
        self.monkey: Monkey = monkey.instantiate(arena=self.arena)

        # environment parameter is the concatenation of monkey's and boxes'
        self._nums_params, self.param_low, self.param_high = [], [], []
        for x in self._components():
            self._nums_params.append(len(x.get_param()))
            self.param_low += [*x.param_low]
            self.param_high += [*x.param_high]

        # state: (*monkey_state, *boxes_state)
        self._state_dims, nvec = [], []
        for x in self._components():
            self._state_dims.append(len(x.state_space.nvec))
            nvec += [*x.state_space.nvec]
        self.state_space = MultiDiscrete(nvec)
        # observation: (*monkey_state, *boxes_colors)
        self._observation_dims = [len(self.monkey.state_space.nvec)]
        nvec = [*self.monkey.state_space.nvec]
        for box in self.boxes:
            self._observation_dims.append(box.num_patches)
            nvec += [box.num_grades+1]*box.num_patches # additional grade for invisible
        self.observation_space = MultiDiscrete(nvec)
        # action: move*look+push
        self._num_moves = 7
        self._push = self._num_moves*self.arena.num_tiles
        self.action_space = Discrete(self._push+1)

    def _components(self):
        return [self.monkey]+self.boxes

    def get_param(self) -> EnvParam:
        r"""Returns environment parameters."""
        param = []
        for x in self._components():
            param += [*x.get_param()]
        return param

    def set_param(self, param: EnvParam) -> None:
        r"""Sets environment parameter."""
        idx = 0
        for x, num_param in zip(self._components(), self._nums_params):
            x.set_param(param[idx:(idx+num_param)])
            idx += num_param

    def get_state(self) -> State:
        state = []
        for x in self._components():
            state += [*x.get_state()]
        return state

    def set_state(self, state: State) -> None:
        idx = 0
        for x, state_dim in zip(self._components(), self._state_dims):
            x.set_state(state[idx:(idx+state_dim)])
            idx += state_dim

    def reset(self, seed: Optional[int] = None) -> tuple[Observation, dict]:
        for x in self._components():
            x.reset(seed)
        observation = self._get_observation()
        info = self._get_info()
        return observation, info

    def step(self, action: int) -> tuple[Observation, float, bool, bool, dict]:
        reward, terminated, truncated = 0., False, False
        if action<self._push:
            move = action%self._num_moves
            look = action//self._num_moves
            reward += self.monkey.step(move, look)
        for box in self.boxes:
            if action==self._push and self.monkey.pos==box.pos and self.monkey.gaze==box.pos:
                reward += box.step(1)
                reward -= self.monkey.push_cost
            else:
                reward += box.step(0)
        observation = self._get_observation()
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> Observation:
        observation = [*self.monkey.get_state()]
        for box in self.boxes:
            if self.monkey.gaze==box.pos:
                colors = box.colors.reshape(-1)
            else:
                colors = np.full((box.num_patches,), fill_value=box.num_grades, dtype=int)
            observation += [*colors]
        return observation

    def _get_info(self) -> dict:
        info = {
            'pos': self.monkey.pos, 'gaze': self.monkey.gaze,
            'foods': [box.food for box in self.boxes],
            'cues': [box.cue for box in self.boxes],
            'colors': [box.colors for box in self.boxes],
        }
        return info

    def play_episode(self,
        episode: Episode,
        aname: str = 'foraging-trial.gif',
        figsize: tuple[float, float] = None,
    ):
        if figsize is None:
            figsize = (4, 4)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        self.arena.plot_map(ax)

        pa = ax.add_patch(Polygon(np.full((1, 2), fill_value=np.nan), edgecolor='none', facecolor='yellow', alpha=0.2))
        sc = ax.scatter(np.nan, np.nan, s=100, marker='o', edgecolor='none', facecolor='blue')
        ti = ax.set_title('')

        _xy = np.stack([
            np.array([np.cos(theta), np.sin(theta)])/(2*self.arena.resol)
            for theta in [i/3*np.pi+np.pi/6 for i in range(6)]
        ])
        def update(t):
            pos = episode.states[t, 0]
            pa.set_xy(_xy+self.arena.anchors[pos])
            gaze = episode.states[t, 1]
            sc.set_offsets(self.arena.anchors[gaze])
            ti.set_text(r'$t$='+'{:d}'.format(t))
            return pa, sc, ti

        ani = FuncAnimation(fig, update, frames=range(episode.num_steps+1), blit=True)
        ani.save(aname, writer='pillow')
        return fig
