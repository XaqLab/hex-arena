import numpy as np
import torch
from gym import Env
from gym.spaces import Discrete, MultiDiscrete
from jarvis.config import Config
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation

from typing import Optional
from irc.buffer import Episode
from irc.distribution import BaseDistribution

from . import rcParams
from .arena import Arena
from .box import FoodBox
from .monkey import Monkey
from .alias import EnvParam, Observation, State

class ForagingEnv(Env):
    r"""Foraging environment with food boxes in a hexagonal arena."""

    def __init__(self,
        arena: Optional[dict] = None,
        boxes: Optional[list[Optional[dict]]] = None,
        monkey: Optional[dict] = None,
        time_cost: Optional[float] = None,
        dt: Optional[float] = None,
    ):
        _rcParams = Config(rcParams.get('env.ForagingEnv._init_'))
        self.time_cost = _rcParams.time_cost if time_cost is None else time_cost
        self.dt = _rcParams.dt if dt is None else dt
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
            box.dt = self.dt
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
        reward, terminated, truncated = -self.time_cost, False, False
        if action<self._push:
            move = action%self._num_moves
            look = action//self._num_moves
            reward += self.monkey.step(move, look)
        for box in self.boxes:
            if action==self._push and self.monkey.pos==box.pos:
                self.monkey.gaze = self.monkey.pos
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
        # episode: Episode,
        pos, gaze, colors, push, success,
        num_steps: Optional[int] = None,
        aname: str = 'foraging-trial.gif',
        figsize: tuple[float, float] = None,
        use_sec: bool = False,
    ):
        if num_steps is None:
            num_steps = len(push)
        else:
            num_steps = min(num_steps, len(push))
        if figsize is None:
            figsize = (4.5, 4)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0.1, 0.05, 0.8, 0.9])
        self.arena.plot_map(ax)

        h_boxes = []
        for box in self.boxes:
            c_size = int(box.num_patches**0.5)
            _colors = np.zeros((c_size, c_size), dtype=int)
            _x, _y = np.array(self.arena.anchors[box.pos])*1.5
            _s = 0.2
            h_box = ax.imshow(
                _colors, extent=[_x-_s, _x+_s, _y-_s, _y+_s],
                vmin=-1, vmax=box.num_grades, cmap='RdYlBu_r',
                zorder=2,
            )
            h_boxes.append(h_box)
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.2])
        h_pos = ax.add_patch(Polygon(
            np.full((1, 2), fill_value=np.nan),
            edgecolor='none', facecolor='yellow', alpha=0.2,
        ))
        h_gaze = ax.scatter(np.nan, np.nan, s=100, marker='o', edgecolor='none', facecolor='blue')
        ti = ax.set_title('')

        _xy = np.stack([
            np.array([np.cos(theta), np.sin(theta)])/(2*self.arena.resol)
            for theta in [i/3*np.pi+np.pi/6 for i in range(6)]
        ])
        def update(t):
            for i, h_box in enumerate(h_boxes):
                h_box.set_data(colors[t, i])
            _pos = pos[t]
            h_pos.set_xy(_xy+self.arena.anchors[_pos])
            if t>0 and _pos in self.arena.boxes and push[t-1]:
                if success[t-1]>0:
                    h_pos.set_facecolor('red')
                else:
                    h_pos.set_facecolor('blue')
            else:
                h_pos.set_facecolor('yellow')
            h_gaze.set_offsets(self.arena.anchors[gaze[t]])
            ti.set_text(r'$t$='+'{:d}'.format(t*self.dt if use_sec else t))
            return *h_boxes, h_pos, h_gaze, ti

        ani = FuncAnimation(fig, update, frames=range(num_steps), blit=True)
        ani.save(aname)
        return fig, ani

    def play_box_beliefs(self,
        episode: Episode,
        p_s: BaseDistribution,
        aname: str = 'cue-belief-trace.gif',
        figsize: tuple[float, float] = None,
        use_sec: bool = False,
    ):
        assert len(self.boxes)==3, "Only implemented for three boxes."
        if figsize is None:
            figsize = (6, 4)
        fig = plt.figure(figsize=figsize)
        colors = ['violet', 'lime', 'tomato']

        nvec = self.state_space.nvec[-6:]
        box_states = np.stack(np.unravel_index(np.arange(np.prod(nvec)), nvec)).T

        beliefs = np.zeros((episode.num_steps+1, 3)) # probability of food in each box
        for t in range(episode.num_steps+1):
            belief = episode.beliefs[t]
            states = np.tile(episode.states[t], (np.prod(nvec), 1))
            states[:, 2:] = box_states
            p_s.set_param_vec(belief)
            with torch.no_grad():
                _probs = p_s.loglikelihoods(
                    torch.tensor(states, dtype=torch.long),
                ).exp().numpy()
            for i in range(3):
                beliefs[t, i] = _probs[box_states[:, 2*i]==1].sum()

        ax = fig.add_axes([0.15, 0.6, 0.8, 0.3])
        c_lines = []
        for i in range(3):
            h, = ax.plot(np.nan, np.nan, color=colors[i])
            c_lines.append(h)
        ax.set_xlim(np.array([-0.05, 1.05])*episode.num_steps)
        ax.set_xticklabels([])
        ax.set_ylim([0, 1.05])
        ax.set_ylabel('Cue')

        ax = fig.add_axes([0.15, 0.2, 0.8, 0.3])
        b_lines = []
        for i in range(3):
            h, = ax.plot(np.nan, np.nan, color=colors[i])
            b_lines.append(h)
        ax.legend(['Box A', 'Box B', 'Box C'], fontsize='x-small', loc='lower right')
        ax.set_xlim(np.array([-0.05, 1.05])*episode.num_steps)
        ax.set_xlabel('Time (sec)' if use_sec else '$t$')
        ax.set_ylim([0, 1.05])
        ax.set_ylabel('Belief')

        def update(t):
            for i, h in enumerate(c_lines):
                h.set_data(np.array([
                    np.arange(t+1),
                    np.array([episode.infos[j]['cues'][i] for j in range(t+1)]),
                ]))
            for i, h in enumerate(b_lines):
                h.set_data(np.array([
                    np.arange(t+1),
                    beliefs[:t+1, i],
                ]))
            return *c_lines, *b_lines

        ani = FuncAnimation(fig, update, frames=range(episode.num_steps+1), blit=True)
        ani.save(aname)
        return fig, ani

    def plot_occupancy(self,
        episodes: list[Episode],
        figsize: tuple[float, float] = None,
    ):
        pos, gaze = [], []
        for episode in episodes:
            for info in episode.infos:
                pos.append(info['pos'])
                gaze.append(info['gaze'])

        _xy = np.stack([
            np.array([np.cos(theta), np.sin(theta)])/(2*self.arena.resol)
            for theta in [i/3*np.pi+np.pi/6 for i in range(6)]
        ])
        cmap = plt.get_cmap('YlOrBr')
        for i in range(2):
            if figsize is None:
                figsize = (4.5, 4)
            fig = plt.figure(figsize=figsize)
            ax = fig.add_axes([0.1, 0.05, 0.8, 0.9])
            self.arena.plot_map(ax)

            if i==0:
                val = pos
                title = 'Monkey position'
                fig_p = fig
            if i==1:
                val = gaze
                title = 'Gaze position'
                fig_g = fig
            counts = np.zeros(self.arena.num_tiles)
            for j in range(self.arena.num_tiles):
                counts[j] = (np.array(val)==j).sum()
            norm = mpl.colors.Normalize(vmin=0, vmax=counts.max()/counts.sum())
            for j in range(self.arena.num_tiles):
                xy = _xy+self.arena.anchors[j]
                ax.add_patch(Polygon(
                    xy, edgecolor='none', facecolor=cmap(counts[j]/counts.max()), zorder=-1,
                ))
            plt.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=ax, fraction=0.1, shrink=0.8, pad=0.1,
                orientation='horizontal', label='Probability',
            )

            ax.set_title(title)
        return fig_p, fig_g
