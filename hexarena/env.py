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
from collections.abc import Iterable
from irc.distribution import BaseDistribution

from . import rcParams
from .arena import Arena
from .box import BaseFoodBox
from .monkey import Monkey
from .alias import EnvParam, Observation, State, Figure, Array

class ForagingEnv(Env):
    r"""Foraging environment with food boxes in a hexagonal arena."""

    def __init__(self,
        *,
        arena: Optional[dict] = None,
        monkey: Optional[dict] = None,
        boxes: Optional[list[Optional[dict]]] = None,
        time_cost: Optional[float] = None,
        dt: Optional[float] = None,
    ):
        _rcParams = rcParams.get('env.ForagingEnv._init_', {})
        self.time_cost = _rcParams.time_cost if time_cost is None else time_cost
        self.dt = _rcParams.dt if dt is None else dt

        arena = Config(arena)
        arena._target_ = 'hexarena.arena.Arena'
        self.arena: Arena = arena.instantiate()

        monkey = Config(monkey)
        monkey._target_ = 'hexarena.monkey.Monkey'
        self.monkey: Monkey = monkey.instantiate(arena=self.arena)

        num_boxes = self.arena.num_boxes
        if boxes is None:
            boxes = [None]*num_boxes
        else:
            assert len(boxes)==num_boxes
        self.boxes: list[BaseFoodBox] = []
        for i in range(num_boxes):
            box = Config(boxes[i])
            if '_target_' not in box:
                box._target_ = _rcParams.box
            box.dt = self.dt
            box = box.instantiate()
            box.pos = self.arena.boxes[i]
            self.boxes.append(box)

        # environment parameter: (*monkey_param, *boxes_param)
        self._param_dims, self.param_low, self.param_high = [], [], []
        for x in self._components():
            self._param_dims.append(len(x.get_param()))
            self.param_low += [*x.param_low]
            self.param_high += [*x.param_high]

        # state: (*monkey_state, *boxes_state)
        self._state_dims, nvec = [], []
        for x in self._components():
            self._state_dims.append(len(x.state_space.nvec))
            nvec += [*x.state_space.nvec]
        self.state_space = MultiDiscrete(nvec)
        # observation: (*monkey_state, *boxes_colors)
        nvec = [*self.monkey.state_space.nvec]
        for box in self.boxes:
            nvec += [box.num_grades+1]*box.num_patches # additional grade for invisible
        self.observation_space = MultiDiscrete(nvec)
        # action: (push, move, look)
        self.action_space = self.monkey.action_space

    def _components(self):
        return [self.monkey]+self.boxes

    def get_param(self) -> EnvParam:
        r"""Returns environment parameters."""
        param = tuple()
        for x in self._components():
            param += tuple([*x.get_param()])
        return param

    def set_param(self, param: EnvParam) -> None:
        r"""Sets environment parameter."""
        idx = 0
        for x, param_dim in zip(self._components(), self._param_dims):
            x.set_param([param[i] for i in range(idx, idx+param_dim)])
            idx += param_dim

    def get_state(self) -> State:
        state = tuple()
        for x in self._components():
            state += tuple([*x.get_state()])
        return state

    def set_state(self, state: State) -> None:
        idx = 0
        for x, state_dim in zip(self._components(), self._state_dims):
            x.set_state([state[i] for i in range(idx, idx+state_dim)])
            idx += state_dim

    def reset(self, seed: Optional[int] = None) -> tuple[Observation, dict]:
        for x in self._components():
            x.reset(seed)
        observation = self._get_observation()
        info = self._get_info()
        return observation, info

    def step(self, action: int) -> tuple[Observation, float, bool, bool, dict]:
        reward, terminated, truncated = -self.time_cost, False, False
        push, move, look = self.monkey.convert_action(action)
        reward += self.monkey.step(push, move, look)
        for box in self.boxes:
            reward += box.step(push and move==box.pos)
        observation = self._get_observation()
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> Observation:
        observation = tuple([*self.monkey.get_state()])
        for box in self.boxes:
            if self.monkey.gaze==box.pos:
                colors = box.colors.reshape(-1)
            else:
                colors = np.full((box.num_patches,), fill_value=box.num_grades, dtype=int)
            observation += tuple([*colors])
        return observation

    def _get_info(self) -> dict:
        info = {
            'pos': self.monkey.pos, 'gaze': self.monkey.gaze,
            'foods': [box.food for box in self.boxes],
            'cues': [box.cue for box in self.boxes],
            'colors': [box.colors for box in self.boxes],
        }
        return info

    def convert_experiment_data(self, block_data: dict, arena_radius=1860.5) -> dict:
        r"""Converts raw experiment data to discrete values.

        Spatial and temporal discretization is done to get integer states,
        observations and actions.

        Args
        ----
        block_data:
            One block of raw data loaded by `utils.load_monkey_data`.

        Returns
        -------
        env_data:
            A dictionary containing discretized data, with keys:
            - `num_steps`: int. Number of steps in the block.
            - `pos`: (num_steps+1,) int array. Monkey position.
            - `gaze`: (num_steps+1,) int array. Gaze position.
            - `push`: (num_steps,) bool array. Whether the button is pushed.
            - `success`: (num_steps,) bool array. Whether the food is obtained.
            - `box`: (num_steps,) int array. Box index of the push, -1 if no
                push is made.
            - `colors`: (num_steps+1, num_boxes, mat_size, mat_size) int array.
                Colors on the food boxes.

        """
        t = block_data['t']
        num_steps = int(np.floor(t.max()/self.dt))

        def get_trajectory(xyz):
            xy = xyz[:, :2]/arena_radius
            vals = []
            for i in range(num_steps+1):
                _xy = xy[(t>=i*self.dt)&(t<(i+1)*self.dt), :]
                if np.all(np.isnan(_xy[:, 0])) or np.all(np.isnan(_xy[:, 1])):
                    vals.append(-1) # data gap
                else:
                    _xy = np.nanmean(_xy, axis=0)
                    vals.append(self.arena.nearest_tile(_xy))
            vals = np.array(vals, dtype=int)
            return vals

        pos = get_trajectory(block_data['pos_xyz'])
        gaze = get_trajectory(block_data['gaze_xyz'])

        push_t, push_id = block_data['push_t'], block_data['push_id']
        push, success, box = [], [], []
        for i in range(1, num_steps+1): # one step fewer than pos and gaze
            push_idxs, = ((push_t>=i*self.dt)&(push_t<(i+1)*self.dt)).nonzero()
            _push = push_id[push_idxs]
            if len(np.unique(_push))>1:
                print(f'multiple boxes pushed at step {i}')
            push.append(len(_push)>0)
            success.append(np.any(block_data['push_flag'][push_idxs]))
            if len(_push)>0:
                b_idx = (int(_push[-1])+1)%3
                box.append(b_idx)
                pos[i] = self.arena.boxes[b_idx]
            else:
                box.append(-1)
        push = np.array(push, dtype=bool)
        success = np.array(success, dtype=bool)

        # actual colors are not provided in the raw data, will use uniform patch
        # estimated from cumulative cue
        colors = []
        _color_size = int(self.boxes[0].num_patches**0.5)
        for i in range(num_steps+1):
            _cues = block_data['cues'][(t>=i*self.dt)&(t<(i+1)*self.dt), :].mean(axis=0)
            _cues = np.floor(_cues*np.array([box.num_grades for box in self.boxes]))
            colors.append(np.tile(_cues[:, None, None], (1, _color_size, _color_size)))
        colors = np.array(colors, dtype=int)

        env_data = {
            'num_steps': num_steps,
            'pos': pos, 'gaze': gaze,
            'push': push, 'success': success, 'box': box,
            'colors': colors,
        }
        return env_data

    def extract_observation_and_action(self, env_data: dict) -> tuple[Array, Array]:
        r"""Extract observation and action sequences.

        Data gaps will be filled from previous frames. Colors of the boxes that
        are not being looked at will be marked with `num_grades` to represent
        'UNKNOWN'.

        Args
        ----
        env_data:
            Discretized data returned by `convert_experiment_data`.

        Returns
        -------
        observations: (num_steps+1, observation_dim) int
            Observations include monkey position and gaze, concatenated with all
            colors of food boxes.
        actions: (num_steps,) int
            Actions by the monkey.

        """
        num_steps = env_data['num_steps']

        observations = np.empty((num_steps+1, len(self.observation_space.nvec)), dtype=int)
        for t in range(num_steps+1):
            for i in range(2):
                if i==0:
                    vals = env_data['pos']
                if i==1:
                    vals = env_data['gaze']
                if vals[t]>=0:
                    observations[t, i] = vals[t]
                else:
                    if t>0:
                        observations[t, i] = observations[t-1, i] # previous frame
                    else:
                        observations[t, i] = vals[(vals>=0).nonzero()[0].min()] # first valid frame
            i = 2
            for b_idx, box in enumerate(self.boxes):
                if env_data['gaze'][t]==self.arena.boxes[b_idx]:
                    colors = env_data['colors'][t, b_idx].reshape(-1)
                else:
                    colors = box.num_grades
                observations[t, i:(i+box.num_patches)] = colors
                i += box.num_patches

        actions = np.empty((num_steps,), dtype=int)
        for t in range(num_steps):
            if env_data['push'][t]:
                pos = self.arena.boxes[env_data['box'][t]]
            else:
                pos = observations[t+1, 0]
            actions[t] = self.monkey.index_action(
                env_data['push'][t], pos, observations[t+1, 1],
            )
        return observations, actions

    def play_episode(self,
        pos, gaze, colors, push, success,
        num_steps: Optional[int] = None,
        figsize: tuple[float, float] = None,
        use_sec: bool = True,
    ) -> tuple[Figure, FuncAnimation]:
        r"""Creates animation of one episode game play.

        Args
        ----
        pos: int array of (num_steps+1,)
            Tile index of monkey positions.
        gaze: int array of (num_steps+1,)
            Tile index of monkey gazes.
        colors: int array of (num_steps+1, num_boxes, mat_size, mat_size)
            Colors shown on all boxes, in range [0, num_grades).
        push: bool array of (num_steps,)
            Whether the button is pushed.
        success: bool array of (num_steps,)
            Whether the food is delivered.
        num_steps:
            Number of steps to show from the start, can be shorter than
            `len(push)`.
        figsize:
            Figure size.
        use_sec:
            Whether to use seconds as time units. If ``False``, will use time
            steps as units.

        Returns
        -------
        fig, ani:
            Object handle for the figure and animation.

        """
        assert len(self.boxes)==3, "Only implemented for three boxes."
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
            _colors = np.zeros((box.mat_size, box.mat_size), dtype=int)
            _x, _y = np.array(self.arena.anchors[box.pos])*1.5
            _s = 0.2
            h_box = ax.imshow(
                _colors, extent=[_x-_s, _x+_s, _y-_s, _y+_s],
                vmin=-1, vmax=box.num_grades, cmap='RdYlBu_r',
                zorder=2,
            )
            h_boxes.append(h_box)
        h_counts = [
            ax.text(9/8, 0.93, '0/0', ha='center'),
            ax.text(-9/8, 0.93, '0/0', ha='center'),
            ax.text(0, -1.69, '0/0', ha='center'),
        ]
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.2])
        h_pos = ax.add_patch(Polygon(
            np.full((1, 2), fill_value=np.nan),
            edgecolor='none', facecolor='yellow', alpha=0.2,
        ))
        h_gaze = ax.scatter(np.nan, np.nan, s=100, marker='o', edgecolor='none', facecolor='blue')
        h_title = ax.set_title('')

        _xy = np.stack([
            np.array([np.cos(theta), np.sin(theta)])/(2*self.arena.resol)
            for theta in [i/3*np.pi+np.pi/6 for i in range(6)]
        ])
        counts = [(0, 0)]*3
        def update(t):
            for i, h_box in enumerate(h_boxes):
                h_box.set_data(colors[t, i])
            _pos = pos[t]
            h_pos.set_xy(_xy+self.arena.anchors[_pos])
            if t>0 and _pos in self.arena.boxes and push[t-1]:
                b_idx = self.arena.boxes.index(_pos)
                _success, _total = counts[b_idx]
                if success[t-1]:
                    _success += 1
                _total += 1
                counts[b_idx] = (_success, _total)
                h_counts[b_idx].set_text('{}/{}'.format(*counts[b_idx]))

                if success[t-1]:
                    h_pos.set_facecolor('darkred')
                else:
                    h_pos.set_facecolor('darkblue')
            else:
                h_pos.set_facecolor('yellow')
            h_gaze.set_offsets(self.arena.anchors[gaze[t]])
            h_title.set_text(r'$t$='+'{}'.format(
                '{:d} sec'.format(int(np.floor(t*self.dt))) if use_sec else t
            ))
            return *h_boxes, h_pos, h_gaze, *h_counts, h_title

        ani = FuncAnimation(fig, update, frames=range(num_steps+1), blit=True)
        return fig, ani

    def plot_occupancy(self,
        pos: Iterable[int], gaze: Iterable[int],
        figsize: tuple[float, float] = None,
    ):
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

    def food_probs(self,
        beliefs, states,
        p_s: BaseDistribution,
    ):
        num_steps = len(beliefs)-1
        nvec = self.state_space.nvec[-6:]
        box_states = np.stack(np.unravel_index(np.arange(np.prod(nvec)), nvec)).T

        probs = np.zeros((num_steps+1, 3)) # probability of food in each box
        for t in range(num_steps+1):
            belief = beliefs[t]
            _states = np.tile(states[t], (np.prod(nvec), 1))
            _states[:, 2:] = box_states
            p_s.set_param_vec(belief)
            with torch.no_grad():
                _probs = p_s.loglikelihoods(
                    torch.tensor(_states, dtype=torch.long),
                ).exp().numpy()
            for i in range(3):
                probs[t, i] = _probs[box_states[:, 2*i]==1].sum()
        return probs

    def play_traces(self,
        vals, num_steps = None,
        figsize: tuple[float, float] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        show_legend: bool = True,
        use_sec: bool = True,
    ) -> tuple[Figure, FuncAnimation]:
        r"""Creates animation for variable traces.

        Args
        ----
        vals: (num_steps+1, 3)
            Values in range [0, 1] for all three boxes, e.g. food availability,
            color cue or belief about the box.
        num_steps:
            Number of steps to show from the start.
        figsize:
            Figure size.
        xlabel, ylabel:
            Labels for x- and y- axis.
        show_legend:
            Whether to show legends.
        use_sec:
            Whether to use seconds as time units. If ``False``, will use time
            steps as units.

        Returns
        -------
        fig, ani:
            Object handle for the figure and animation.

        """
        assert self.arena.num_boxes==3, "Only implemented for three boxes."
        if figsize is None:
            figsize = (6, 2.5)
        fig = plt.figure(figsize=figsize)
        colors = ['violet', 'lime', 'tomato']

        if num_steps is None:
            num_steps = vals.shape[0]-1
        else:
            num_steps = min(num_steps, vals.shape[0]-1)

        ax = fig.add_axes([0.15, 0.25, 0.8, 0.6])
        h_lines = []
        for i in range(3):
            h, = ax.plot(np.nan, np.nan, color=colors[i])
            h_lines.append(h)
        if show_legend:
            ax.legend(['Box A', 'Box B', 'Box C'], fontsize='x-small', loc='upper right')
        ax.set_xlim([0, num_steps*self.dt if use_sec else num_steps])
        if xlabel is None:
            ax.set_xlabel('Time (sec)' if use_sec else '$t$')
        else:
            ax.set_xlabel(xlabel)
        ax.set_ylim([0, 1.05])
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        def update(t):
            for i, h in enumerate(h_lines):
                h.set_data(np.array([
                    np.arange(t+1)*self.dt if use_sec else np.arange(t+1),
                    vals[:t+1, i],
                ]))
            return *h_lines,

        ani = FuncAnimation(fig, update, frames=range(num_steps+1), blit=True)
        return fig, ani
