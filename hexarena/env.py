import numpy as np
import torch
from gymnasium import Env
from gymnasium.spaces import Discrete, MultiDiscrete
from jarvis.config import Config
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.animation import FuncAnimation

from typing import Optional
from collections.abc import Collection, Iterable
from irc.distribution import BaseDistribution, CompositeDistribution
from irc.buffer import Episode

from . import rcParams
from .arena import Arena
from .box import BaseFoodBox
from .monkey import Monkey
from .alias import (
    EnvParam, Observation, State,
    Figure, Axes, Artist, Array, Tensor,
)

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

        self.num_boxes = self.arena.num_boxes
        if boxes is None:
            boxes = [None]*self.num_boxes
        else:
            assert len(boxes)==self.num_boxes
        self.boxes: list[BaseFoodBox] = []
        for i in range(self.num_boxes):
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
            _nvec = x.state_space.nvec
            self._state_dims.append(len(_nvec))
            nvec += [*_nvec]
        self.state_space = MultiDiscrete(nvec)
        # observation: (*monkey_state, *boxes_colors)
        nvec = [*self.monkey.state_space.nvec]
        for box in self.boxes:
            nvec += [box.num_grades+1]*box.num_patches # additional grade for invisible
        self.observation_space = MultiDiscrete(nvec)
        # action: (push, move, look)
        self.action_space = self.monkey.action_space

    def _components(self) -> list:
        r"""Returns a list of environment components."""
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
        r"""Returns environment state."""
        state = tuple()
        for x in self._components():
            state += tuple([*x.get_state()])
        return state

    def set_state(self, state: State) -> None:
        r"""Sets environment state."""
        idx = 0
        for x, state_dim in zip(self._components(), self._state_dims):
            x.set_state([state[i] for i in range(idx, idx+state_dim)])
            idx += state_dim

    def reset(self, seed: Optional[int] = None) -> tuple[Observation, dict]:
        for i, x in enumerate(self._components()):
            x.reset(None if seed is None else seed+i)
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
        r"""Returns observation.

        The monkey has full knowledge of its own state, and only the box where
        the gaze is located gives valid color cues. Colors of other boxes are
        set to a constant `box.num_patches` to represent 'UNKNOWN'.

        """
        observation = tuple([*self.monkey.get_state()])
        for box in self.boxes:
            if self.monkey.gaze==box.pos:
                colors = box.colors.reshape(-1)
            else:
                colors = np.full((box.num_patches,), fill_value=box.num_grades, dtype=int)
            observation += tuple([*colors])
        return observation

    def _get_info(self) -> dict:
        r"""Returns information about environment."""
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
        arena_radius:
            Radius of the arena, (possibly) in mm.

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

        # actual colors are not provided in the raw data, will use a uniform
        # patch estimated from the cumulative cue
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
                else: # deal with data gap
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

    def convert_episode_data(self,
        episode: Episode,
    ) -> tuple[Array, Array, list[Optional[bool]]]:
        r"""Converts episode data to interpretable variables.

        Args
        ----
        episode:
            An episode returned by a BeliefAgent, e.g. `episode =
            agent.run_one_episode(...)`. 'infos' is needed to fully prepare the
            data.

        Returns
        -------
        pos, gaze: (num_steps+1,)
            Integer arrays for monkey position and gaze.
        rewarded: (num_steps+1,)
            Whether the agent is rewarded. See `plot_e_view` for more details.
        foods: (num_steps+1, num_boxes)
            Bool arrays for food status.
        colors: (num_steps+1, num_boxes, mat_size, mat_size)
            Integer arrays for color cues. All boxes regardless of the monkey
            gaze are prepared.
        counts: (num_steps+1, num_boxes, 2)
            Successful and total push counts of each box.

        """
        assert episode.infos is not None, "Need 'infos' to extract colors of all boxes."
        pos, gaze, rewarded, foods, colors, counts = [], [], [], [], [], []
        counts_t = np.zeros((self.num_boxes, 2))
        for t, info in enumerate(episode.infos):
            pos.append(info['pos'])
            gaze.append(info['gaze'])
            if t==0:
                rewarded.append(None)
            else:
                push, move, _ = self.monkey.convert_action(episode.actions[t-1])
                if push:
                    success = episode.rewards[t-1]>0
                    b_idx = self.arena.boxes.index(move)
                    rewarded.append(success)
                    counts_t[b_idx, 0] += int(success)
                    counts_t[b_idx, 1] += 1
                else:
                    rewarded.append(None)
            foods.append(info['foods'])
            colors.append(info['colors'])
            counts.append(counts_t.copy())
        pos = np.array(pos).astype(int)
        gaze = np.array(gaze).astype(int)
        foods = np.array(foods).astype(bool)
        colors = np.array(colors).astype(int)
        counts = np.stack(counts).astype(int)
        return pos, gaze, rewarded, foods, colors, counts

    def plot_e_view(self,
        ax: Axes,
        pos: int, gaze: int, rewarded: Optional[bool],
        foods: Optional[Array], colors: Optional[Array],
        counts: Optional[Array],
        artists: Optional[list[Artist]] = None,
    ) -> list[Artist]:
        r"""Plots experimenter view of one step.

        Monkey position, monkey gaze, food status and color cues will be plotted
        for one step.

        Args
        ----
        ax:
            Axis to plot figure.
        pos, gaze:
            Monkey position and gaze.
        rewarded:
            Whether a reward was just obtained. If ``True``, the position tile
            will be colored red. If ``False``, the position tile will be colored
            blue.
        foods: (num_boxes,)
            Bool array of food status in each box.
        colors: (num_boxes, mat_size, mat_size)
            Int array of color cues of each box.
        counts: (num_boxes, 2)
            Push counts since the beginning of the episode. Two numbers in each
            row are counts of successful pushes and total pushes.
        artists:
            Artist of objects to render. If not ``None``, properties of each
            artist will be updated accordingly.

        Returns
        -------
        artists:
            A list of artists, including:
            - A patch for monkey position,
            - A marker for monkey gaze,
            - Lines for food status.
            - Images for color cues.
            - Texts for push counts.

        """
        if rewarded is None:
            tile_color = 'yellow'
        else:
            tile_color = 'red' if rewarded else 'blue'
        if foods is None:
            foods_color = ['none']*self.num_boxes
        else:
            foods_color = ['red' if food else 'blue' for food in foods]
        if colors is None:
            colors = np.full((self.num_boxes, 1, 1), np.nan)
        if counts is None:
            texts = ['']*self.num_boxes
        else:
            texts = ['{}/{}'.format(*count) for count in counts]
        if artists is None:
            h_pos = self.arena.plot_tile(ax, pos, tile_color)
            h_pos.set_alpha(0.4)
            h_gaze = ax.scatter(
                *self.arena.anchors[gaze], s=100,
                marker='o', edgecolor='none', facecolor='blue',
            )
            h_foods, h_boxes, h_counts = [], [], []
            for i, box in enumerate(self.boxes):
                x, y = np.array(self.arena.anchors[box.pos])*1.5
                s = 0.2
                h_food, = ax.plot(
                    [x-s, x+s], [y+1.2*s, y+1.2*s],
                    color=foods_color[i], linewidth=2, zorder=2,
                )
                h_foods.append(h_food)
                h_box = ax.imshow(
                    colors[i], extent=[x-s, x+s, y-s, y+s], cmap='RdYlBu_r',
                    vmin=-1, vmax=box.num_grades, zorder=2,
                )
                h_boxes.append(h_box)
                h_count = ax.text(
                    x, y+np.sign(y)*1.8*s, texts[i],
                    ha='center', va='center',
                )
                h_counts.append(h_count)
        else:
            h_pos, h_gaze = artists[:2]
            h_foods = artists[2:(2+self.num_boxes)]
            h_boxes = artists[(2+self.num_boxes):(2+2*self.num_boxes)]
            h_counts = artists[(2+2*self.num_boxes):(2+3*self.num_boxes)]
            self.arena.plot_tile(ax, pos, tile_color, h_pos)
            h_gaze.set_offsets(self.arena.anchors[gaze])
            for i in range(self.num_boxes):
                h_foods[i].set_color(foods_color[i])
                h_boxes[i].set_data(colors[i])
                h_counts[i].set_text(texts[i])
        artists = [h_pos, h_gaze]+h_foods+h_boxes+h_counts
        return artists

    def play_e_view(self,
        pos, gaze, rewarded=None, foods=None, colors=None, counts=None,
        tmin: Optional[int] = None,
        tmax: Optional[int] = None,
        figsize: tuple[float, float] = None,
        use_sec: bool = True,
    ):
        r"""Creates animation of experimenter view.

        Args
        ----
        pos, gaze, rewarded, foods, colors, counts:
            Variables typically returned by `convert_episode_data`, covering
            time step interval [0, num_steps].
        tmin, tmax:
            Time index range to visualize.
        figsize:
            Figure size.
        use_sec:
            Whether to use seconds as time units. If ``False``, will use time
            steps as units.

        Returns
        -------
        fig, ani:
            Object of the figure and animation.

        """
        assert len(pos)==len(gaze)
        rewarded = [None]*len(pos) if rewarded is None else rewarded
        foods = [None]*len(pos) if foods is None else foods
        colors = [None]*len(pos) if colors is None else colors
        counts = [None]*len(pos) if counts is None else counts
        tmin = 0 if tmin is None else tmin
        tmax = len(pos) if tmax is None else tmax
        if figsize is None:
            figsize = (4.5, 4)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0.1, 0.05, 0.8, 0.9])
        self.arena.plot_mesh(ax)
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.6, 1.2])
        artists = self.plot_e_view(
            ax, pos[0], gaze[0], rewarded[0], foods[0], colors[0], counts[0],
        )
        h_title = ax.set_title('')

        def update(t, artists):
            artists = self.plot_e_view(
                ax, pos[t], gaze[t], rewarded[t],
                foods[t], colors[t], counts[t], artists,
            )
            h_title.set_text(r'$t$='+'{}'.format(
                '{:d} sec'.format(int(np.floor(t*self.dt))) if use_sec else t
            ))
            return *artists, h_title
        ani = FuncAnimation(
            fig, update, fargs=(artists,), frames=range(tmin, tmax), blit=True,
        )
        return fig, ani

    def convert_beliefs(self,
        p_s: CompositeDistribution,
        beliefs: Tensor,
    ):
        assert isinstance(p_s, CompositeDistribution)
        d_sets = [[0], [1]]+[[2*i+2, 2*i+3] for i in range(self.num_boxes)]
        assert p_s.d_sets==d_sets
        p_boxes = []
        for k in range(2+self.num_boxes):
            dist = p_s.s_dists[k]
            ps = []
            for belief in beliefs:
                param_vec = p_s.set_param_vec(k, belief)
                ps.append(dist.loglikelihoods(dist.all_xs, param_vec).exp())
            ps = torch.stack(ps).data.cpu().numpy()
            if k==0:
                p_pos = ps
            elif k==1:
                p_gaze = ps
            else:
                p_boxes.append(ps.reshape(len(beliefs), 2, -1))
        p_boxes = np.stack(p_boxes, axis=1)
        return p_pos, p_gaze, p_boxes

    def plot_a_view(self,
        ax_pos: Axes, p_pos: Array,
        ax_gaze: Axes, p_gaze: Array,
        axes_boxes: Collection[Axes], p_boxes: Array,
        kwargs_pos: Optional[dict] = None,
        kwargs_gaze: Optional[dict] = None,
        kwargs_box: Optional[dict] = None,
        artists: Optional[list[Artist]] = None,
    ) -> list[Artist]:
        r"""Plots agent view of one step.

        Args
        ----
        ax_pos, ax_gaze:
            Axis to plot `p_pos` and `p_gaze` respectively.
        p_pos, p_gaze: (num_tiles,)
            Probability distribution of monkey position and gaze.
        ax_boxes:
            Axes to plot `p_boxes`.
        p_boxes: (num_boxes, 2, num_levels)
            Joint probability of food status and cue level for each box.
        kwargs_pos, kwargs_gaze:
            Keyword arguments passed to `plot_map` for `p_pos` and `p_gaze`.
        kwargs_box:
            Keyword arguments for plotting box beliefs.

        """
        assert len(p_pos)==self.arena.num_tiles
        assert len(p_gaze)==self.arena.num_tiles
        assert len(axes_boxes)==self.num_boxes
        kwargs_pos = Config(kwargs_pos)
        kwargs_gaze = Config(kwargs_gaze)
        kwargs_box = Config(kwargs_box)
        kwargs_box.fill({
            'origin': 'lower', 'vmin': 0, 'vmax': p_boxes.max(), 'cmap': 'Reds',
        })
        if artists is None:
            h_pos = self.arena.plot_map(ax_pos, p_pos, clabel='$P_\mathrm{position}$', **kwargs_pos)
            h_gaze = self.arena.plot_map(ax_gaze, p_gaze, clabel='$P_\mathrm{gaze}$', **kwargs_gaze)
            h_boxes = []
            for i in range(self.num_boxes):
                ax = axes_boxes[i]
                h_boxes.append(ax.imshow(
                    p_boxes[i].T, aspect=2.5/self.boxes[i].num_levels, **kwargs_box,
                ))
                ax.set_xticks([0, 1])
                ax.set_xticklabels(['F', 'T'])
                ax.set_yticks([0, self.boxes[i].num_levels-1])
                if i==0:
                    ax.set_ylabel('Box cue', labelpad=-10)
                    ax.set_yticklabels(['min', 'max'])
                else:
                    ax.set_yticklabels([])
                ax.set_title('Box {}'.format(i+1))
            artists = h_pos+h_gaze+h_boxes
        else:
            n = self.arena.num_tiles
            h_pos = artists[:n]
            self.arena.plot_map(ax_pos, p_pos, h_tiles=h_pos)
            h_gaze = artists[n:2*n]
            self.arena.plot_map(ax_gaze, p_gaze, h_tiles=h_gaze)
            h_boxes = artists[2*n:2*n+self.num_boxes]
            for i in range(self.num_boxes):
                h_boxes[i].set_data(p_boxes[i].T)
        return artists

    def play_a_view(self,
        p_pos, p_gaze, p_boxes,
        tmin: Optional[int] = None,
        tmax: Optional[int] = None,
        figsize = None,
        use_sec: bool = True,
    ):
        r"""Creates animation of agent view.

        Args
        ----
        p_pos, p_gaze, p_boxes:
            Variables typically returned by `convert_beliefs`, covering
            time step interval [0, num_steps].
        tmin, tmax:
            Time index range to visualize.
        figsize:
            Figure size.
        use_sec:
            Whether to use seconds as time units. If ``False``, will use time
            steps as units.

        Returns
        -------
        fig, ani:
            Object of the figure and animation.

        """
        tmin = 0 if tmin is None else tmin
        tmax = len(p_pos) if tmax is None else tmax
        if figsize is None:
            figsize = (7, 6)
        fig = plt.figure(figsize=figsize)
        ax_pos = fig.add_axes([0.05, 0.05, 0.44, 0.55])
        ax_gaze = fig.add_axes([0.51, 0.05, 0.44, 0.55])

        n = self.num_boxes
        gap = 0.02
        width = (0.8-(n-1)*gap)/n
        axes_boxes = []
        for i in range(n):
            axes_boxes.append(fig.add_axes([0.15+i*(width+gap), 0.65, width, 0.25]))

        artists = self.plot_a_view(
            ax_pos, p_pos[0], ax_gaze, p_gaze[0], axes_boxes, p_boxes[0],
            kwargs_pos={'vmax': p_pos.max()}, kwargs_gaze={'vmax': p_gaze.max()},
            kwargs_box={'vmax': p_boxes.max()},
        )
        plt.colorbar(artists[-1], ax=axes_boxes, fraction=1/8, label='$P_\mathrm{box}$')
        _ax = fig.add_axes([0.05, 0.9, 0.1, 0]) # dummy axis since fig.text doesn't work
        _ax.axis('off')
        h_time = _ax.set_title('', fontsize='small')

        def update(t, artists):
            artists = self.plot_a_view(
                ax_pos, p_pos[t], ax_gaze, p_gaze[t],
                axes_boxes, p_boxes[t], artists=artists,
            )
            h_time.set_text(r'$t$='+'{}'.format(
                '{:d} sec'.format(int(np.floor(t*self.dt))) if use_sec else t
            ))
            return *artists, h_time
        ani = FuncAnimation(
            fig, update, fargs=(artists,), frames=range(tmin, tmax), blit=True,
        )
        return fig, ani

    def plot_occupancy(self,
        pos: Iterable[int], gaze: Iterable[int],
        figsize: tuple[float, float] = None,
    ):
        r""""Plots occupancy map of behavior.

        Args
        ----
        pos, gaze:
            Integers for monkey position and gaze at each time step.
        figsize:
            Figure size for each heat map.

        Returns
        -------
        fig_p, fig_g:
            Heat map figure for position and gaze histogram respectively.

        """
        figs = []
        for val, title in zip(
            [pos, gaze], ['Monkey position', 'Monkey gaze']
        ):
            counts = np.zeros(self.arena.num_tiles)
            for i in range(self.arena.num_tiles):
                counts[i] = (np.array(val)==i).sum()
            ps = counts/counts.sum()
            fig, ax = self.arena.plot_heatmap(ps, figsize, clabel='Probability')
            ax.set_title(title)
            figs.append(fig)
        fig_p, fig_g = figs
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
        assert self.num_boxes==3, "Only implemented for three boxes."
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
