import warnings
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Discrete, Box, Dict
from jarvis.config import Config
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from collections.abc import Sequence

from .arena import Arena
from .monkey import Monkey
from .box import BaseFoodBox
from .color import get_cmap
from .alias import EnvParam, Figure, Axes, Artist, Array
from .utils import get_food_avails


class ForagingEnv(Env):
    r"""Foraging environment with food boxes in a hexagonal arena.

    Args
    ----
    arena, monkey:
        The arena and monkey.
    boxes:
        Food boxes.
    time_cost:
        Cost for each time step.
    dt:
        Time step in unit of second.

    """

    def __init__(self,
        *,
        arena: Arena|dict|None = None,
        monkey: Monkey|dict|None = None,
        boxes: list[BaseFoodBox|dict|None]|None = None,
        dt: float = 1.,
        shared_param_names: list[str]|None = None,
    ):
        self.dt = dt
        if shared_param_names is None:
            shared_param_names = ['kappa']
        self.shared_param_names = shared_param_names

        if isinstance(arena, Arena):
            self.arena = arena
        else:
            arena = Config(arena).fill({'_target_': 'hexarena.arena.Arena'})
            self.arena: Arena = arena.instantiate()

        if isinstance(monkey, Monkey):
            self.monkey = monkey
        else:
            monkey = Config(monkey).fill({'_target_': 'hexarena.monkey.Monkey'})
            self.monkey: Monkey = monkey.instantiate(arena=self.arena)

        self.num_boxes = self.arena.num_boxes
        if boxes is None:
            boxes = [None]*self.num_boxes
        elif len(boxes)==1:
            boxes *= self.num_boxes
        assert len(boxes)==self.num_boxes
        self.boxes: list[BaseFoodBox] = []
        for i in range(self.num_boxes):
            if isinstance(boxes[i], BaseFoodBox):
                box = boxes[i]
            else:
                box = Config(boxes[i]).fill({'_target_': 'hexarena.box.PoissonBox'})
                box: BaseFoodBox = box.instantiate()
            box.dt = self.dt
            box.pos = self.arena.boxes[i]
            self.boxes.append(box)

        self.agt_space = self.monkey.state_space
        self.env_space = Dict({
            f'box_{i}': self.boxes[i].state_space for i in range(self.num_boxes)
        })
        self.obs_space = Dict({
            'color': Box(-np.inf, np.inf, shape=(2,)),
            'rewarded': Discrete(2),
        })
        # state: (*monkey_state, *boxes_state)
        self.state_space = Dict({
            'monkey': self.agt_space, **self.env_space,
        })
        # observation: (*monkey_state, *seen_color, rewarded)
        self.observation_space = Dict({
            'monkey': self.agt_space, **self.obs_space,
        })
        # action: (push, move, look)
        self.action_space = self.monkey.action_space

        self.rng = np.random.default_rng()

    def __repr__(self) -> str:
        a_str = str(self.arena)
        a_str = a_str[0].lower()+a_str[1:]
        return "Foraging in {} (time step {:.2g} sec)".format(a_str, self.dt)

    def _components(self) -> list[Monkey|BaseFoodBox]:
        r"""Returns a list of environment components."""
        return [self.monkey]+self.boxes

    def get_agt_state(self) -> tuple[int, int]:
        return self.monkey.get_state()

    def set_agt_state(self, state: tuple[int, int]) -> None:
        self.monkey.set_state(state)

    def get_env_state(self) -> dict:
        return {
            f'box_{i}': self.boxes[i].get_state() for i in range(self.num_boxes)
        }

    def set_env_state(self, state: dict) -> None:
        for i in range(self.num_boxes):
            self.boxes[i].set_state(state[f'box_{i}'])

    def _get_obs(self, rewarded: bool) -> dict:
        r"""Returns observation of environment.

        The monkey has full knowledge of its own state, a color observation from
        the box it is looking at, and whether reward is provided. When the
        monkey is not looking at any box, color observation `(0, 0)` will be
        returned instead.

        Args
        ----
        rewarded:
            Whether the monkey is rewarded at the current time step.

        Returns
        -------
        obs:
            A dictionary with keys
            - 'color': circular coordinates of seen color, see `Monkey.look` for
            more details.
            - 'rewarded': binary variable of whether rewarded or not.

        """
        box = next((box for box in self.boxes if self.monkey.gaze==box.pos), None)
        obs = {
            'color': (0., 0.) if box is None else self.monkey.look(box.colors),
            'rewarded': int(rewarded),
        }
        return obs

    def _get_observation_and_info(self, rewarded: bool) -> tuple[dict, dict]:
        r"""Returns observation and info for Gymnasium API.

        The method should be called at the end of `reset` or `step`.

        Args
        ----
        rewarded:
            Whether the monkey is rewarded, see `_get_obs` for more details.

        Returns
        -------
        observation:
            The observation combining both monkey state and its observation on
            the environment.
            - 'monkey': monkey state, see `Monkey.get_state` for more details.
            - 'color': seen color, see `_get_obs` for more details.
            - 'rewarded': rewarded condition, see `_get_obs` for more details.
        info:
            A dictionary containing POMDP related information, with keys:
            - 'agt_state': the monkey state.
            - 'env_state': the states of food boxes, see `get_env_state` for
            more details.
            - 'obs': observation of environment, see `_get_obs` for more details.
            - 'colors': array of shape `(n_boxes, h, w)` for color patterns on
            all boxes.

        """
        agt_state = self.get_agt_state()
        env_state = self.get_env_state()
        obs = self._get_obs(rewarded)
        observation = {'monkey': agt_state, **obs}
        info = {
            'agt_state': agt_state, 'env_state': env_state, 'obs': obs,
            'colors': np.stack([box.colors for box in self.boxes]),
        }
        return observation, info

    def get_param(self) -> EnvParam:
        r"""Returns environment parameters."""
        param = self.monkey.get_param()
        for name in self.shared_param_names:
            val, *_ = self.boxes[0]._get_param(name)
            param += val
        for box in self.boxes:
            for name in box.param_names:
                if name not in self.shared_param_names:
                    val, *_ = box._get_param(name)
                    param += val
        return param

    def set_param(self, param: EnvParam) -> None:
        r"""Sets environment parameter."""
        n = len(self.monkey.get_param())
        self.monkey.set_param(param[:n])
        c = n
        for name in self.shared_param_names:
            val, *_ = self.boxes[0]._get_param(name)
            n = len(val)
            for box in self.boxes:
                box._set_param(name, param[c:c+n])
            c += n
        for box in self.boxes:
            for name in box.param_names:
                if name not in self.shared_param_names:
                    val, *_ = box._get_param(name)
                    n = len(val)
                    box._set_param(name, param[c:c+n])
                    c += n

    def param_bounds(self) -> tuple[EnvParam, EnvParam]:
        r"""Returns lower and upper bound of environment parameters."""
        low, high = self.monkey.param_bounds()
        for name in self.shared_param_names:
            _, _low, _high = self.boxes[0]._get_param(name)
            low += _low
            high += _high
        for box in self.boxes:
            for name in box.param_names:
                if name not in self.shared_param_names:
                    _, _low, _high = box._get_param(name)
                    low += _low
                    high += _high
        return low, high

    def reset(self, seed: int|None = None, options: dict|None = None) -> tuple[dict, dict]:
        r"""Resets environment.

        Random number generator is reset according to `seed`, and linked to `rng`
        of each component.

        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        for x in self._components():
            x.rng = self.rng
            x.reset()
        observation, info = self._get_observation_and_info(False) # no reward at reset
        return observation, info

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        push, move, look = self.monkey.convert_action(action)
        reward = self.monkey.step(push, move, look)
        rewarded = False
        for box in self.boxes:
            _reward = box.step(push and move==box.pos)
            reward += _reward
            rewarded |= _reward>0
        observation, info = self._get_observation_and_info(rewarded)
        return observation, reward, False, False, info

    def convert_experiment_data(self, block_data: dict, arena_radius=1860.) -> dict:
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
            - `pos`: int, `(num_steps+1,)`. Monkey position in `[-1, num_tiles)`,
            with '-1' for missing data.
            - `gaze`: int, `(num_steps+1,)`. Gaze position in `[-1, num_tiles)`,
            with '-1' for missing data.
            - `colors`: float, `(num_steps+1, num_boxes, height, width)`. Colors
            on the food boxes.
            - `push`: int, `(num_steps,)`. Index of the box pushed at each time
            step. The values are in `[-1, num_boxes)` with ``-1`` for no push.
            - `rewarded`: bool, `(num_steps+1,)`. Whether the monkey is rewarded
            or not. The first element is always ``False`` corresponding to the
            environment reset.
            - `foods`: bool, `(num_steps+1, num_boxes)`. Food availability at
            each time step for all boxes.

        """
        t = block_data['t']
        num_steps = int(np.floor(t.max()/self.dt))

        def get_trajectory(xyz):
            r"""Extracts 2D trajectories as in tile indices.

            Args
            ----
            xyz: (n_stamps, 3), float
                Raw 3D coordinates in mm with time stamps marked in `t`.

            Returns
            -------
            vals: (n_steps,), int
                Tile indices at each time step of size `self.dt`. '-1' for
                missing data.

            """
            xy = xyz[:, :2]/arena_radius
            vals = []
            for i in range(num_steps+1):
                _xy = xy[(t>=(i-0.5)*self.dt)&(t<(i+0.5)*self.dt), :]
                if np.all(np.isnan(_xy[:, 0])) or np.all(np.isnan(_xy[:, 1])):
                    vals.append(-1) # data gap
                else:
                    _xy = np.nanmean(_xy, axis=0)
                    vals.append(self.arena.nearest_tile(_xy))
            vals = np.array(vals, dtype=int)
            return vals
        pos = get_trajectory(block_data['pos_xyz'])
        gaze = get_trajectory(block_data['gaze_xyz'])

        # actual color movie is not saved, using independent cue array generated
        # at each time step instead
        self.reset(seed=0) # remove stochasticity of box 'get_colors' method
        colors = []
        for i in range(num_steps+1):
            cues = np.clip(block_data['cues'][t>=i*self.dt, :][0], a_min=0, a_max=0.999)
            colors.append([])
            for box, cue in zip(self.boxes, cues):
                colors[-1].append(box.get_colors(cue))
        colors = np.array(colors, dtype=float)

        push_t, push_idx = block_data['push_t'], block_data['push_idx']
        push, rewarded = [], [False]
        for i in range(num_steps):
            t_idxs, = ((push_t>=i*self.dt)&(push_t<(i+1)*self.dt)).nonzero()
            pushes = push_idx[t_idxs]
            if len(pushes)==0: # no push
                push.append(-1)
                rewarded.append(False)
            else:
                if len(np.unique(pushes))>1:
                    warnings.warn(
                        f"Multiple boxes pushed at step {i}, only the last push will be recorded. "
                        f"Consider using smaller time step (current dt={self.dt})."
                    )
                push.append(pushes[-1].item())
                pos[i+1] = self.arena.boxes[push[-1]]
                rewarded.append(np.any(block_data['push_flag'][t_idxs]))
        push = np.array(push).astype(int)
        rewarded = np.array(rewarded).astype(bool)

        foods = []
        for b_idx in range(self.num_boxes):
            if block_data['gamma_shape']==1:
                tau = block_data['taus'][b_idx].item()
                first_rewarded = block_data['push_flag'][block_data['push_idx']==b_idx][0].item()
            else:
                tau, first_rewarded = None, None
            foods.append(get_food_avails(
                block_data['push_t'][block_data['push_idx']==b_idx],
                block_data['intervals'][b_idx], num_steps, dt=self.dt,
                tau=tau, first_rewarded=first_rewarded,
            ))
        foods = np.stack(foods, axis=1)

        env_data = {
            'pos': pos, 'gaze': gaze, 'colors': colors,
            'push': push, 'rewarded': rewarded, 'foods': foods,
        }
        return env_data

    def convert_simulated_data(self,
        observations: list[dict],
        infos: list[dict],
        actions: Sequence[int],
    ) -> dict:
        r"""Converts an episode ran by the environment.

        Args
        ----
        infos:
            `info` returned by `reset` and `step` in one episode.

        Returns
        -------
        env_data:
            Converted data with the same format as returned by
            `convert_experiment_data`.

        """
        pos, gaze, colors, foods = [], [], [], []
        for info in infos:
            pos.append(info['agt_state'][0])
            gaze.append(info['agt_state'][1])
            colors.append(info['colors'])
            foods.append([
                info['env_state'][f'box_{i}']['food']
                for i in range(self.num_boxes)
            ])
        pos = np.array(pos).astype(int)
        gaze = np.array(gaze).astype(int)
        colors = np.stack(colors).astype(float)
        foods = np.stack(foods).astype(bool)
        push = []
        for action in actions:
            _push, _move, _ = self.monkey.convert_action(action)
            if _push:
                push.append(self.arena.boxes.index(_move))
            else:
                push.append(-1)
        push = np.array(push).astype(int)
        rewarded = np.array([o['rewarded'] for o in observations]).astype(bool)
        env_data = {
            'pos': pos, 'gaze': gaze, 'colors': colors,
            'push': push, 'rewarded': rewarded, 'foods': foods,
        }
        return env_data

    def _count_pushes(self,
        push: Array, rewarded: Array,
    ) -> Array:
        r"""Counts successful and total pushes for all boxes.

        Args
        ----
        push: int, `(num_steps,)`
            Index of the box pushed, see `convert_experiment_data` for more
            details.
        rewarded: bool, `(num_steps+1,)`
            Whether the monkey is rewarded or not, see `convert_experiment_data`
            for more details.

        Returns
        -------
        counts: int, `(num_steps+1, num_boxes, 2)`
            `counts[..., 0]` is the counts of successful pushes for each box.
            `counts[..., 1]` is the counts of total pushes for each box.

        """
        num_steps = len(push)
        counts = np.zeros((num_steps+1, self.num_boxes, 2), dtype=int)
        for t in range(num_steps):
            counts[t+1] = counts[t]
            if rewarded[t+1]: # successful count
                counts[t+1, push[t], 0] += 1
            if push[t]>=0: # total count
                counts[t+1, push[t], 1] += 1
        return counts

    def extract_observation_action_reward(self, env_data: dict) -> tuple[Array, Array, Array]:
        r"""Extract observation, action and reward sequences.

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
        rewards: (num_steps,) float
            Rewards computed based on monkey action and push outcomes.

        """
        raise NotImplementedError
        num_steps = env_data['num_steps']

        self.reset(seed=0) # remove stochasticity of monkey 'look' method
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
                if observations[t, 1]==self.arena.boxes[b_idx]:
                    color = self.monkey.look(env_data['colors'][t, b_idx])
                else:
                    color = self.monkey.num_grades
                observations[t, i+b_idx] = color
            i += self.num_boxes
            observations[t, i] = int(t>0 and env_data['success'][t-1])

        actions = np.empty((num_steps,), dtype=int)
        for t in range(num_steps):
            if env_data['push'][t]:
                pos = self.arena.boxes[env_data['box_idx'][t]]
            else:
                pos = observations[t+1, 0]
            actions[t] = self.monkey.index_action(
                env_data['push'][t], pos, observations[t+1, 1],
            )

        rewards = np.empty((num_steps,), dtype=float)
        for t in range(num_steps):
            pos, gaze = env_data['pos'][t], env_data['gaze'][t]
            self.monkey.set_state((pos, gaze))
            push, move, look = env_data['push'][t], env_data['pos'][t+1], env_data['gaze'][t+1]
            reward = self.monkey.step(push, move, look)
            reward += -self.time_cost
            if env_data['success'][t]:
                for box in self.boxes:
                    if move==box.pos:
                        reward += box.reward
            rewards[t] = reward

        return observations, actions, rewards

    def summarize_episodes(self,
        observations: list[Sequence[dict]],
        actions: list[Sequence[int]],
    ) -> tuple[Array, Array, Array, Array]:
        r"""Summarizes multiple episodes.

        Args
        ----
        observations:
            Observations of all episodes. Each item is of length 'T+1'.
        actions:
            Actions of all episodes. Each item is of length 'T'.

        Returns
        -------
        push_freq, food_freq: (num_boxes,)
            Push frequency and food frequency of each box. Success rate can be
            computed by `food_freq/push_freq`.
        pos_hist, gaze_hist: (num_tiles,)
            Position and gaze histogram.

        """
        raise NotImplementedError
        assert len(observations)==len(actions), "Number of episodes inconsistent."
        num_episodes = len(observations)
        push_count = np.zeros(self.arena.num_boxes)
        food_count = np.zeros(self.arena.num_boxes)
        pos_hist = np.zeros(self.arena.num_tiles)
        gaze_hist = np.zeros(self.arena.num_tiles)
        for e_idx in range(num_episodes):
            for t, action in enumerate(actions[e_idx]):
                push, move, look = self.monkey.convert_action(action)
                if push:
                    b_idx = self.arena.boxes.index(move)
                    push_count[b_idx] += 1
                    food_count[b_idx] += observations[e_idx][t+1][-1]
                pos_hist[move] += 1
                gaze_hist[look] += 1
        num_steps = sum(pos_hist)
        push_freq = push_count/num_steps
        food_freq = food_count/num_steps
        pos_hist /= num_steps
        gaze_hist /= num_steps
        return push_freq, food_freq, pos_hist, gaze_hist

    def plot_arena(self,
        ax: Axes, pos: int, gaze: int, rewarded: bool|None,
        foods: Array|None, colors: Array|None, counts: Array|None,
        artists: list[Artist]|None = None,
    ) -> list[Artist]:
        r"""Plots experimenter view of one step.

        Monkey position, monkey gaze, food status and color cues will be plotted
        for one step.

        Args
        ----
        ax:
            Axis to plot figure.
        pos, gaze:
            Tile index for monkey position and gaze.
        rewarded:
            Whether the monkey was rewarded from last action. The position tile
            will be colored red if ``True``, blue if ``False``, and yellow if
            ``None``.
        foods: bool, `(num_boxes,)`
            Food status in each box. If provided, red (``True``) or blue
            (``False``) bars will be plotted.
        colors: (num_boxes, height, width)
            Float array of color cues of each box, in [0, 1).
        counts: (num_boxes, 2)
            Push counts since the beginning of the episode. Two numbers in each
            row are counts of successful pushes and total pushes.
        artists:
            Artist of objects to render. If provided, properties of each artist
            will be updated accordingly.

        Returns
        -------
        artists:
            A list of artists, including:
            - A patch for monkey position,
            - A marker for monkey gaze,
            - Bars for food status.
            - Images for color cues.
            - Texts for push counts.

        """
        if rewarded is None:
            tile_color = 'yellow'
        else:
            tile_color = 'red' if rewarded else 'blue'
        if foods is None or colors is None:
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
                *(self.arena.anchors[gaze] if gaze>=0 else (np.nan, np.nan)),
                s=200/self.arena.resol, marker='o', edgecolor='none', facecolor='green',
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
                *_, h, w = colors.shape
                if h>w:
                    extent = [x-s*w/h, x+s*w/h, y-s, y+s]
                else:
                    extent = [x-s, x+s, y-s*h/w, y+s*h/w]
                h_box = ax.imshow(
                    colors[i], extent=extent, zorder=2,
                    vmin=0, vmax=1, cmap=get_cmap(), interpolation='nearest',
                )
                h_boxes.append(h_box)
                h_count = ax.text(
                    x, y+np.sign(y)*1.8*s, texts[i], ha='center', va='center',
                )
                h_counts.append(h_count)
        else:
            h_pos, h_gaze = artists[:2]
            h_foods = artists[2:(2+self.num_boxes)]
            h_boxes = artists[(2+self.num_boxes):(2+2*self.num_boxes)]
            h_counts = artists[(2+2*self.num_boxes):(2+3*self.num_boxes)]
            self.arena.plot_tile(ax, pos, tile_color, h_tile=h_pos)
            h_gaze.set_offsets(self.arena.anchors[gaze] if gaze>=0 else (np.nan, np.nan))
            for i in range(self.num_boxes):
                h_foods[i].set_color(foods_color[i])
                h_boxes[i].set_data(colors[i])
                h_counts[i].set_text(texts[i])
        artists = [h_pos, h_gaze]+h_foods+h_boxes+h_counts
        return artists

    def plot_beliefs(self,
        axes: Sequence[Axes], p_boxes: Array, p_max: float|None = None,
        artists: list[Artist]|None = None,
    ) -> list[Artist]:
        r"""Plots agent view of one step.

        Beliefs about box states including both food availability and cue level,
        are plotted with a shared color map.

        Args
        ----
        axes:
            Axis object for each box.
        p_boxes: float, (num_boxes, 2, num_levels)
            Probability values for each food-cue combination. Values over the
            last two dimensions will sum up to 1.
        p_max:
            Maximum value to determine color maps.
        artists:
            Artist of objects to render. See `plot_arena` for more details.

        Returns
        -------
        artists:
            A list of artists containing heat maps for each box.

        """
        raise NotImplementedError
        assert len(axes)==self.num_boxes
        if artists is None:
            if p_max is None:
                p_max = np.nanmax(p_boxes)
            h_boxes = []
            for i, ax in enumerate(axes):
                n_levels = self.boxes[i].num_levels
                for k in range(2):
                    h, = ax.plot(
                        (np.arange(n_levels)+0.5)/n_levels, p_boxes[i, k],
                        color='blue' if k==0 else 'red',
                    )
                    h_boxes.append(h)
                ax.set_xticks([0.5/n_levels, (n_levels-0.5)/n_levels])
                if i==self.num_boxes-1:
                    ax.set_xticklabels(['min', 'max'])
                    ax.set_xlabel('Color cue', labelpad=-10)
                else:
                    ax.set_xticklabels([])
                ax.set_ylim([-0.01, 1.05*p_max])
                ax.set_ylabel('Box {}'.format(i+1))
            ax = axes[0]
            ax.set_title('Belief')
            ax.legend(h_boxes[:2], ['No food', 'With food'], loc='upper center', fontsize='x-small')
            artists = h_boxes
        else:
            h_boxes = artists
            for i in range(self.num_boxes):
                for k in range(2):
                    h_boxes[2*i+k].set_ydata(p_boxes[i, k])
        return artists

    def play_episode(self,
        pos: Array, gaze: Array|None = None,
        colors: Array|None = None,
        push: Array|None = None,
        rewarded: Array|None = None,
        foods: Array|None = None,
        beliefs: Array|None = None,
        tmin: int|None = None, tmax: int|None = None,
        figsize: tuple[float, float]|None = None,
        use_sec: bool = True,
    ) -> tuple[Figure, FuncAnimation]:
        r"""Creates animation of one episode.

        See more details in `plot_arena` and `plot_beliefs`.

        Args
        ----
        pos, gaze, colors, push, rewarded, foods:
            Episode data formatted by the environment, see
            `convert_experiment_data` for more details.
        beliefs:
            Beliefs about the environment.
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
        n_steps = len(pos)-1
        if gaze is None:
            gaze = np.full((n_steps+1,), fill_value=-1, dtype=int)
        else:
            assert gaze.shape==(n_steps+1,)
        if colors is None:
            colors = [None]*(n_steps+1)
        else:
            assert colors.shape[:2]==(n_steps+1, self.num_boxes)
        if push is None or rewarded is None:
            rewarded = [None]*(n_steps+1)
            counts = [None]*(n_steps+1)
        else:
            assert push.shape==(n_steps,) and rewarded.shape==(n_steps+1,)
            rewarded = [None]+[rewarded[t+1] if push[t]>=0 else None for t in range(n_steps)]
            counts = self._count_pushes(push, rewarded)
        if foods is None:
            foods = [None]*(n_steps+1)
        else:
            assert foods.shape==(n_steps+1, self.num_boxes)
        if beliefs is not None:
            raise NotImplementedError("Visualization of beliefs not implemented.")

        tmin = 0 if tmin is None else tmin
        tmax = n_steps+1 if tmax is None else min(tmax, n_steps+1)
        if figsize is None:
            figsize = (4.5, 4) if beliefs is None else (7.5, 4)
        fig = plt.figure(figsize=figsize)

        if beliefs is None:
            ax = fig.add_axes([0.1, 0.05, 0.8, 0.9])
        else:
            ax = fig.add_axes([0.05, 0.05, 0.45, 0.9])
        self.arena.plot_mesh(ax)
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.6, 1.2])
        artists_a = self.plot_arena(
            ax, pos[tmin], gaze[tmin], rewarded[tmin], foods[tmin], colors[tmin], counts[tmin],
        )
        h_title = ax.set_title('')

        if beliefs is None:
            artists_b = []
        else:
            n = self.num_boxes
            gap = 0.05
            h = (0.8-(n-1)*gap)/n
            axes = []
            for i in range(n):
                axes.append(fig.add_axes([0.6, 0.1+(n-i-1)*(h+gap), 0.35, h]))
            artists_b = self.plot_beliefs(
                axes, p_boxes[tmin], p_max=p_boxes[tmin:tmax].max(),
            )

        def update(t, artists_a, artists_b):
            artists_a = self.plot_arena(
                ax, pos[t], gaze[t], rewarded[t],
                foods[t], colors[t], counts[t], artists_a,
            )
            if beliefs is not None:
                raise NotImplementedError
                artists_b = self.plot_beliefs(
                    axes, p_boxes[t], artists=artists_b,
                )
            h_title.set_text(r'$t$='+'{}'.format(
                '{:d} sec'.format(int(np.floor(t*self.dt))) if use_sec else t
            ))
            return *artists_a, *artists_b, h_title
        ani = FuncAnimation(
            fig, update, fargs=(artists_a, artists_b), frames=range(tmin, tmax), blit=True,
        )
        return fig, ani

    def play_traces(self,
        vals, num_steps = None,
        figsize: tuple[float, float]|None = None,
        xlabel: str|None = None,
        ylabel: str|None = None,
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
        raise NotImplementedError
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
        return fig, FuncAnimation
