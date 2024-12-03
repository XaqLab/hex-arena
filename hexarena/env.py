import warnings
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Discrete, MultiDiscrete
from jarvis.config import Config
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from collections.abc import Sequence

from .arena import Arena
from .monkey import Monkey
from .box import BaseFoodBox
from .color import get_cmap
from .alias import EnvParam, Observation, State, Figure, Axes, Artist, Array

class ForagingEnv(Env):
    r"""Foraging environment with food boxes in a hexagonal arena.

    Args
    ----
    arena, monkey:
        The arena and monkey object.
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
        time_cost: float = 0.,
        dt: float = 1.,
    ):
        self.time_cost = time_cost
        self.dt = dt

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

        # state: (*monkey_state, *boxes_state)
        self._state_dims, nvec = (), ()
        for x in self._components():
            _nvec = x.state_space.nvec
            self._state_dims += (len(_nvec),)
            nvec += (*_nvec,)
        self.state_space = MultiDiscrete(nvec)
        # observation: (*monkey_state, *seen_colors, rewarded)
        nvec = (*self.monkey.state_space.nvec,)
        nvec += (self.monkey.num_grades+1,)*self.num_boxes
        nvec += (2,)
        self.observation_space = MultiDiscrete(nvec)
        # action: (push, move, look)
        self.action_space = self.monkey.action_space

        self.known_dim = len(self.monkey.state_space.nvec)

        self.rng = np.random.default_rng()

    def __repr__(self) -> str:
        a_str = str(self.arena)
        a_str = a_str[0].lower()+a_str[1:]
        return "Foraging in {} (time step {:.2g} sec)".format(a_str, self.dt)

    @property
    def spec(self) -> dict:
        return {
            '_target_': 'hexarena.env.ForagingEnv',
            'arena': self.arena.spec,
            'monkey': self.monkey.spec,
            'boxes': [box.spec for box in self.boxes],
            'time_cost': self.time_cost, 'dt': self.dt,
        }

    def _components(self) -> list[Monkey|BaseFoodBox]:
        r"""Returns a list of environment components."""
        return [self.monkey]+self.boxes

    def get_param(self) -> EnvParam:
        r"""Returns environment parameters."""
        param = [self.time_cost]
        for x in self._components():
            param += [*x.get_param()]
        return param

    def set_param(self, param: EnvParam) -> None:
        r"""Sets environment parameter."""
        self.time_cost, = param[:1]
        c = 1
        for x in self._components():
            val = x.get_param()
            n = len(val)
            val = param[c:c+n]
            x.set_param(val)
            c += n

    def param_bounds(self) -> tuple[EnvParam, EnvParam]:
        r"""Returns lower and upper bound of environment parameters."""
        param_low, param_high = [0], [np.inf]
        for x in self._components():
            low, high = x.param_bounds()
            param_low += [*low]
            param_high += [*high]
        return param_low, param_high

    def get_state(self) -> State:
        r"""Returns environment state."""
        state = ()
        for x in self._components():
            state += (*x.get_state(),)
        return state

    def set_state(self, state: State) -> None:
        r"""Sets environment state."""
        idx = 0
        for x, state_dim in zip(self._components(), self._state_dims):
            x.set_state([state[i] for i in range(idx, idx+state_dim)])
            idx += state_dim

    def reset(self, seed: int|None = None) -> tuple[Observation, dict]:
        r"""Resets environment.

        Random number generator is reset according to `seed`, and linked to `rng`
        of each component.

        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        for x in self._components():
            x.rng = self.rng
            x.reset()
        observation = self._get_observation(False)
        info = self._get_info()
        return observation, info

    def step(self, action: int) -> tuple[Observation, float, bool, bool, dict]:
        reward, terminated, truncated = -self.time_cost, False, False
        push, move, look = self.monkey.convert_action(action)
        reward += self.monkey.step(push, move, look)
        rewarded = False
        for box in self.boxes:
            _reward = box.step(push and move==box.pos)
            reward += _reward
            rewarded |= _reward>0
        observation = self._get_observation(rewarded)
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def _get_observation(self, rewarded: bool) -> Observation:
        r"""Returns observation.

        The monkey has full knowledge of its own state, and only the box where
        the gaze is at gives valid color cues. Colors of other boxes are set to
        a constant `monkey.num_patches` to represent 'UNKNOWN'.

        """
        observation = (*self.monkey.get_state(),)
        for box in self.boxes:
            if self.monkey.gaze==box.pos:
                color = self.monkey.look(box.colors)
            else:
                color = self.monkey.num_grades
            observation += (color,)
        observation += (int(rewarded),)
        return observation

    def _get_info(self) -> dict:
        r"""Returns information about environment."""
        info = {
            'pos': self.monkey.pos, 'gaze': self.monkey.gaze,
            'foods': [box.food for box in self.boxes],
            'levels': [box.level for box in self.boxes],
            'colors': [box.colors for box in self.boxes],
        }
        return info

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
            - `num_steps`: int. Number of steps in the block.
            - `pos`: (num_steps+1,) int array. Monkey position.
            - `gaze`: (num_steps+1,) int array. Gaze position.
            - `push`: (num_steps,) bool array. Whether the button is pushed.
            - `success`: (num_steps,) bool array. Whether the food is obtained.
            - `box`: (num_steps,) int array. Box index of the push, -1 if no
                push is made.
            - `counts`: (num_steps+1, num_boxes, 2) int array. Push and success
                counts for each box.
            - `colors`: (num_steps+1, num_boxes, mat_size, mat_size) int array.
                Colors on the food boxes.

        """
        t = block_data['t']
        num_steps = int(np.floor(t.max()/self.dt))-1

        def get_trajectory(xyz):
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

        push_t, push_idx = block_data['push_t'], block_data['push_idx']
        push, success, box, t_wait = [], [], [], [np.zeros(self.num_boxes)]
        for i in range(1, num_steps+1): # one step fewer than pos and gaze
            t_idxs, = ((push_t>=i*self.dt)&(push_t<(i+1)*self.dt)).nonzero()
            pushes = push_idx[t_idxs]
            if len(np.unique(pushes))>1:
                warnings.warn(
                    f"""Multiple boxes pushed at step {i}, only the last push will be recorded."""
                )
            push.append(len(pushes)>0)
            success.append(np.any(block_data['push_flag'][t_idxs]))
            t_wait.append(t_wait[-1]+1)
            if len(pushes)>0:
                b_idx = pushes[-1]
                box.append(b_idx)
                pos[i] = self.arena.boxes[b_idx] # set monkey position
                t_wait[-1][b_idx] = 0
            else:
                box.append(-1)
        push = np.array(push, dtype=bool)
        success = np.array(success, dtype=bool)
        box = np.array(box, dtype=int)
        t_wait = np.stack(t_wait).astype(int)
        counts = np.stack([
            np.cumsum(
                np.stack([success, push], axis=1)*(box==b_idx)[:, None], axis=0,
            ) for b_idx in range(self.num_boxes)
        ], axis=1)
        counts = np.concatenate([
            np.zeros((1, *counts.shape[1:])), counts,
        ], axis=0).astype(int)

        # actual colors are not provided in the raw data, will use a uniform
        # patch estimated from the cumulative cue
        # TODO update with true pattern
        colors = []
        mat_size = self.boxes[0].mat_size
        for i in range(num_steps+1):
            _cues = np.clip(block_data['cues'][t>=(i+1)*self.dt, :][0], a_min=0, a_max=0.999)
            _cues = np.floor(_cues*np.array([box.num_grades for box in self.boxes]))
            colors.append(np.tile(_cues[:, None, None], (1, mat_size, mat_size)))
        colors = np.array(colors, dtype=int)

        env_data = {
            'num_steps': num_steps, 'pos': pos, 'gaze': gaze,
            'push': push, 'success': success, 'box': box, 't_wait': t_wait, 'counts': counts,
            'colors': colors,
        }
        return env_data

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
            for b_idx, box in enumerate(self.boxes): # TODO update with true pattern
                if observations[t, 1]==self.arena.boxes[b_idx]:
                    colors = env_data['colors'][t, b_idx].reshape(-1)
                else:
                    colors = box.num_grades
                observations[t, i:(i+box.num_patches)] = colors
                i += box.num_patches
            i = 5
            observations[t, i] = int(t>0 and env_data['success'][t-1])

        actions = np.empty((num_steps,), dtype=int)
        for t in range(num_steps):
            if env_data['push'][t]:
                pos = self.arena.boxes[env_data['box'][t]]
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
        observations: list[Sequence[Observation]],
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
            Whether a reward was just obtained. The position tile will be
            colored red if ``True``, blue if ``False``, and yellow if not
            provided.
        foods: (num_boxes,)
            Bool array of food status in each box. If provided, red or blue bars
            will be plotted.
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
                *(self.arena.anchors[gaze] if gaze>=0 else (np.nan, np.nan)), s=100,
                marker='o', edgecolor='none', facecolor='green',
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
                    colors[i], extent=[x-s, x+s, y-s, y+s], zorder=2,
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
        assert len(axes)==self.num_boxes
        if artists is None:
            if p_max is None:
                p_max = np.nanmax(p_boxes)
            h_boxes = []
            for i, ax in enumerate(axes):
                h_boxes.append(ax.imshow(
                    p_boxes[i], aspect=self.boxes[i].num_levels/4, origin='lower',
                    vmin=0, vmax=p_max, cmap='Reds',
                ))
                ax.set_xticks([0, self.boxes[i].num_levels-1])
                if i==self.num_boxes-1:
                    ax.set_xticklabels(['min', 'max'])
                    ax.set_xlabel('Cue level', labelpad=-10)
                else:
                    ax.spines['bottom'].set_visible(False)
                    ax.set_xticklabels([])
                ax.set_yticks([0, 1])
                ax.set_yticklabels(
                    ['Empty', 'Food'], fontsize='xx-small',
                    rotation=90, va='center',
                )
                ax.set_ylabel('Box {}'.format(i+1))
            artists = h_boxes
        else:
            h_boxes = artists
            for i in range(self.num_boxes):
                h_boxes[i].set_data(p_boxes[i])
        return artists

    def play_episode(self,
        pos: Array, gaze: Array,
        rewarded=None, foods=None, colors=None, counts=None, p_boxes=None,
        tmin: int|None = None, tmax: int|None = None,
        figsize: tuple[float, float]|None = None,
        use_sec: bool = True,
    ) -> tuple[Figure, FuncAnimation]:
        r"""Creates animation of one episode.

        See more details in `plot_arena` and `plot_beliefs`.

        Args
        ----
        pos, gaze: int, (num_steps+1,)
            Tile index for monkey position and gaze.
        rewarded: bool|None, (num_steps+1,)
            Whether the agent is rewarded, ``None`` stands for no push.
        foods: bool, (num_steps+1, num_boxes)
            Food status for all boxes.
        colors: float, (num_steps+1, num_boxes, height, width)
            Color cues of all boxes, regardless of the monkey gaze.
        counts: int, (num_steps+1, num_boxes, 2)
            Push counts since the beginning of the episode.
        p_boxes: float, (num_steps+1, num_boxes, 2, num_levels)
            Beliefs about each box.
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
        if p_boxes is not None:
            assert len(p_boxes)==len(pos)
        tmin = 0 if tmin is None else tmin
        tmax = len(pos) if tmax is None else min(tmax, len(pos))
        if figsize is None:
            figsize = (4.5, 4) if p_boxes is None else (7.5, 4)
        fig = plt.figure(figsize=figsize)

        if p_boxes is None:
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

        if p_boxes is None:
            artists_b = []
        else:
            n = self.num_boxes
            gap = 0.05
            h = (0.8-(n-1)*gap)/n
            axes = []
            for i in range(n):
                axes.append(fig.add_axes([0.55, 0.1+(n-i-1)*(h+gap), 0.35, h]))
            artists_b = self.plot_beliefs(
                axes, p_boxes[tmin], p_max=p_boxes[tmin:tmax].max(),
            )
            cbar = plt.colorbar(artists_b[-1], ax=axes, fraction=1/8)
            cbar.set_label('$P_\mathrm{box}$')
            cbar.ax.locator_params(nbins=5)

        def update(t, artists_a, artists_b):
            artists_a = self.plot_arena(
                ax, pos[t], gaze[t], rewarded[t],
                foods[t], colors[t], counts[t], artists_a,
            )
            if p_boxes is not None:
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


class SimilarBoxForagingEnv(ForagingEnv):
    r"""Environment where some box parameters are shared."""

    def __init__(self,
        box: dict|None = None,
        **kwargs,
    ):
        r"""
        Args
        ----
        box:
            A dictionary specifying shared parameters of boxes.

        """
        box = Config(box).fill({'_target_': 'hexarena.box.StationaryBox'})
        if kwargs.get('boxes') is None:
            kwargs['boxes'] = [box]
        else:
            for i in range(len(kwargs['boxes'])):
                kwargs['boxes'][i].update(box)
        super().__init__(**kwargs)
        self._shared_names = [n for n in self.boxes[0].param_names if n in box]
        for i in range(self.num_boxes):
            for name in self._shared_names:
                self.boxes[i].param_names.remove(name)

    def get_param(self) -> EnvParam:
        param = []
        for name in self._shared_names:
            val, *_ = self.boxes[0]._get_param(name)
            param += val
        param += super().get_param()
        return param

    def set_param(self, param: EnvParam) -> None:
        c = 0
        for name in self._shared_names:
            val, *_ = self.boxes[0]._get_param(name)
            n = len(val)
            val = param[c:c+n]
            for box in self.boxes:
                box._set_param(name, val)
            c += n
        super().set_param(param[c:])

    def param_bounds(self) -> tuple[EnvParam, EnvParam]:
        param_low, param_high = [], []
        for name in self._shared_names:
            _, low, high = self.boxes[0]._get_param(name)
            param_low += [*low]
            param_high += [*high]
        low, high = super().param_bounds()
        param_low += [*low]
        param_high += [*high]
        return param_low, param_high


class BlindWrapper(ForagingEnv):
    r"""Wrapper to disable color cue information."""

    def __init__(self,
        env: ForagingEnv,
    ):
        self.wrapped = env
        nvec = (*self.monkey.state_space.nvec, 2)
        self.observation_space = MultiDiscrete(nvec)

    def __getattr__(self, name: str):
        return getattr(self.wrapped, name)

    def _get_observation(self, rewarded: bool) -> Observation:
        r"""
        Only monkey state and whether it get rewarded is observed.

        """
        observation = (*self.monkey.get_state(), int(rewarded))
        return observation

    def extract_observation_action_reward(self, env_data: dict) -> tuple[Array, Array, Array]:
        r"""
        Color cue information is discarded.

        """
        observations, actions, rewards = self.wrapped.extract_observation_action_reward(env_data)
        observations = observations[:, list(range(len(self.monkey.state_space.nvec)))+[-1]]
        return observations, actions, rewards


class SingleBoxEnv:
    r"""An environment in which navigation is not needed."""

    def __init__(self,
        box: BaseFoodBox|dict|None = None,
        push_cost: float = 1.,
        time_cost: float = 0.,
        dt: float = 1.,
    ):
        self.push_cost = push_cost
        self.time_cost = time_cost
        self.dt = dt

        box = Config(box).fill({'_target_': 'hexarena.box.PoissonBox'})
        self.box: BaseFoodBox = box.instantiate()

        self.state_space = self.box.state_space
        self.observation_space = MultiDiscrete([self.box.num_patches, 2])
        self.action_space = Discrete(2)

        self.known_dim = 0

        self.rng = np.random.default_rng()


    def __repr__(self) -> str:
        b_str = str(self.box)
        b_str = b_str[0].lower()+b_str[1:]
        return f"Foraging at {b_str}"

    @property
    def spec(self) -> dict:
        return {
            '_target_': 'hexarena.env.SingleBoxEnv',
            'box': self.box.spec,
            'push_cost': self.push_cost,
            'time_cost': self.time_cost,
            'dt': self.dt,
        }

    def get_state(self) -> State:
        return self.box.get_state()

    def set_state(self, state: State) -> None:
        self.box.set_state(state)

    def _get_observation(self, rewarded: bool) -> Observation:
        colors = self.box.colors.reshape(-1)
        observation = tuple(colors)+(int(rewarded),)
        return observation

    def _get_info(self) -> dict:
        info = {
            'food': self.box.food, 'level': self.box.level, 'colors': self.box.colors,
        }
        return info

    def reset(self, seed: int|None = None) -> tuple[Observation, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.box.rng = self.rng
        self.box.reset()
        observation = self._get_observation(False)
        info = self._get_info()
        return observation, info

    def step(self, action: int) -> tuple[Observation, float, bool, bool, dict]:
        reward, terminated, truncated = -self.time_cost, False, False
        push = action==1
        if push:
            reward -= self.push_cost
        _reward = self.box.step(push)
        reward += _reward
        rewarded = _reward>0
        observation = self._get_observation(rewarded)
        info = self._get_info()
        return observation, reward, terminated, truncated, info
