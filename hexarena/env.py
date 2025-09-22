import warnings
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Discrete, Box, Dict
from jarvis.config import Config
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from collections.abc import Sequence

from .arena import Arena
from .monkey import BaseMonkey, ArenaMonkey
from .box import BaseFoodBox, PoissonBox
from .color import get_cmap
from .alias import EnvParam, Figure, Axes, Artist, Array
from .utils import get_food_avails


class BaseForagingEnv(Env):
    r"""Base class for a foraging environment with multiple boxes.

    The environment contains multiple boxes each with its own dynamics, the
    monkey needs to push the button to open a box for food and possibly afer
    observing the color cue on boxes.

    Args
    ----
    monkey:
        Monkey instance or a dictionary to specify one.
    boxes:
        The food boxes or a list of dictionaries to specify them.
    dt:
        Time step size, in seconds.
    shared_param_names:
        Shared parameter names of boxes, this reduces the number of parameters
        of the environment.

    """

    def __init__(self,
        monkey: BaseMonkey|dict|None = None,
        boxes: list[BaseFoodBox|dict|None]|None = None,
        dt: float = 1.,
        shared_param_names: list[str]|None = None,
    ):
        self.dt = dt
        if shared_param_names is None:
            shared_param_names = ['kappa']
        self.shared_param_names = shared_param_names

        if isinstance(monkey, BaseMonkey):
            self.monkey = monkey
        else:
            monkey = Config(monkey).fill({'_target_': 'hexarena.monkey.BaseMonkey'})
            self.monkey: BaseMonkey = monkey.instantiate()

        self.n_boxes = 3 if boxes is None else len(boxes)
        if boxes is None:
            boxes = [None]*self.n_boxes
        self.boxes: list[BaseFoodBox] = []
        for i in range(self.n_boxes):
            if isinstance(boxes[i], BaseFoodBox):
                box = boxes[i]
            else:
                box = Config(boxes[i]).fill({'_target_': 'hexarena.box.PoissonBox'})
                box: BaseFoodBox = box.instantiate()
            box.dt = self.dt
            self.boxes.append(box)

        self.agt_space = self.monkey.state_space
        self.env_space = Dict({
            f'box_{i}': self.boxes[i].state_space for i in range(self.n_boxes)
        })
        self.obs_space: Dict = Dict({'rewarded': Discrete(2)})
        self.rng = np.random.default_rng()

    @property
    def observation_space(self) -> Dict:
        return Dict({
            **self.agt_space, **self.obs_space,
        })

    def get_agt_state(self) -> tuple[int, int]:
        return self.monkey.get_state()

    def set_agt_state(self, state: tuple[int, int]) -> None:
        self.monkey.set_state(state)

    def get_env_state(self) -> dict:
        return {
            f'box_{i}': self.boxes[i].get_state() for i in range(self.n_boxes)
        }

    def set_env_state(self, state: dict) -> None:
        for i in range(self.n_boxes):
            self.boxes[i].set_state(state[f'box_{i}'])

    def get_obs(self, rewarded: bool) -> dict:
        r"""Returns observation of environment.

        Args
        ----
        rewarded:
            Whether the monkey is rewarded at the current time step.

        Returns
        -------
        obs:
            A dictionary with keys:
            - 'rewarded': binary variable of whether rewarded or not.

        """
        obs = {'rewarded': int(rewarded)}
        return obs

    def _get_observation_and_info(self, rewarded: bool) -> tuple[dict, dict]:
        r"""Returns observation and info for Gymnasium API.

        The method should be called at the end of `reset` or `step`.

        Args
        ----
        rewarded:
            Whether the monkey is rewarded, see `get_obs` for more details.

        Returns
        -------
        observation:
            The observation combining both monkey state and its observation on
            the environment. See `BaseMonkey.get_state` and `get_obs` for more
            details.
        info:
            A dictionary containing POMDP related information, with keys:
            - 'agt_state': the monkey state.
            - 'env_state': the states of food boxes, see `get_env_state` for
            more details.
            - 'obs': observation of environment, see `get_obs` for more details.
            - 'colors': array of shape `(n_boxes, h, w)` for color patterns on
            all boxes.

        """
        agt_state = self.get_agt_state()
        env_state = self.get_env_state()
        obs = self.get_obs(rewarded)
        observation = {**agt_state, **obs}
        info = {
            'agt_state': agt_state, 'env_state': env_state, 'obs': obs,
        }
        if np.all([box.cue_in_state for box in self.boxes]):
            info['colors'] = np.stack([box.colors for box in self.boxes])
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
        r"""Resets environment."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.monkey.rng = self.rng
        self.monkey.reset()
        if np.all([box.tau_in_state for box in self.boxes]):
            taus = self.rng.permutation([box.tau for box in self.boxes])
            for i, box in enumerate(self.boxes):
                box.tau = taus[i]
        for box in self.boxes:
            box.rng = self.rng
            box.reset()
        observation, info = self._get_observation_and_info(False) # no reward at reset
        return observation, info

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        raise NotImplementedError

    def convert_experiment_data(self, block_data: dict) -> dict:
        r"""Converts raw experiment data to sequences at discrete time steps.

        Args
        ----
        block_data:
            One block of raw data loaded by `utils.load_monkey_data`.

        Returns
        -------
        env_data:
            A dictionary containing sequential data, with keys:
            - `cues`: float, `(n_steps+1, n_boxes)`. Cue values in [0, 1] of
            each box.
            - `colors`: float, `(n_steps+1, n_boxes, height, width)`. 2D color
            pattern on the food boxes.
            - `push`: int, `(n_steps,)`. Index of the box pushed at each time
            step. The values are in `[-1, n_boxes)` with ``-1`` for no push.
            - `rewarded`: bool, `(n_steps+1,)`. Whether the monkey is rewarded
            or not. The first element is always ``False`` corresponding to the
            environment reset.
            - `foods`: bool, `(n_steps+1, n_boxes)`. Food availability at each
            time step for all boxes.

        """
        t = block_data['t']
        n_steps = int(np.floor(t.max()/self.dt))

        # actual color movie is not saved, using independent cue array generated
        # at each time step instead
        self.reset(seed=0) # remove stochasticity of box 'get_colors' method
        cues, colors = [], []
        for i in range(n_steps+1):
            cues.append(np.clip(block_data['cues'][t>=i*self.dt, :][0], a_min=0, a_max=0.999))
            colors.append([])
            for box, cue in zip(self.boxes, cues[-1]):
                colors[-1].append(box.get_colors(cue))
        cues = np.stack(cues) # (n_steps+1, n_boxes)
        colors = np.array(colors, dtype=float) # (n_steps+1, n_boxes, height, width)

        # if mulitple pushes happen during one time step, verify they belong to the same box
        push_t, push_idx = block_data['push_t'], block_data['push_idx']
        push, rewarded = [], [False]
        for i in range(n_steps):
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
                rewarded.append(np.any(block_data['push_flag'][t_idxs]))
        push = np.array(push).astype(int)
        rewarded = np.array(rewarded).astype(bool)

        # food availability is inferred from the actual intervals
        foods = []
        for b_idx in range(self.n_boxes):
            if block_data['gamma_shape']==1:
                tau = block_data['taus'][b_idx].item()
                first_rewarded = block_data['push_flag'][block_data['push_idx']==b_idx][0].item()
            else:
                tau, first_rewarded = None, None
            foods.append(get_food_avails(
                block_data['push_t'][block_data['push_idx']==b_idx],
                block_data['intervals'][b_idx], n_steps, dt=self.dt,
                tau=tau, first_rewarded=first_rewarded,
            ))
        foods = np.stack(foods, axis=1)

        env_data = {
            'cues': cues, 'colors': colors,
            'push': push, 'rewarded': rewarded, 'foods': foods,
        }
        return env_data

    def _extract_env_states(self, env_data: dict) -> list:
        r"""Extracts box states from the converted data.

        Args
        ----
        env_data:
            See `extract_episode` for more details.

        Returns
        -------
        env_states:
            A list of box states.

        """
        n_steps = len(env_data['push'])
        env_states = []
        if np.all([isinstance(box, PoissonBox) for box in self.boxes]):
            for t in range(n_steps+1):
                env_state = {}
                for i, box in enumerate(self.boxes):
                    box_state = {'food': int(env_data['foods'][t, i])}
                    if box.cue_in_state:
                        box_state['cue'] = np.array([env_data['cues'][t, i]])
                    env_state[f'box_{i}'] = box_state
                    if box.tau_in_state:
                        box_state['tau'] = np.array([box.tau])
                env_states.append(env_state)
        else:
            raise NotImplementedError
        return env_states

    def _extract_obss(self, env_data: dict) -> list:
        r"""Extracts explicit observations from the coverted data.

        Args
        ----
        env_data:
            See `extract_episode` for more details.

        Returns
        -------
        obss:
            A list of observations.

        """
        n_steps = len(env_data['push'])
        obss = [{'rewarded': int(env_data['rewarded'][t])} for t in range(n_steps+1)]
        return obss

    def extract_episode(self, env_data: dict) -> tuple[list, list, list, Array]:
        r"""Extracts an episode from the converted data.

        Missing data will be filled in by certain strategy.

        Args
        ----
        env_data:
            A dictionary containing sequential data in one block, discretized
            in time with time bin size `dt`, see `convert_experiment_data` for
            more details.

        Returns
        -------
        agt_states, env_states, obss: `(n_steps+1,)`
            Each is a list of corresponding variable, as in `info` returned by
            `reset` and `step`.
        actions: int, `(n_steps,)`
            Integers for action taken at each time step.

        """
        raise NotImplementedError


class BanditForagingEnv(BaseForagingEnv):
    r"""Foraging environment similar to multi-armed bandit problem.

    Spatial and visual aspect of the food boxes are ignored, therefore the
    monkey only chooses to push one of the boxes or not push at each time step.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_space = Discrete(self.n_boxes+1)

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        # action==n_boxes is 'no push'
        reward = self.monkey.step(action<self.n_boxes)
        rewarded = False
        for i in range(self.n_boxes):
            _reward = self.boxes[i].step(action==i)
            reward += _reward
            rewarded |= _reward>0
        observation, info = self._get_observation_and_info(rewarded)
        return observation, reward, False, False, info

    def extract_episode(self, env_data):
        n_steps = len(env_data['push'])
        agt_states = [{} for _ in range(n_steps+1)]
        env_states = self._extract_env_states(env_data)
        obss = self._extract_obss(env_data)
        actions = np.mod(env_data['push'], self.n_boxes+1).astype(int)
        return agt_states, env_states, obss, actions


class ArenaForagingEnv(BaseForagingEnv):
    r"""Foraging environment in a hexagonal arena.

    Three food boxes are installed on the walls of a hexagonal arena, and the
    monkey needs to move inside the arena to get close to one of the boxes. When
    the monkey looks at one of the box, it gets the color cue of it.

    """

    monkey: ArenaMonkey

    def __init__(self,
        arena: Arena|dict|None = None,
        monkey: ArenaMonkey|dict|None = None,
        **kwargs,
    ):
        if isinstance(arena, Arena):
            self.arena = arena
        else:
            arena = Config(arena).fill({'_target_': 'hexarena.arena.Arena'})
            self.arena: Arena = arena.instantiate()

        if isinstance(monkey, dict) or monkey is None:
            monkey = Config(monkey).fill({'_target_': 'hexarena.monkey.ArenaMonkey'})
        monkey.arena = self.arena
        super().__init__(monkey=monkey, **kwargs)
        assert self.arena.n_boxes==self.n_boxes
        for i in range(self.n_boxes):
            self.boxes[i].pos = self.arena.boxes[i]

        self.obs_space = Dict({
            'color': Box(-np.inf, np.inf, shape=(2,)),
            'rewarded': Discrete(2),
        })
        self.action_space = self.monkey.action_space

    def get_obs(self, rewarded: bool) -> dict:
        r"""Returns observation of environment.

        Besides whether it gets rewarded, the monkey gets color observation from
        the box it is looking at. When the monkey is not looking at any box,
        color observation `(0, 0)` will be returned instead.

        Returns
        -------
        obs:
            A dictionary with keys:
            - 'rewarded': binary variable of whether rewarded or not.
            - 'color': circular coordinates of seen color, see `Monkey.look` for
            more details.

        """
        obs = super().get_obs(rewarded)
        box = next((box for box in self.boxes if self.monkey.gaze==box.pos), None)
        obs.update({
            'color': np.array(
                (0., 0.) if box is None else self.monkey.look(box.colors)
            ),
        })
        return obs

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
        r"""Converts raw experiment data to sequences at discrete time steps.

        Spatial discretization is done to `pos` and `gaze`.

        Args
        ----
        block_data:
            One block of raw data, see `BaseForagingEnv.convert_experiment_data`
            for more details.
        arena_radius:
            Radius of the arena, in mm.

        Returns
        -------
        env_data:
            A dictionary containing sequential data, with keys additional to
            what `BaseForagingEnv` returned:
            - `pos`: int, `(n_steps+1,)`. Monkey position in `[-1, n_tiles)`,
            with '-1' for missing data.
            - `gaze`: int, `(n_steps+1,)`. Gaze position in `[-1, n_tiles)`,
            with '-1' for missing data.

        """
        env_data = super().convert_experiment_data(block_data)
        n_steps = len(env_data['push'])
        t = block_data['t']

        def get_trajectory(xyz: Array) -> Array:
            r"""Extracts 2D trajectories as in tile indices.

            Args
            ----
            xyz: float, `(n_stamps, 3)`
                Raw 3D coordinates in mm with time stamps marked in `t`.

            Returns
            -------
            vals: int, `(n_steps,)`
                Tile indices at each time step of size `self.dt`. '-1' for
                missing data.

            """
            xy = xyz[:, :2]/arena_radius
            vals = []
            for i in range(n_steps+1):
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

        # when a box is pushed, monkey location is forced to be at the tile
        for i in range(n_steps):
            if env_data['push'][i]>=0:
                pos[i+1] = self.arena.boxes[env_data['push'][i]]

        env_data.update({'pos': pos, 'gaze': gaze})
        return env_data

    def convert_simulated_data(self,
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
        pos, gaze, rewarded, colors, foods = [], [], [], [], []
        for info in infos:
            pos.append(info['agt_state']['pos'])
            gaze.append(info['agt_state']['gaze'])
            rewarded.append(info['obs']['rewarded'])
            colors.append(info['colors'])
            self.set_env_state(info['env_state'])
            foods.append([
                self.boxes[i].food for i in range(self.n_boxes)
            ])
        pos = np.array(pos).astype(int)
        gaze = np.array(gaze).astype(int)
        rewarded = np.array(rewarded).astype(bool)
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
        counts = np.zeros((num_steps+1, self.n_boxes, 2), dtype=int)
        for t in range(num_steps):
            counts[t+1] = counts[t]
            if rewarded[t+1]: # successful count
                counts[t+1, push[t], 0] += 1
            if push[t]>=0: # total count
                counts[t+1, push[t], 1] += 1
        return counts

    def _extract_agt_states(self, env_data: dict) -> list[dict]:
        r"""Extracts monkey states."""
        n_steps = len(env_data['push'])
        _agt_states = {
            'pos': np.empty((n_steps+1,), dtype=int),
            'gaze': np.empty((n_steps+1,), dtype=int),
        }
        for key in _agt_states:
            vals = env_data[key]
            for t in range(n_steps+1):
                if vals[t]>=0:
                    _agt_states[key][t] = vals[t]
                else: # deal with data gap
                    if t>0:
                        _agt_states[key][t] = _agt_states[key][t-1] # previous step
                    else:
                        _agt_states[key][t] = vals[(vals>=0).nonzero()[0].min()] # first valid step
        agt_states = [{key: _agt_states[key][t].item() for key in _agt_states} for t in range(n_steps+1)]
        return agt_states

    def _extract_obss(self, env_data: dict):
        obss = super()._extract_obss(env_data)
        n_steps = len(env_data['push'])
        self.reset(seed=0) # remove stochasticity of monkey 'look' method
        for t in range(n_steps+1):
            b_idx = next((i for i, box in enumerate(self.boxes) if env_data['gaze'][t]==box.pos), None)
            obss[t].update({
                'color': np.array(
                    (0., 0.) if b_idx is None else self.monkey.look(env_data['colors'][t, b_idx])
                ),
            })
        return obss

    def extract_episode(self, env_data: dict) -> tuple[list, list, list, Array]:
        agt_states = self._extract_agt_states(env_data)
        env_states = self._extract_env_states(env_data)
        obss = self._extract_obss(env_data)

        n_steps = len(env_data['push'])
        actions = []
        for t in range(n_steps):
            actions.append(self.monkey.index_action(
                env_data['push'][t]>=0,
                agt_states[t+1]['pos'], agt_states[t+1]['gaze'],
            ))
        actions = np.array(actions).astype(int)
        return agt_states, env_states, obss, actions

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
            foods_color = ['none']*self.n_boxes
        else:
            foods_color = ['red' if food else 'blue' for food in foods]
        if colors is None:
            colors = np.full((self.n_boxes, 1, 1), np.nan)
        if counts is None:
            texts = ['']*self.n_boxes
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
                *_, h, w = colors.shape
                s = 0.2 # dimension for color cue
                dx, dy = min(w/h, 1)*s, min(1, h/w)*s
                h_food, = ax.plot(
                    [x-dx, x+dx], [y+1.2*dy, y+1.2*dy],
                    color=foods_color[i], linewidth=2, zorder=2,
                )
                h_foods.append(h_food)
                h_box = ax.imshow(
                    colors[i], extent=[x-dx, x+dx, y-dy, y+dy], zorder=2,
                    vmin=0, vmax=1, cmap=get_cmap(), interpolation='nearest',
                )
                h_boxes.append(h_box)
                h_count = ax.text(
                    x, y+np.sign(y)*1.8*s, texts[i], ha='center', va='center',
                )
                h_counts.append(h_count)
        else:
            h_pos, h_gaze = artists[:2]
            h_foods = artists[2:(2+self.n_boxes)]
            h_boxes = artists[(2+self.n_boxes):(2+2*self.n_boxes)]
            h_counts = artists[(2+2*self.n_boxes):(2+3*self.n_boxes)]
            self.arena.plot_tile(ax, pos, tile_color, h_tile=h_pos)
            h_gaze.set_offsets(self.arena.anchors[gaze] if gaze>=0 else (np.nan, np.nan))
            for i in range(self.n_boxes):
                h_foods[i].set_color(foods_color[i])
                h_boxes[i].set_data(colors[i])
                h_counts[i].set_text(texts[i])
        artists = [h_pos, h_gaze]+h_foods+h_boxes+h_counts
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
            assert colors.shape[:2]==(n_steps+1, self.n_boxes)
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
            assert foods.shape==(n_steps+1, self.n_boxes)
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
            n = self.n_boxes
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
