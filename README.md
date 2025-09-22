# hex-arena
Monkey foraging experiment in a hexagonal arena

## Requirements
[`jarvis>=0.7`](https://github.com/lizhe07/jarvis/releases/tag/v0.7.0)

## Foraging experiment as a partially observable Markov decision process (POMDP)
The monkey is placed in a hexagonal arena where three food boxes are mounted on the walls. Food will
become available following a random process per each box independently. Monkey will get the food if
he pushes the box button after the food becomes available, otherwise the box gets resets.
Spatial-temporal color cues are displayed outside each box to indicate either the quality of the box
or the availability of the food, depending on the experiment condition.

The `gymnasium` environment implemented will use discretized spatial positions and time steps.
```python
from hexarena.env import ArenaForagingEnv

env = ArenaForagingEnv(
    boxes=[{'tau': tau} for tau in [35, 21, 15]],
)
```
An overhead view of a monkey following a random policy is shown below, along with the color cues and
push results of each box.
<center><img src='figures/random.policy.gif' width=350></center>

The agent state includes:
- 'pos': agent's current location,
- 'gaze': where the agent is looking at.

Both 'pos' and 'gaze' are integers in `{0, 1, ..., N}` for the tile index in the arena, in which `N`
is the number of tiles decided by the spatial resolution.

The environment state is composed of three box states, with each including:
- 'food': binary, whethere there is food available
- 'cue': float, a number in [0, 1) that specifies the color cue value

The spatial-temporal color pattern is converted from the 'cue' value onto a circular colormap, where
'cue=0' maps to blue and 'cue=1' maps to red. When the monkey looks at one of the box, it integrates
the pixel values in a random attention window by circular averaging, and gets a noisy observation in
the form of `(c_x, c_y)` inside the unit circle. `(0, 0)` is used when the monkey is not looking at
any box.

The full observation of the environment is thus the concatenation of agent state and direct
observation:
- 'rewarded': binary, where a food is obtained
- 'color': `(c_x, c_y)` for the visual feedback

Action space of the agent is an integer in `{0, 1, ..., N^2+3*N}`, which can be converted to a tuple
`(push, move, look)` by `env.monkey.convert_action`. When `push==False`, both `move` and `look` are
in `{0, 1, ..., N}`. When `push==True`, `move` can only take three values, namely indices of the
tiles where three boxes are mounted.

Please refer to the [notebook](env-example.ipynb) for more details.
