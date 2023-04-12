import h5py
import numpy as np

from .env import ForagingEnv

def compute_cue(t, push_t, rate):
    r"""Computes cumulative probabilities given push times and Poisson rate."""
    push_t = np.concatenate([[0], push_t, [np.inf]])
    cue = np.empty(t.shape)

    for i in range(1, len(push_t)):
        idxs = (t>=push_t[i-1])&(t<push_t[i])
        _t = t[idxs]
        cue[idxs] = 1-np.exp(-(_t-push_t[i-1])*rate)
    return cue

def load_monkey_data(filename, block_idx: int) -> dict:
    r"""Loads one block data from mat file.

    Compatible with 'monkey-data_04052023/testSession2.mat'.

    """
    block_data = {}
    with h5py.File(filename, 'r') as f:
        block = f[f['block']['continuous'][block_idx][0]]
        block_data['t'] = np.array(block['t']).squeeze()
        block_data['pos_xyz'] = np.array(block['position']).squeeze()
        block_data['gaze_xyz'] = np.array(block['eyeArenaInt']).squeeze()

        block = f[f['block']['events'][block_idx][0]]
        block_data['push_t'] = np.array(block['tPush']['all'], dtype=float).squeeze()
        block_data['push_id'] = np.array(block['tPush']['id'], dtype=int).squeeze()
        block_data['push_flag'] = np.array(block['pushLogical']['all'], dtype=bool).squeeze()

        block = f[f['block']['params'][block_idx][0]]
        block_data['kappas'] = np.array(block['kappa']).squeeze()
        block_data['rates'] = 1/np.array(block['schedules']).squeeze()

        cues = []
        for box_idx in range(3):
            push_t = block_data['push_t'][block_data['push_id']==((box_idx+1)%3)+1]
            rate = block_data['rates'][box_idx]
            cues.append(compute_cue(block_data['t'], push_t, rate))
        block_data['cues'] = np.stack(cues, axis=1)
    return block_data

def discretize_monkey_data(block_data, env: ForagingEnv, arena_radius=1860.5):
    r"""Discretizes on space and time according to environment."""
    t = block_data['t']
    num_steps = int(np.ceil(t.max()/env.dt))

    xy = block_data['pos_xyz'][:, :2]/arena_radius
    pos = []
    for i in range(num_steps):
        _xy = xy[(t>=i*env.dt)&(t<(i+1)*env.dt), :]
        if np.all(np.isnan(_xy[:, 0])) or np.all(np.isnan(_xy[:, 1])):
            pos.append(-1)
        else:
            _xy = np.nanmean(_xy, axis=0)
            pos.append(env.arena.nearest_tile(_xy))
    pos = np.array(pos, dtype=int)

    xy = block_data['gaze_xyz'][:, :2]/arena_radius
    gaze = []
    for i in range(num_steps):
        _xy = xy[(t>=i*env.dt)&(t<(i+1)*env.dt), :]
        if np.all(np.isnan(_xy[:, 0])) or np.all(np.isnan(_xy[:, 1])):
            gaze.append(-1)
        else:
            _xy = np.nanmean(_xy, axis=0)
            gaze.append(env.arena.nearest_tile(_xy))
    gaze = np.array(gaze, dtype=int)

    push, success = [], []
    for i in range(1, num_steps):
        push_idxs, = ((block_data['push_t']>=i*env.dt)&(block_data['push_t']<(i+1)*env.dt)).nonzero()
        _push = np.unique(block_data['push_id'][push_idxs])
        if len(_push)>1:
            print(f'multiple boxes pushed at step {i}')
        push.append(len(_push)>0)
        success.append(np.any(block_data['push_flag'][push_idxs]))
    push = np.array(push, dtype=bool)
    success = np.array(success, dtype=bool)

    colors = []
    _color_size = int(env.boxes[0].num_patches**0.5)
    for i in range(num_steps):
        _cues = block_data['cues'][(t>=i*env.dt)&(t<(i+1)*env.dt), :].mean(axis=0)
        _cues = np.floor(_cues*np.array([box.num_grades for box in env.boxes]))
        colors.append(np.tile(_cues[:, None, None], (1, _color_size, _color_size)))
    colors = np.array(colors, dtype=int)

    env_data = {
        'num_steps': num_steps-1,
        'pos': pos, 'gaze': gaze,
        'push': push, 'success': success,
        'colors': colors,
    }
    return env_data

def extract_observation_action(env_data, env):
    r"""Extract observation and action sequences."""
    num_steps = env_data['num_steps']

    observations = np.empty((num_steps+1, len(env.observation_space.nvec)), dtype=int)
    for i in range(num_steps+1):
        for j in range(2):
            if j==0:
                vals = env_data['pos']
            if j==1:
                vals = env_data['gaze']
            if vals[i]>=0:
                observations[i, j] = vals[i]
            else:
                if i>0:
                    observations[i, j] = observations[i-1, j]
                else:
                    observations[i, j] = vals[(vals>=0).nonzero()[0].min()]
        j = 2
        for box_idx in range(3):
            if env_data['gaze'][i]==env.arena.boxes[box_idx]:
                colors = env_data['colors'][i, box_idx].reshape(-1)
            else:
                colors = env.boxes[box_idx].num_grades
            observations[i, j:(j+env.boxes[box_idx].num_patches)] = colors
            j += env.boxes[box_idx].num_patches

    actions = np.empty((num_steps,), dtype=int)
    for i in range(num_steps):
        if env_data['push'][i]:
            actions[i] = env._push
            continue
        if observations[i+1, 0]==observations[i, 0]:
            move = 6
        else:
            x1, y1 = env.arena.anchors[observations[i+1, 0]]
            x0, y0 = env.arena.anchors[observations[i, 0]]
            theta = np.arctan2(y1-y0, x1-x0)
            move = int(np.round(theta/(np.pi/3)))%6
        look = observations[i+1, 1]
        actions[i] = move+7*look

    return observations, actions
