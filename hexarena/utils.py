import h5py
import numpy as np

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
