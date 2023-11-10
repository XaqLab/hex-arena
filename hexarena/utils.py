import h5py
import numpy as np


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
        block_data['cues'] = np.stack([np.array(block['rewardProb'][f'box{i}']).squeeze() for i in range(1, 4)])

        block = f[f['block']['events'][block_idx][0]]
        block_data['push_t'] = np.array(block['tPush']['all'], dtype=float).squeeze()
        block_data['push_id'] = np.array(block['tPush']['id'], dtype=int).squeeze()
        block_data['push_flag'] = np.array(block['pushLogical']['all'], dtype=bool).squeeze()

        block = f[f['block']['params'][block_idx][0]]
        block_data['kappas'] = np.array(block['kappa']).squeeze()
        block_data['taus'] = np.array(block['schedules']).squeeze()
    return block_data
