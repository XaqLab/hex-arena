import h5py
import numpy as np


def load_monkey_data(filename, block_idx: int) -> dict:
    r"""Loads one block data from mat file.

    Compatible with 'monkey-data_04052023/testSession2.mat', see `dataset_info.rtf`
    for more details.

    Args
    ----
    filename:
        Path to data file.
    block_idx:
        Index of experiment block, ranging in [0, 7).

    Returns
    -------
    block_data:
        A dictionary containing raw data, time unit is 'sec' and space unit is
        `mm`. It contains keys:
        - `t`: (num_steps,). Time axis of the block.
        - `pos_xyz`: (num_steps, 3). 3D coordinates of the monkey position.
        - `gaze_xyz`: (num_steps, 3). 3D coordinates of the monkey gaze.
        - `cues`: (num_steps, 3). Color cue values for three boxes.
        - `push_t`: (num_events,). Time of push events.
        - `push_id`: (num_events,). Box ID of each push, in [1, 3].
        - `push_flag`: (num_events,). Whether a reward is obtained of each push.
        - `kappas`: (3,). Noise level of each box.
        - `taus`: (3,). Time constants of exponential distribution of reward
            intervals.

    """
    block_data = {}
    with h5py.File(filename, 'r') as f:
        block = f[f['block']['continuous'][block_idx][0]]
        block_data['t'] = np.array(block['t']).squeeze()
        block_data['pos_xyz'] = np.array(block['position']).squeeze()
        block_data['gaze_xyz'] = np.array(block['eyeArenaInt']).squeeze()
        block_data['cues'] = np.stack([np.array(block['rewardProb'][f'box{i}']).squeeze() for i in range(1, 4)], axis=1)

        block = f[f['block']['events'][block_idx][0]]
        block_data['push_t'] = np.array(block['tPush']['all'], dtype=float).squeeze()
        block_data['push_id'] = np.array(block['tPush']['id'], dtype=int).squeeze()
        block_data['push_flag'] = np.array(block['pushLogical']['all'], dtype=bool).squeeze()

        block = f[f['block']['params'][block_idx][0]]
        block_data['kappas'] = np.array(block['kappa']).squeeze()
        block_data['taus'] = np.array(block['schedules']).squeeze()
    return block_data
