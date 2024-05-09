import h5py
import numpy as np
from irc.utils import ProgressBarCallback as _ProgressBarCallback


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
        - `cues`: (num_steps, 3). Color cue values for three boxes, ordered as
        'northeast' (0), 'northwest' (1), 'south' (2).
        - `push_t`: (num_events,). Time of push events.
        - `push_idx`: (num_events,). Box index of each push, in [0, 3).
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
        block_data['cues'] = np.stack([
            np.array(block['rewardProb'][f'box{i}']).squeeze() for i in [2, 3, 1]
        ], axis=1)

        block = f[f['block']['events'][block_idx][0]]
        block_data['push_t'] = np.array(block['tPush']['all'], dtype=float).squeeze()
        block_data['push_idx'] = (np.array(block['tPush']['id'], dtype=int).squeeze()+1)%3
        block_data['push_flag'] = np.array(block['pushLogical']['all'], dtype=bool).squeeze()

        block = f[f['block']['params'][block_idx][0]]
        kappa2, kappa0, kappa1 = np.array(block['kappa']).squeeze()
        block_data['kappas'] = np.array([kappa0, kappa1, kappa2])
        assert len(np.unique(block_data['kappas']))==1, "Noise level should be the same for all boxes."
        tau2, tau0, tau1 = np.array(block['schedules']).squeeze()
        block_data['taus'] = np.array([tau0, tau1, tau2])
    return block_data


def align_monkey_data(block_data: dict, block_idx: int) -> dict:
    r"""Rotates and flips data in space to have a fixed box order.

    Args
    ----
    block_data:
        Data directly read from mat file using `load_monkey_data`.
    block_idx:
        Index of experiment block, ranging in [0, 7).

    Returns
    -------
    block_data:
        All information regarding spatial coordinates and box identity is
        properly transformed so that box 0 has the slowest rate and box 2 has
        the fastest.

    """
    def rot_xy(xy, theta):
        t = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        xy = np.matmul(xy, t)
        return xy
    assert block_idx in [0, 2, 4, 6] # only four sessions with low noise are considered for now
    if block_idx==0:
        # to rotate -120 deg, box 0->2, 1->0, 2->1
        for key in ['pos_xyz', 'gaze_xyz']:
            xy = block_data[key][:, :2]
            xy = rot_xy(xy, -2*np.pi/3)
            block_data[key][:, :2] = xy
        new_order = [1, 2, 0]
    if block_idx==2:
        # flip along 30 deg axis, box 0->0, 1->2, 2->1
        for key in ['pos_xyz', 'gaze_xyz']:
            xy = block_data[key][:, :2]
            xy = rot_xy(xy, np.pi/3)
            xy[:, 0] = -xy[:, 0]
            xy = rot_xy(xy, -np.pi/3)
            block_data[key][:, :2] = xy
        new_order = [0, 2, 1]
    if block_idx==4:
        # to rate 120 deg, box 0->1, 1->2, 2->0
        for key in ['pos_xyz', 'gaze_xyz']:
            xy = block_data[key][:, :2]
            xy = rot_xy(xy, 2*np.pi/3)
            block_data[key][:, :2] = xy
        new_order = [2, 0, 1]
    if block_idx==6:
        # no change
        new_order = [0, 1, 2]
    push_idx = block_data['push_idx'].copy()
    for i in range(3):
        push_idx[block_data['push_idx']==new_order[i]] = i
    block_data['push_idx'] = push_idx
    block_data['cues'] = block_data['cues'][:, new_order]
    block_data['taus'] = block_data['taus'][new_order]
    return block_data


class ProgressBarCallback(_ProgressBarCallback):
    r"""Callback for update progress bar with recent running statistics."""

    def __init__(self,
        pbar, disp_freq: int = 128,
        gamma: float = 0.99,
    ):
        r"""
        Args
        ----
        pbar, disp_freq:
            See `irc.utils.ProgressBarCallback` for more details.
        gamma:
            Decay factor for computing running average.

        """
        super().__init__(pbar, disp_freq)
        self.gamma = gamma
        self.reward = 0. # reward rate
        self.food = 0. # frequency of getting food

    def _on_step(self) -> bool:
        for reward, info in zip(self.locals['rewards'], self.locals['infos']):
            self.reward = self.gamma*self.reward+(1-self.gamma)*reward
            self.freq = self.gamma*self.freq+(1-self.gamma)*info['observation'][-1]
        if self.n_calls%self.disp_freq==0:
            self.pbar.set_description(
                "[Reward rate {:.2f}], [Food freq {:.2f}]".format(self.reward, self.freq)
            )
        return super()._on_step()
