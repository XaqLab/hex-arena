import os
from pathlib import Path
import h5py
import numpy as np
from irc.utils import ProgressBarCallback as _ProgressBarCallback

from . import DATA_DIR

def get_data_pth(subject: str) -> Path:
    r"""Returns mat file path.

    Args
    ----
    subject:
        Subject name, can be 'marco', 'dylan' or 'viktor'.

    Returns
    -------
    data_pth:
        Path to the mat file.

    """
    data_pth = DATA_DIR/f'data_{subject}.mat'
    assert os.path.exists(data_pth), f"{data_pth} not found"
    return data_pth


def get_valid_blocks(
    subject: str,
    min_duration: float = 300.,
    min_pos_ratio: float = 0.,
    min_gaze_ratio: float = 0.,
    min_push: int = 5,
    min_reward: int = 5,
) -> dict[tuple[str, int], dict[str, int|float]]:
    r"""Returns information of valid blocks saved in the mat file.

    For exponential schedule (Gamma shape 1), only blocks with time constants
    '{15, 21, 35}' is valid. For Gamma schedule of shape 10, only blocks with
    time constants '{7, 14, 21}' is valid.

    Args
    ----
    subject:
        Subject name.
    min_duration:
        Mininum duration of a block, in seconds.
    min_pos_ratio:
        Minimum valid data ratio for monkey position data.
    min_gaze_ratio:
        Minimum valid data ratio for monkey gaze data.
    min_push:
        Minimum number of pushes.
    min_reward:
        Minimum counts of collected rewards.

    Returns
    -------
    block_infos:
        A dict with block ID `(session_id, block_idx)` as keys, each value is a
        dict containing summary information of valid blocks, containing keys:
        - 'duration': float, block duration in seconds
        - 'pos_ratio': float, valid data ratio of position data
        - 'gaze_ratio': float, valid data ratio of gaze data
        - 'push': int, number of pushes
        - 'reward': int, number of rewards
        - 'kappa': float, cue reliability
        - 'gamma': float, shape parameter of Gamma distribution

    """
    with h5py.File(get_data_pth(subject), 'r') as f:
        num_sessions = len(f['session']['id'])
        block_infos = {}
        for s_idx in range(num_sessions):
            session_id = ''.join([chr(c) for c in f[f['session']['id'][s_idx, 0]][:, 0]])
            blocks = f[f['session']['block'][s_idx, 0]]
            num_blocks = len(blocks['continuous'])
            for b_idx in range(num_blocks):
                block = f[blocks['events'][b_idx][0]]
                tic = np.array(block['tStartBeh'])[0, 0]
                toc = np.array(block['tEndBeh'])[0, 0]
                duration = toc-tic
                if duration<min_duration:
                    continue
                meta = {'duration': np.round(duration*1000)/1000} # precision 1 ms

                block = f[blocks['continuous'][b_idx][0]]
                t = np.array(block['t']).squeeze()
                if np.unique(np.diff(t)).std()>1e-3:
                    continue
                mask = t<=duration
                pos_xyz = np.array(block['position']).squeeze()
                if pos_xyz.shape==(len(t), 3):
                    pos_xyz = pos_xyz[mask]
                    pos_ratio = 1-np.any(np.isnan(pos_xyz[:, :2]), axis=1).mean()
                else:
                    pos_ratio = 0.
                meta.update({'pos_ratio': pos_ratio})
                gaze_xyz = np.array(block['eyeArenaInt']).squeeze()
                if gaze_xyz.shape==(len(t), 3):
                    gaze_xyz = gaze_xyz[mask]
                    gaze_ratio = 1-np.any(np.isnan(gaze_xyz[:, :2]), axis=1).mean()
                else:
                    gaze_ratio = 0.
                meta.update({'gaze_ratio': gaze_ratio})
                if meta['pos_ratio']<min_pos_ratio or meta['gaze_ratio']<min_gaze_ratio:
                    continue

                block = f[blocks['events'][b_idx][0]]
                pushes = np.array(block['tPush']['all'], dtype=float).squeeze()
                flags = np.array(block['pushLogical']['all'], dtype=bool).squeeze()
                meta.update({
                    'push': len(pushes), 'reward': np.sum(flags),
                })
                if meta['push']<min_push or meta['reward']<min_reward:
                    continue

                block = f[blocks['params'][b_idx][0]]
                kappas = np.array(block['kappa']).squeeze()
                if np.any(np.isnan(kappas)) or len(np.unique(kappas))>1:
                    continue
                meta['kappa'] = np.unique(kappas).item()
                taus = np.array(block['schedules']).squeeze()
                if np.any(np.isnan(taus)):
                    continue
                gamma = np.array(block['gammaShape'])[0, 0]
                if gamma==1. and set(taus)!=set([15., 21., 35.]):
                    continue
                if gamma==10. and set(taus)!=set([7., 14., 21.]):
                    continue
                meta['gamma'] = gamma

                block_infos[(session_id, b_idx)] = meta
    return block_infos


def _infer_slope(t, cues):
    r"""Infers slope of linear probability cue.

    Args
    ----
    t: (num_steps)
        Time stamps of each sample, monotonically increasing.
    cues: (num_steps)
        Color cues that linearly increases from 0 after each box push.

    Returns
    -------
    t_draws: (num_steps)
        Drawn time intervals after each push. Each entry corresponds to the
        latest push.

    """
    idxs, = np.nonzero(np.diff(cues)<0)
    idxs = np.concatenate([[0], idxs+1, [len(t)]])
    t_draws = np.empty(t.shape, dtype=float)
    for i in range(len(idxs)-1):
        _t = t[idxs[i]:idxs[i+1]]
        _cues = cues[idxs[i]:idxs[i+1]]
        mask = _cues<1
        t_draw = np.polyfit(_cues[mask], _t[mask], 1)[0]
        t_draws[idxs[i]:idxs[i+1]] = t_draw
    return t_draws


def load_monkey_data(subject: str, session_id: str, block_idx: int) -> dict:
    r"""Loads one block data from mat file.

    Compatible with data files prepared in May 2024, e.g. 'data_Marco.mat',
    see `dataset_info.rtf` for more details. Only exponential food schedule is
    supported now.

    Args
    ----
    subject:
        Subject name.
    session_id:
        A 8-digit string of experiment session, typically in the format of
        'YYYYMMDD'.
    block_idx:
        Index of experiment block, starting from 0.

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
        - `gamma_shape`: The shape parameter of Gamma distribution, `1.` means
        exponential distribution.
        - `kappas`: (3,). Noise level of each box.
        - `taus`: (3,). Time constants of exponential distribution of reward
            intervals.
        - `intervals`: list of 1D array. The reward intervals at pushes of each
        boxes. In `gamma_shape==10` experiments, the initial reward interval is
        included, while in `gamma_shape==1` experiments, the first interval is
        not saved.

    """
    block_data = {}
    with h5py.File(get_data_pth(subject), 'r') as f:
        num_sessions = len(f['session']['id'])
        session_ids = []
        for s_idx in range(num_sessions):
            session_ids.append(''.join([chr(c) for c in f[f['session']['id'][s_idx, 0]][:, 0]]))
        session_idx = session_ids.index(session_id)
        blocks = f[f['session']['block'][session_idx, 0]]

        block = f[blocks['events'][block_idx][0]]
        tic = np.array(block['tStartBeh'])[0, 0]
        toc = np.array(block['tEndBeh'])[0, 0]
        duration = toc-tic

        block = f[blocks['continuous'][block_idx][0]]
        block_data['t'] = np.array(block['t']).squeeze()
        mask = block_data['t']<=duration
        block_data['pos_xyz'] = np.array(block['position']).squeeze()
        block_data['gaze_xyz'] = np.array(block['eyeArenaInt']).squeeze()
        for key in ['pos_xyz', 'gaze_xyz']:
            if block_data[key].shape==(len(block_data['t']), 3):
                block_data[key] = block_data[key][mask]
            else:
                block_data[key] = np.full((mask.sum(), 3), fill_value=np.nan)
        block_data['t'] = block_data['t'][mask]
        block_data['cues'] = np.stack([
            np.array(block['visualCueSignal'][f'box{i}']).squeeze() for i in [2, 3, 1]
        ], axis=1)[mask]

        block = f[blocks['events'][block_idx][0]]
        block_data['push_t'] = np.array(block['tPush']['all'], dtype=float).squeeze()
        block_data['push_idx'] = (np.array(block['tPush']['id'], dtype=int).squeeze()+1)%3
        block_data['push_flag'] = np.array(block['pushLogical']['all'], dtype=bool).squeeze()

        block = f[blocks['params'][block_idx][0]]
        block_data['gamma_shape'] = np.array(block['gammaShape']).item()
        kappa2, kappa0, kappa1 = np.array(block['kappa']).squeeze()
        block_data['kappas'] = np.array([kappa0, kappa1, kappa2])
        assert len(np.unique(block_data['kappas']))==1, "Noise level should be the same for all boxes."
        tau2, tau0, tau1 = np.array(block['schedules']).squeeze()
        block_data['taus'] = np.array([tau0, tau1, tau2])
        intervals2 = np.array(block['rewardWaitTime']['box1']).squeeze()
        intervals0 = np.array(block['rewardWaitTime']['box2']).squeeze()
        intervals1 = np.array(block['rewardWaitTime']['box3']).squeeze()
        block_data['intervals'] = [intervals0, intervals1, intervals2]
    return block_data


def align_monkey_data(block_data: dict) -> dict:
    r"""Rotates and flips data in space to have a fixed box order.

    Args
    ----
    block_data:
        Data directly read from mat file using `load_monkey_data`.

    Returns
    -------
    block_data:
        All information regarding spatial coordinates and box identity is
        properly transformed so that box 0 has the slowest rate (tau=35.) and
        box 2 has the fastest (tau=15.).

    """
    def rot_xy(xy, theta):
        t = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        xy = np.matmul(xy, t)
        return xy
    tau_2, tau_1, tau_0 = np.sort(block_data['taus'])
    if tuple(block_data['taus'])==(tau_0, tau_1, tau_2):
        # no change
        new_order = [0, 1, 2]
    if tuple(block_data['taus'])==(tau_0, tau_2, tau_1):
        # flip along 30 deg axis, box 0->0, 1->2, 2->1
        for key in ['pos_xyz', 'gaze_xyz']:
            xy = block_data[key][:, :2]
            xy = rot_xy(xy, np.pi/3)
            xy[:, 0] = -xy[:, 0]
            xy = rot_xy(xy, -np.pi/3)
            block_data[key][:, :2] = xy
        new_order = [0, 2, 1]
    if tuple(block_data['taus'])==(tau_1, tau_0, tau_2):
        # flip along y axis, box 0->1, 1->0, 2->2
        for key in ['pos_xyz', 'gaze_xyz']:
            xy = block_data[key][:, :2]
            xy[:, 0] = -xy[:, 0]
            block_data[key][:, :2] = xy
        new_order = [1, 0, 2]
    if tuple(block_data['taus'])==(tau_1, tau_2, tau_0):
        # rotate 120 deg, box 0->1, 1->2, 2->0
        for key in ['pos_xyz', 'gaze_xyz']:
            xy = block_data[key][:, :2]
            xy = rot_xy(xy, 2*np.pi/3)
            block_data[key][:, :2] = xy
        new_order = [2, 0, 1]
    if tuple(block_data['taus'])==(tau_2, tau_0, tau_1):
        # to rotate -120 deg, box 0->2, 1->0, 2->1
        for key in ['pos_xyz', 'gaze_xyz']:
            xy = block_data[key][:, :2]
            xy = rot_xy(xy, -2*np.pi/3)
            block_data[key][:, :2] = xy
        new_order = [1, 2, 0]
    if tuple(block_data['taus'])==(tau_2, tau_1, tau_0):
        # flip along -30 deg axis, box 0->2, 1->1, 2->0
        for key in ['pos_xyz', 'gaze_xyz']:
            xy = block_data[key][:, :2]
            xy = rot_xy(xy, -np.pi/3)
            xy[:, 0] = -xy[:, 0]
            xy = rot_xy(xy, np.pi/3)
            block_data[key][:, :2] = xy
        new_order = [2, 1, 0]
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
        gamma: float|None = None,
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
        if gamma is None:
            gamma = 0.5**(1/self.disp_freq)
        self.gamma = gamma
        self.reward = None # reward rate
        self.food = None # frequency of getting food
        self.logs: list[tuple[float, float]] = []

    def _on_step(self) -> bool:
        reward = np.mean(self.locals['rewards'])
        food = np.mean([info['observation'][-1] for info in self.locals['infos']])
        self.reward = reward if self.reward is None else self.gamma*self.reward+(1-self.gamma)*reward
        self.food = food if self.food is None else self.gamma*self.food+(1-self.gamma)*food
        if self.n_calls%self.disp_freq==0:
            self.pbar.set_description(
                "[Reward {:.2f}], [Food {:.2f}]".format(self.reward, self.food)
            )
            self.logs.append((self.reward, self.food))
        return super()._on_step()
