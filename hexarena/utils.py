import h5py
import numpy as np
from irc.utils import ProgressBarCallback as _ProgressBarCallback


def get_valid_blocks(
    filename,
    min_duration: float = 300.,
    min_pos_ratio: float = 0.,
    min_gaze_ratio: float = 0.,
    requires_pos: bool = False,
    requires_gaze: bool = False,
    min_push: int = 5,
    min_reward: int = 5,
) -> dict:
    r"""Returns valid blocks in mat file."""
    with h5py.File(filename, 'r') as f:
        num_sessions = len(f['session']['id'])
        meta = {}
        for s_idx in range(num_sessions):
            session_id = ''.join([chr(c) for c in f[f['session']['id'][s_idx, 0]][:, 0]])
            meta[session_id] = {}
            blocks = f[f['session']['block'][s_idx, 0]]
            num_blocks = len(blocks['continuous'])
            for b_idx in range(num_blocks):
                block = f[blocks['events'][b_idx][0]]
                tic = np.array(block['tStartBeh'])[0, 0]
                toc = np.array(block['tEndBeh'])[0, 0]
                duration = toc-tic
                if duration<min_duration:
                    continue
                _meta = {'duration': np.round(duration*1000)/1000} # precision 1 ms

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
                _meta.update({'pos_ratio': pos_ratio})
                gaze_xyz = np.array(block['eyeArenaInt']).squeeze()
                if gaze_xyz.shape==(len(t), 3):
                    gaze_xyz = gaze_xyz[mask]
                    gaze_ratio = 1-np.any(np.isnan(gaze_xyz[:, :2]), axis=1).mean()
                else:
                    gaze_ratio = 0.
                _meta.update({'gaze_ratio': gaze_ratio})
                if _meta['pos_ratio']<min_pos_ratio or _meta['gaze_ratio']<min_gaze_ratio:
                    continue

                block = f[blocks['events'][b_idx][0]]
                pushes = np.array(block['tPush']['all'], dtype=float).squeeze()
                flags = np.array(block['pushLogical']['all'], dtype=bool).squeeze()
                _meta.update({
                    'push': len(pushes), 'reward': np.sum(flags),
                })
                if _meta['push']<min_push or _meta['reward']<min_reward:
                    continue

                block = f[blocks['params'][b_idx][0]]
                kappas = np.array(block['kappa']).squeeze()
                if np.any(np.isnan(kappas)) or len(np.unique(kappas))>1:
                    continue
                taus = np.array(block['schedules']).squeeze()
                if np.any(np.isnan(taus)):
                    continue

                meta[session_id][b_idx] = _meta
            if len(meta[session_id])==0:
                meta.pop(session_id)
    return meta


def load_monkey_data(filename, session_id: str, block_idx: int) -> dict:
    r"""Loads one block data from mat file.

    Compatible with data files prepared in May 2024, e.g. 'data_Marco.mat',
    see `dataset_info.rtf` for more details. Only exponential food schedule is
    supported now.

    Args
    ----
    filename:
        Path to data file.
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
        - `kappas`: (3,). Noise level of each box.
        - `taus`: (3,). Time constants of exponential distribution of reward
            intervals.

    """
    block_data = {}
    with h5py.File(filename, 'r') as f:
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
                block_data[key] = np.full((len(block_data['t']), 3), fill_value=np.nan)
        block_data['t'] = block_data['t'][mask]
        block_data['cues'] = np.stack([
            np.array(block['visualCueSignal'][f'box{i}']).squeeze() for i in [2, 3, 1]
        ], axis=1)[mask]

        block = f[blocks['events'][block_idx][0]]
        block_data['push_t'] = np.array(block['tPush']['all'], dtype=float).squeeze()
        block_data['push_idx'] = (np.array(block['tPush']['id'], dtype=int).squeeze()+1)%3
        block_data['push_flag'] = np.array(block['pushLogical']['all'], dtype=bool).squeeze()

        block = f[blocks['params'][block_idx][0]]
        kappa2, kappa0, kappa1 = np.array(block['kappa']).squeeze()
        block_data['kappas'] = np.array([kappa0, kappa1, kappa2])
        assert len(np.unique(block_data['kappas']))==1, "Noise level should be the same for all boxes."
        tau2, tau0, tau1 = np.array(block['schedules']).squeeze()
        block_data['taus'] = np.array([tau0, tau1, tau2])
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
        self.reward = 0. # reward rate
        self.food = 0. # frequency of getting food

    def _on_step(self) -> bool:
        for reward, info in zip(self.locals['rewards'], self.locals['infos']):
            self.reward = self.gamma*self.reward+(1-self.gamma)*reward
            self.food = self.gamma*self.food+(1-self.gamma)*info['observation'][-1]
        if self.n_calls%self.disp_freq==0:
            self.pbar.set_description(
                "[Reward rate {:.2f}], [Food freq {:.2f}]".format(self.reward, self.food)
            )
        return super()._on_step()
