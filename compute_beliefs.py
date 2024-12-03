import os
from pathlib import Path
import numpy as np
from jarvis.config import from_cli, Config
from jarvis.utils import tensor2array, array2tensor
from jarvis.manager import Manager
from irc.model import SamplingBeliefModel

from hexarena import DATA_DIR, STORE_DIR
from hexarena.env import SimilarBoxForagingEnv
from hexarena.alias import Array
from hexarena.utils import get_valid_blocks, load_monkey_data, align_monkey_data


def prepare_blocks(data_dir: Path, subject: str, kappa: float) -> list[tuple[str, int]]:
    r"""Prepares blocks to process.

    Args
    ----
    data_dir:
        Directory that contains experiment data file, e.g. 'data_marco.mat'.
    subject:
        Subject name.
    kappa:
        Cue reliability parameter.

    Returns
    -------
    block_ids:
        The block ID `(session_id, block_idx)` of all blocks to be processed.

    """
    data_path = data_dir/f'data_{subject}.mat'
    assert os.path.exists(data_path), f"{data_path} not found"
    assert subject in ['marco', 'viktor'], (
        "Only 'marco' and 'viktor' can be processed now."
    )
    block_infos = get_valid_blocks(data_path, min_pos_ratio=0.5, min_gaze_ratio=0.1)

    block_ids = []
    for session_id in block_infos:
        for block_idx in block_infos[session_id]:
            block_data = load_monkey_data(data_path, session_id, block_idx)
            to_process = np.all(block_data['kappas']==kappa)
            if subject=='marco' and set(block_data['taus'])!=set([15., 21., 35.]):
                to_process = False
            if subject=='viktor' and set(block_data['taus'])!=set([7., 14., 21.]):
                to_process = False
            if to_process:
                block_ids.append((session_id, block_idx))
    print(f'{len(block_ids)} valid blocks found.')
    return block_ids

def create_model(subject: str, kappa: float) -> tuple[SimilarBoxForagingEnv, SamplingBeliefModel]:
    r"""Creates default belief model.

    Args
    ----
    subject:
        Subject name, only 'marco' and 'viktor' are supported now.
    kappa:
        Cue reliability parameter.

    Returns
    -------
    env:
        A foraging environment. For 'marco', it is the exponential schedule. For
        'viktor' it is the Gamma distribution schedule. Color cues are also
        presented in different ways.
    model:
        A sampling-based belief model, with state independence specified.

    """
    if subject=='marco':
        env = SimilarBoxForagingEnv(
            box={
                '_target_': 'hexarena.box.StationaryBox', 'kappa': kappa, 'num_levels': 10,
            },
            boxes=[{'tau': tau} for tau in [35, 21, 15]],
        )
        model = SamplingBeliefModel(env,
            s_idcs=[[0], [1], [2, 3], [4, 5], [6, 7]],
        )
    if subject=='viktor':
        env = SimilarBoxForagingEnv(
            box={
                '_target_': 'hexarena.box.GammaLinearBox', 'kappa': kappa, 'max_interval': 40,
            },
            boxes=[{'tau': tau} for tau in [21, 14, 7]],
        )
        phi = {
            'embedder._target_': 'hexarena.box.LinearBoxStateEmbedder',
            'mlp_features': [16, 8],
        }
        model = SamplingBeliefModel(
            env, p_s={'phis': [phi]*3},
        )
    return env, model

def create_manager(
    data_dir: Path, store_dir: Path, subject: str, kappa: float,
    save_interval: int = 10, patience: float = 12.,
) -> Manager:
    r"""Creates a manager to handle batch processing.

    Args
    ----
    data_dir, store_dir:
        Directory of data and storage respectively.
    subject:
        Subject name.
    kappa:
        Cue reliability parameter.
    save_interval, patience:
        Arguments of the manager object, see `jarvis.manager.Manager` for more
        details.

    Returns
    -------
    manager:
        A manager object that computes beliefs for each experiment block. Batch
        processing and resuming from checkpoint are supported.

    """
    manager = Manager(
        store_dir=Path(store_dir)/'beliefs'/subject,
        save_interval=save_interval, patience=patience,
    )
    manager.data_path = data_dir/f'data_{subject}.mat'
    manager.env, manager.model = create_model(subject, kappa)
    manager.model.use_sample = True
    manager.default = {
        'num_samples': 1000,
    }

    def setup(config: Config):
        r"""
        config:
          - session_id: str
          - block_idx: int
          - num_samples: int

        """
        block_data = load_monkey_data(manager.data_path, config.session_id, config.block_idx)
        block_data = align_monkey_data(block_data)
        env_data = manager.env.convert_experiment_data(block_data)
        manager.observations, manager.actions, _ = \
            manager.env.extract_observation_action_reward(env_data)
        manager.model.num_samples = config.num_samples
        return len(manager.actions)
    def reset():
        observations = manager.observations
        known, belief, _ = manager.model.init_belief(observations[0])
        manager.knowns = [known]
        manager.beliefs = [belief]
    def step():
        t = len(manager.knowns)-1
        known, belief, _ = manager.model.update_belief(
            manager.knowns[t], manager.beliefs[t],
            manager.actions[t], manager.observations[t+1],
        )
        manager.knowns.append(known)
        manager.beliefs.append(belief)
    def get_ckpt():
        return tensor2array({
            'knowns': manager.knowns,
            'beliefs': manager.beliefs,
        })
    def load_ckpt(ckpt):
        ckpt = array2tensor(ckpt, manager.model.device)
        manager.knowns = ckpt['knowns']
        manager.beliefs = ckpt['beliefs']
        return len(manager.knowns)-1
    def pbar_desc(config):
        return f'{config.session_id}-B{config.block_idx}'
    manager.setup = setup
    manager.reset = reset
    manager.step = step
    manager.get_ckpt = get_ckpt
    manager.load_ckpt = load_ckpt
    manager.pbar_desc = pbar_desc
    return manager

def fetch_beliefs(
    data_dir: Path, store_dir: Path, subject: str,
    session_id: str, block_idx: int, num_samples: int = 1000,
) -> tuple[Array, Array, Array, Array]:
    r"""Fetches computed beliefs.

    Args
    ----
    data_dir, store_dir, subject:
        Arguments of `create_manager`.
    session_id, block_idx, num_samples:
        Identifier of computed results.

    Returns
    -------
    observations, actions, knowns, beliefs:
        Sequences of data for the specified block.

    """
    manager = create_manager(data_dir, store_dir, subject)
    ckpt = manager.process({
        'session_id': session_id, 'block_idx': block_idx, 'num_samples': num_samples,
    })
    observations = np.array(ckpt['observations'])
    actions = np.array(ckpt['actions'])
    knowns = np.array(ckpt['knowns'])
    beliefs = np.array(ckpt['beliefs'])
    return observations, actions, knowns, beliefs

def main(
    data_dir: Path|str,
    store_dir: Path|str,
    subject: str, kappa: float,
    num_samples: int = 1000,
    num_works: int|None = None,
    save_interval: int = 10,
    patience: float = 12.,
):
    data_dir, store_dir = Path(data_dir), Path(store_dir)
    block_ids = prepare_blocks(data_dir, subject, kappa)
    manager = create_manager(
        data_dir, store_dir, subject, kappa, save_interval, patience,
    )
    configs = [Config({
        'session_id': session_id,
        'block_idx': block_idx,
        'num_samples': num_samples,
    }) for session_id, block_idx in block_ids]
    np.random.default_rng().shuffle(configs)
    manager.batch(
        configs, num_works=num_works, pbar_kw={'unit': 'block', 'leave': True},
        process_kw={'pbar_kw': {'unit': 'step'}},
    )

if __name__=='__main__':
    main(**from_cli().fill({
        'data_dir': DATA_DIR,
        'store_dir': STORE_DIR,
        'subject': 'marco',
        'kappa': 0.1,
    }))
