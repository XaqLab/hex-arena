import os
from pathlib import Path
import numpy as np
import torch
from jarvis.config import from_cli, Config
from jarvis.utils import tensor2array, array2tensor
from jarvis.manager import Manager
from irc.model import SamplingBeliefModel

from hexarena import DATA_DIR, STORE_DIR
from hexarena.env import SimilarBoxForagingEnv
from hexarena.alias import Array, Tensor
from hexarena.utils import get_valid_blocks, load_monkey_data, align_monkey_data


def get_data_path(data_dir: Path, subject: str) -> Path:
    r"""Returns data file path.

    Args
    ----
    data_dir:
        Directory that contains experiment data file, e.g. 'data_marco.mat'.
    subject:
        Subject name, only 'marco' and 'viktor' are supported now.

    Returns
    -------
    data_path:
        Path to the MAT data file.

    """
    data_path = data_dir/f'data_{subject}.mat'
    return data_path

def prepare_blocks(
    data_dir: Path, subject: str, kappa: float,
    verbose: bool = True,
) -> list[tuple[str, int]]:
    r"""Prepares blocks to process.

    Args
    ----
    data_dir, subject:
        Data directory and subject name, see `get_data_path` for more details.
    kappa:
        Cue reliability parameter. The same value is used for all boxes, and
        higher values mean less noise.
    verbose:
        Whether to print a summary message.

    Returns
    -------
    block_ids:
        The block ID `(session_id, block_idx)` of all blocks to be processed.

    """
    data_path = get_data_path(data_dir, subject)
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
    if verbose:
        print(f'{len(block_ids)} valid blocks found.')
    return block_ids

def create_model(
    subject: str, kappa: float,
    env_kw: dict|None = None,
    model_kw: dict|None = None,
) -> tuple[SimilarBoxForagingEnv, SamplingBeliefModel]:
    r"""Creates default belief model.

    Args
    ----
    subject:
        Subject name, see `get_data_path` for more details.
    kappa:
        Cue reliability parameter, see `prepare_blocks` for more details.
    env_kw:
        Keyword arguments of the environment.
    model_kw:
        Keyword arguments of the belief model.

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
        env_kw = Config(env_kw).fill({
            'box': {
                '_target_': 'hexarena.box.StationaryBox',
                'num_levels': 10,
            },
            'boxes': [{'tau': tau} for tau in [35, 21, 15]],
        })
        model_kw = Config(model_kw).fill({
            's_idcs': [[0], [1], [2, 3], [4, 5], [6, 7]],
        })
    if subject=='viktor':
        env_kw = Config(env_kw).fill({
            'box': {
                '_target_': 'hexarena.box.GammaLinearBox',
                'max_interval': 40,
            },
            'boxes': [{'tau': tau} for tau in [21, 14, 7]],
        })
        model_kw = Config(model_kw).fill({
            'p_s.phis': [{
                'embedder._target_': 'hexarena.box.LinearBoxStateEmbedder',
                'mlp_features': [16, 8],
            }]*3,
        })
    env_kw.update({'box.kappa': kappa})
    env = SimilarBoxForagingEnv(**env_kw)
    model = SamplingBeliefModel(env, **model_kw)
    return env, model

def create_manager(
    data_dir: Path, store_dir: Path, subject: str, kappa: float,
    save_interval: int = 10, patience: float = 12.,
) -> Manager:
    r"""Creates a manager to compute beliefs.

    Args
    ----
    data_dir:
        Data directory, see `get_data_path` for more details.
    store_dir:
        Directory for storing computed beliefs.
    subject:
        Subject name, see `get_data_path` for more details.
    kappa:
        Cue reliability parameter, see `prepare_blocks` fore more details.
    save_interval, patience:
        Arguments of the Manager object, see `jarvis.manager.Manager` for more
        details.

    Returns
    -------
    manager:
        A Manager object that computes beliefs for each experiment block. Batch
        processing and resuming from checkpoint are supported.

    """
    manager = Manager(
        store_dir=store_dir/'beliefs'/subject,
        save_interval=save_interval, patience=patience,
    )
    manager.data_path = get_data_path(data_dir, subject)
    manager.env, manager.model = create_model(subject, kappa)
    manager.model.use_sample = True
    manager.default = {
        'subject': subject, 'kappa': kappa,
        'num_samples': 1000,
    }

    def setup(config: Config):
        r"""
        config:
          - subject: str
          - kappa: float
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
) -> tuple[Array, Array, Array, Tensor]:
    r"""Fetches computed beliefs.

    Args
    ----
    data_dir, store_dir, subject:
        Arguments of `create_manager`.
    session_id, block_idx, num_samples:
        Identifier of computed results, see `create_manager.setup` for more
        details.

    Returns
    -------
    observations, actions, knowns, beliefs:
        Sequences of data for the specified block.

    """
    data_path = get_data_path(data_dir, subject)
    block_data = load_monkey_data(data_path, session_id, block_idx)
    kappa = np.unique(block_data['kappas']).item()
    manager = create_manager(data_dir, store_dir, subject, kappa)
    manager.process({
        'session_id': session_id, 'block_idx': block_idx, 'num_samples': num_samples,
    })
    observations, actions = manager.observations, manager.actions
    knowns = np.array(manager.knowns)
    beliefs = torch.stack(manager.beliefs)
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
    r"""

    Uses a manager to compute beliefs of all blocks of given subject and cue
    reliability.

    Args
    ----
    data_dir:
        Data directory, see `get_data_path` for more details.
    store_dir:
        Directory of storage, see `create_manager` for more details.
    subject:
        Subject name, see `get_data_path` for more details.
    kappa:
        Cue reliability, see `prepare_blocks` for more details.
    num_samples:
        Number of state samples used in estimating new belief at each time step.
    num_works:
        Number of blocks to process.
    save_interval, patience:
        Arguments of the Manager object, see `jarvis.manager.Manager` for more
        details.

    """
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
