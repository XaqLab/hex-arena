import os, torch
from pathlib import Path
import numpy as np
from jarvis.config import from_cli, Config
from jarvis.manager import Manager
from irc.model import SamplingBeliefModel

from hexarena.env import SimilarBoxForagingEnv
from hexarena.alias import Array
from hexarena.utils import get_valid_blocks, load_monkey_data, align_monkey_data


DATA_DIR = Path(__file__).parent/'data'
STORE_DIR = Path(__file__).parent/'store'

def prepare_blocks(data_dir: Path, subject: str) -> list[tuple[str, int]]:
    r"""Prepares blocks to process.

    Args
    ----
    data_dir:
        Directory that contains experiment data file, e.g. 'data_marco.mat'.
    subject:
        Subject name.

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
            to_process = False
            if subject=='marco':
                if np.all(block_data['kappas']==0.01) and set(block_data['taus'])==set([15., 21., 35.]):
                    to_process = True
            if subject=='viktor':
                if np.all(block_data['kappas']==0.) and set(block_data['taus'])==set([7., 14., 21.]):
                    to_process = True
            if to_process:
                block_ids.append((session_id, block_idx))
    print(f'{len(block_ids)} valid blocks found.')
    return block_ids

def create_model(subject: str) -> tuple[SimilarBoxForagingEnv, SamplingBeliefModel]:
    r"""Creates default belief model.

    Args
    ----
    subject:
        Subject name, only 'marco' and 'viktor' are supported now.

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
                '_target_': 'hexarena.box.StationaryBox',
                'num_patches': 1, 'num_levels': 10, 'num_grades': 8,
            },
            boxes=[{'tau': tau} for tau in [35, 21, 15]],
        )
        model = SamplingBeliefModel(env,
            s_idcs=[[0], [1], [2, 3], [4, 5], [6, 7]],
        )
    if subject=='viktor':
        env = SimilarBoxForagingEnv(
            box={
                '_target_': 'hexarena.box.GammaLinearBox',
                'num_patches': 1, 'max_interval': 40,
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
    data_dir: Path, store_dir: Path, subject: str,
    save_interval: int = 10, patience: float = 12.,
) -> Manager:
    r"""Creates a manager to handle batch processing.

    Args
    ----
    data_dir, store_dir:
        Directory of data and storage respectively.
    subject_dir:
        Subject name.
    save_interval, patience:
        Arguments of the manager object, see `jarvis.manager.Manager` for more
        details.

    Returns
    -------
    manager:
        A manager object that computes beliefs for each experiment block. Batch
        processing and resuming from checkpoint are supported.

    """
    data_path = data_dir/f'data_{subject}.mat'
    env, model = create_model(subject)
    model.use_sample = True
    workspace = {}
    def setup(config: Config):
        r"""
        config:
          - session_id: str
          - block_idx: int
          - num_samples: int

        """
        session_id, block_idx = config.session_id, config.block_idx
        block_data = load_monkey_data(data_path, session_id, block_idx)
        block_data = align_monkey_data(block_data)
        env_data = env.convert_experiment_data(block_data)
        observations, actions, _ = env.extract_observation_action_reward(env_data)
        workspace['observations'] = observations
        workspace['actions'] = actions
        model.num_samples = config.num_samples
        return len(actions)
    def reset():
        observations = workspace['observations']
        known, belief, info = model.init_belief(observations[0])
        workspace.update({
            'knowns': [known], 'beliefs': [belief], 'infos': [info],
        })
    def step():
        t = len(workspace['knowns'])-1
        known, belief, info = model.update_belief(
            workspace['knowns'][t], workspace['beliefs'][t],
            workspace['actions'][t], workspace['observations'][t+1],
        )
        workspace['knowns'].append(known)
        workspace['beliefs'].append(belief)
        workspace['infos'].append(info)
    def get_ckpt():
        return {
            'observations': workspace['observations'], 'actions': workspace['actions'],
            'knowns': workspace['knowns'], 'infos': workspace['infos'],
            'beliefs': [b.data.cpu().numpy() for b in workspace['beliefs']],
        }
    def load_ckpt(ckpt):
        workspace.update({
            'knowns': ckpt['knowns'], 'infos': ckpt['infos'],
            'beliefs': [torch.tensor(b, device=model.device) for b in ckpt['beliefs']],
        })
        return len(workspace['knowns'])-1
    def pbar_desc(config):
        return f'{config.session_id}-B{config.block_idx}'

    manager = Manager(
        store_dir=Path(store_dir)/'beliefs'/subject,
        save_interval=save_interval, patience=patience,
    )
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
    subject: str,
    num_samples: int,
    num_works: int|None = None,
    save_interval: int = 10,
    patience: float = 12.,
):
    data_dir, store_dir = Path(data_dir), Path(store_dir)
    block_ids = prepare_blocks(data_dir, subject)
    manager = create_manager(
        data_dir, store_dir, subject, save_interval, patience,
    )
    configs = [{
        'session_id': session_id,
        'block_idx': block_idx,
        'num_samples': num_samples,
    } for session_id, block_idx in block_ids]
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
        'num_samples': 1000,
    }))
