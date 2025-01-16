import numpy as np
import torch
from jarvis.config import from_cli, Config
from jarvis.utils import tensor2array, array2tensor
from jarvis.manager import Manager

from .. import STORE_DIR
from ..alias import Array, Tensor
from ..utils import load_monkey_data, align_monkey_data
from .common import get_block_ids, create_env_and_model


def create_manager(
    subject: str, kappa: float, num_samples: int,
    save_interval: int = 10, patience: float = 12.,
) -> Manager:
    r"""Creates a manager for computing beliefs.

    Args
    ----
    subject, kappa, num_samples:
        Subject name, cue reliability and number of state samples, see `main`
        for more details.
    save_interval, patience:
        Arguments of the manager, see `Manager` for more details.

    """
    manager = Manager(
        STORE_DIR/'beliefs'/subject, save_interval=save_interval, patience=patience,
    )
    manager.env, manager.model = create_env_and_model(subject, kappa)
    manager.model.use_sample = True
    manager.default = {
        'subject': subject, 'kappa': kappa, 'num_samples': num_samples,
    }

    def setup(config: Config) -> int:
        r"""
        config:
          - subject: str
          - kappa: float
          - session_id: str
          - block_idx: int
          - num_samples: int

        """
        block_data = load_monkey_data(config.subject, config.session_id, config.block_idx)
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
    subject: str, kappa: float, num_samples: int,
    session_id: str, block_idx: int,
) -> tuple[Array, Array, Array, Tensor]:
    r"""Fetches computed beliefs for one block.

    Args
    ----
    subject, kappa, num_samples:
        Subject name, cue reliability and number of state samples. See `main`
        for more details.
    session_id, block_idx:
        Session ID and block index for specifying a block.

    Returns
    -------
    observations, actions, knowns, beliefs:
        Sequences of data for the specified block.

    """
    manager = create_manager(subject, kappa)
    manager.process({
        'session_id': session_id, 'block_idx': block_idx, 'num_samples': num_samples,
    })
    observations, actions = manager.observations, manager.actions
    knowns = np.array(manager.knowns)
    beliefs = torch.stack(manager.beliefs)
    return observations, actions, knowns, beliefs


def main(
    subject: str, kappa: float, num_samples: int,
    num_works: int|None = None,
    **kwargs,
):
    r"""
    Computes beliefs of all blocks given subject and cue reliability.

    Args
    ----
    subject_name:
        Subject name, can be 'marco', 'dylan' or 'viktor'.
    kappa:
        Cue reliability, higher value means more reliable color cue.
    num_samples:
        Number of state samples used in estimating new belief at each time step.
    num_works:
        Number of blocks to process.
    kwargs:
        Keyword arguments of the manager, see `Manager` for more details.

    """
    manager = create_manager(subject, kappa, num_samples, **kwargs)
    block_ids = get_block_ids(subject, kappa)
    configs = [Config({
        'session_id': session_id, 'block_idx': block_idx,
    }) for session_id, block_idx in block_ids]
    np.random.default_rng().shuffle(configs)
    manager.batch(
        configs, num_works=num_works,
        pbar_kw={'unit': 'block', 'leave': True},
        process_kw={'pbar_kw': {'unit': 'steps'}},
    )

if __name__=='__main__':
    main(**from_cli().fill({
        'subject': 'marco',
        'kappa': 0.1,
        'num_samples': 1000,
    }))
