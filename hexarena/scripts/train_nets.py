from pathlib import Path
import numpy as np
from jarvis.config import from_cli, choices2configs, Config
from jarvis.manager import Manager
from jarvis.utils import tqdm, get_defaults, tensor2array, array2tensor
from irc.model import NetworkBeliefModel

from .. import STORE_DIR
from .common import get_block_ids, create_env_and_model
from .compute_beliefs import fetch_beliefs


def create_manager(
    subject: str, kappa: float, num_samples: int, read_only: bool = False,
    save_interval: int = 1, patience: float = 24.,
) -> Manager:
    r"""Creates a manager for training belief nets.

    Args
    ----
    subject, kappa, num_samples:
        Subject name, cue reliability and number of state samples, see `main`
        for more details.
    read_only:
        If ``True``, pre-computed beliefs are not loaded, for fetching purpose
        only.
    save_interval, patience:
        Arguments of the manager, see `Manager` for more details.

    """
    manager = Manager(
        STORE_DIR/'belief_nets'/subject, save_interval=save_interval, patience=patience,
    )
    block_ids = get_block_ids(subject, kappa)
    if not read_only:
        manager.observations, manager.actions, _, manager.beliefs = zip(*[
            fetch_beliefs(
                subject, kappa, num_samples, session_id, block_idx,
            ) for session_id, block_idx in tqdm(block_ids, unit='block', leave=False)
        ])
    manager.default = {
        'subject': subject, 'kappa': kappa, 'num_samples': num_samples,
        'seed': 0, 'model_kw': {get_defaults(NetworkBeliefModel)['z_dim']},
        'update_kw': {'z_reg': 1e-4, 'num_epochs': 300},
        'init_kw': {'z_reg': 1e-4, 'num_epochs': 200},
    }

    def setup(config: Config):
        r"""
        config:
          - subject: str        # subject name
          - kappa: float        # stimulus reliability
          - num_samples: int    # number of samples used in belief computation
          - seed: int           # network initialiation seed
          - model_kw: dict      # arguments of SamplingBeliefModel
          - init_kw: dict       # arguments for training init_net
          - update_kw: dict     # arguments for training update_net
        """
        manager.seed = config.seed
        manager.model_kw = config.model_kw
        manager.train_kw = {
            'update_kw': config.update_kw, 'init_kw': config.init_kw,
        }
        _, manager.model = create_env_and_model(subject, kappa, model_kw=manager.model_kw)
        return 1 # one epoch for manager
    def reset():
        manager.model.init_net.reset(manager.seed)
        manager.model.update_net.reset(manager.seed)
        manager.stats_u, manager.stats_i = None, None
    def step():
        manager.stats_u, manager.stats_i = manager.model.train_nets(
            manager.observations, manager.actions, manager.beliefs,
            strict_init=True, **manager.train_kw,
        )
    def get_ckpt():
        return tensor2array({
            'state_dict': manager.model.state_dict(),
            'stats_u': manager.stats_u, 'stats_i': manager.stats_i,
        })
    def load_ckpt(ckpt):
        ckpt = array2tensor(ckpt, manager.model.device)
        manager.model.load_state_dict(ckpt['state_dict'])
        manager.stats_u = ckpt['stats_u']
        manager.stats_i = ckpt['stats_i']
        return 1
    manager.setup = setup
    manager.reset = reset
    manager.step = step
    manager.get_ckpt = get_ckpt
    manager.load_ckpt = load_ckpt
    return manager


def fetch_best_net(
    subject: str, kappa: float, num_samples: int,
) -> tuple[Config, NetworkBeliefModel]|None:
    r"""Fetches the best NetworkBeliefModel instance.

    The NetworkBeliefModel object with lowest validation KL divergence of belief
    update net is considered as the best model.

    Args
    ----
    subject, kappa, num_samples:
        Subject name, cue reliability and number of state samples. See `main`
        for more details.

    """
    manager = create_manager(subject, kappa, num_samples, read_only=True)
    cond = {'subject': subject, 'kappa': kappa, 'num_samples': num_samples}
    min_loss, best_key = float('inf'), None
    for key, _ in manager.completed(cond=cond):
        stats = manager.ckpts[key]['stats_u']
        losses = (stats['losses_val']*stats['alphas']).sum(axis=1)
        loss = stats['losses_val'][np.argmin(losses), 0]
        if loss<min_loss:
            min_loss = loss
            best_key = key
    if best_key is None:
        print(f"No trained networks found for {subject} (kappa={kappa}, num_samples={num_samples})")
        return None
    manager.setup(manager.configs[best_key])
    manager.load_ckpt(manager.ckpts[best_key])
    model = manager.model
    return manager.configs[best_key], model


def main(
    subject: str, kappa: float, num_samples: int,
    choices: dict|Path|str|None = None,
    num_works: int|None = None,
    **kwargs,
):
    r"""Trains belief networks for a belief model.

    Args
    ----
    data_dir, store_dir, subject, kappa, num_samples, patience:
        Arguments of `create_manager`.
    choices:
        Job specifications which is a dict containing possible values.
    num_works:
        Number of works to process, see `Manager.batch` for more details.

    """
    manager = create_manager(
        subject, kappa, num_samples, **kwargs,
    )
    if choices is None or isinstance(choices, dict):
        choices = Config(choices).fill({
            'seed': list(range(6)),
            'model_kw.z_dim': [8, 16, 32],
        })
    configs = choices2configs(choices)
    manager.batch(configs, 1, num_works)


if __name__=='__main__':
    main(**from_cli().fill({
        'subject': 'marco',
        'kappa': 0.1,
        'num_samples': 1000,
    }))
