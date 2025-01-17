from pathlib import Path
import numpy as np
import torch
from jarvis.config import from_cli, choices2configs, Config
from jarvis.manager import Manager
from jarvis.utils import tqdm, get_defaults, tensor2array, array2tensor
from irc.dist.net import BaseDistributionNet

from .. import STORE_DIR
from .common import get_block_ids, create_env_and_model
from .compute_beliefs import fetch_beliefs


def create_manager(
    subject: str, kappa: float, num_samples: int,
    save_interval: int = 1, patience: float = 1.,
) -> Manager:
    r"""Creates a manager for compressing beliefs.

    Args
    ----
    subject, kappa, num_samples:
        Subject name, cue reliability and number of state samples, see `main`
        for more details.
    save_interval, patience:
        Arguments of the manager, see `Manager` for more details.

    """
    manager = Manager(
        STORE_DIR/'belief_vaes'/subject, save_interval=save_interval, patience=patience,
    )
    manager.env, manager.model = create_env_and_model(subject, kappa)
    manager.block_ids = get_block_ids(subject, kappa)
    manager.default = {
        'subject': subject, 'kappa': kappa,
        'num_samples': num_samples, 'split': 0.95, 'seed': 0,
        'vae_kw': {
            k: v for k, v in get_defaults(BaseDistributionNet).items()
            if k not in ['z_dim', 'p_x', 'ebd_y']
        },
        'regress_kw': get_defaults(BaseDistributionNet.train),
    }

    def setup(config: Config, read_only: bool = False):
        r"""
        config:
          - subject: str        # subject name
          - kappa: float        # stimulus reliability
          - num_samples: int    # number of samples used in belief computation
          - z_dim: int          # latent state dimension of VAE
          - vae_kw: dict        # arguments for creating VAE, see `BaseDistributionNet`
          - regress_kw: dict    # arguments for `BaseDistributionNet.regress`.
          - split: float        # train/test split of blocks
          - seed: int           # random seed for train/test split

        """
        manager.subject, manager.kappa = config.subject, config.kappa
        manager.num_samples = config.num_samples
        manager.z_dim, manager.vae_kw = config.z_dim, config.vae_kw
        manager.regress_kw = config.regress_kw
        manager.seed = config.seed
        manager.belief_vae = manager.model.create_belief_vae(
            z_dim=manager.z_dim, **manager.vae_kw,
        )
        if not read_only:
            _, _, _, manager.beliefs = zip(*[
                fetch_beliefs(
                    subject, kappa, num_samples, session_id, block_idx,
                ) for session_id, block_idx in tqdm(
                    manager.block_ids, desc='Fetching beliefs', unit='block', leave=False,
                )
            ])
            n = len(manager.block_ids)
            n_train = int(np.floor(n*config.split))
            _idxs = np.random.default_rng(config.seed).choice(n, n, replace=False)
            manager.idxs = {'train': _idxs[:n_train], 'test': _idxs[n_train:]}
        return float('inf')
    def reset():
        manager.belief_vae.reset(manager.seed)
        manager.losses = {'train': [], 'val': [], 'test': []} # KL losses only
        manager.min_loss, manager.best_state = float('inf'), None
    def step():
        beliefs_train = torch.cat([manager.beliefs[i] for i in manager.idxs['train']])
        manager.belief_vae.rng = np.random.default_rng(manager.seed) # reset train/val split
        stats = manager.belief_vae.regress(
            beliefs_train, beliefs_train, num_epochs=1,
            pbar_kw={'disable': True}, **manager.regress_kw,
        )
        loss = (stats['losses_val'][-1]*stats['alphas']).sum()
        if loss<manager.min_loss:
            manager.min_loss = loss
            manager.best_state = manager.belief_vae.state_dict()
        manager.losses['train'].append(stats['losses_train'][-1, 0])
        manager.losses['val'].append(stats['losses_val'][-1, 0])

        beliefs_test = torch.cat([manager.beliefs[i] for i in manager.idxs['test']])
        with torch.no_grad():
            _, _, recons = manager.belief_vae(beliefs_test)
        kl_losses, _ = zip(*[
            manager.belief_vae.p_x.kl_divergence(recons[i], beliefs_test[i])
            for i in range(len(recons))
        ])
        manager.losses['test'].append(torch.stack(kl_losses).mean().item())
    def get_ckpt():
        return tensor2array({
            'losses': manager.losses,
            'min_loss': manager.min_loss, 'best_state': manager.best_state,
            'last_state': manager.belief_vae.state_dict(),
        })
    def load_ckpt(ckpt):
        ckpt = array2tensor(ckpt, manager.belief_vae.device)
        manager.losses = ckpt['losses']
        manager.min_loss = ckpt['min_loss']
        manager.best_state = ckpt['best_state']
        manager.belief_vae.load_state_dict(ckpt['last_state'])
        return len(manager.losses['test'])

    manager.setup = setup
    manager.reset = reset
    manager.step = step
    manager.get_ckpt = get_ckpt
    manager.load_ckpt = load_ckpt
    return manager


def fetch_best_vae(
    subject: str, kappa: float, num_samples: int, z_dim: int,
    min_epoch: int = 20, cond: dict|None = None,
) -> BaseDistributionNet|None:
    r"""Fetches the best belief VAE satisfying conditions.

    Args
    ----
    data_dir, store_dir, subject, kappa, num_samples:
        Arguments for the manager, see `create_manager` for more details.
    z_dim:
        Latent state dimension of belief VAE.
    min_epoch:
        Mininum number of training epochs, see `Manager.completed` for more
        details.
    cond:
        Conditions of VAE configuration, e.g. the regularization coefficients.
        See `Manager.completed` for more details.

    Returns
    -------
    belief_vae:
        The best belief VAE in terms of the testing loss. If no trained VAEs are
        found, return ``None``.

    """
    manager = create_manager(subject, kappa, num_samples)
    cond = Config(cond).fill({
        'subject': subject, 'kappa': kappa, 'num_samples': num_samples, 'z_dim': z_dim,
    })
    min_loss, best_key = float('inf'), None
    for key, _ in manager.completed(min_epoch, cond=cond):
        ckpt = manager.ckpts[key]
        losses_val = ckpt['losses']['val']
        losses_test = ckpt['losses']['test']
        loss = losses_test[losses_val.index(min(losses_val))]
        if loss<min_loss:
            min_loss = loss
            best_key = key
    if best_key is None:
        print(f"No VAE satisfying conditions found (kappa {kappa}, z_dim {z_dim}).")
        return None
    manager.setup(manager.configs[best_key], read_only=True)
    manager.load_ckpt(manager.ckpts[best_key])
    belief_vae = manager.belief_vae
    return belief_vae


def main(
    subject: str, kappa: float, num_samples: int,
    choices: dict|Path|str|None = None,
    num_epochs: int = 80,
    num_works: int|None = None,
    max_errors: int = 10,
    **kwargs,
):
    r"""Compresses pre-computed beliefs of blocks of the same environment.

    Args
    ----
    subject_name:
        Subject name, can be 'marco', 'dylan' or 'viktor'.
    kappa:
        Cue reliability, higher value means more reliable color cue.
    num_samples:
        Number of state samples used in estimating new belief at each time step.
    choices:
        Job specifications, corresponding to a dictionary with keys of 'config'.
        See `setup` in `create_manager` for more details.
    num_epochs, num_works, max_errors:
        Arguments for batch processing, see `Manager.batch` for more details.
    kwargs:
        Keyword arguments of the manager, see `Manager` for more details.

    """
    manager = create_manager(subject, kappa, num_samples, **kwargs)
    if choices is None or isinstance(choices, dict):
        choices = Config(choices).fill({
            'z_dim': list(range(12)),
            'seed': list(range(6)),
            'regress_kw.z_reg': [1., 0.1, 0.01],
        })
    configs = choices2configs(choices)
    manager.batch(
        configs, num_epochs, num_works, max_errors,
        pbar_kw={'unit': 'block', 'leave': True},
    )


if __name__=='__main__':
    main(**from_cli().fill({
        'subject': 'marco',
        'kappa': 0.1,
        'num_samples': 1000,
    }))

