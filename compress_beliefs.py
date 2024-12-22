from pathlib import Path
import numpy as np
import torch
from jarvis.config import from_cli, choices2configs, Config
from jarvis.manager import Manager
from jarvis.utils import tqdm, tensor2array, array2tensor, get_defaults
from irc.dist.net import BaseDistributionNet

from hexarena import DATA_DIR, STORE_DIR

from compute_beliefs import (
    get_data_path, create_model, prepare_blocks, fetch_beliefs,
)


def create_manager(
    data_dir: Path, store_dir: Path, subject: str, kappa: float,
    num_samples: int = 1000, patience: float = 12.,
) -> Manager:
    r"""Creates a manager to train belief VAE models.

    Args
    ----
    data_dir:
        Data directory, see `get_data_path` for more details.
    store_dir:
        Directory of storing trained VAE models.
    subject:
        Subject name, see `get_data_path` for more details.
    kappa:
        Cue reliability, see `get_data_path` for more details.
    num_samples:
        Number of samples used in belief estimation.
    patience:
        Arguments of the Manager object, see `Manager` for more details.

    """
    manager = Manager(
        store_dir=store_dir/'belief_vaes'/subject, patience=patience,
    )
    manager.block_ids = prepare_blocks(data_dir, subject, kappa, verbose=False)
    _, _, _, manager.beliefs = zip(*[
        fetch_beliefs(
            data_dir, store_dir, subject, session_id, block_idx, num_samples,
        ) for session_id, block_idx in tqdm(
            manager.block_ids, desc='Fetching beliefs', unit='block', leave=False,
        )
    ])
    manager.data_path = get_data_path(data_dir, subject)
    manager.env, manager.model = create_model(subject, kappa)
    manager.default = {
        'subject': subject, 'kappa': kappa,
        'num_samples': 1000, 'split': 0.95, 'seed': 0,
        'vae_kw': {
            k: v for k, v in get_defaults(BaseDistributionNet).items()
            if k not in ['z_dim', 'p_x', 'ebd_y']
        },
        'regress_kw': get_defaults(BaseDistributionNet.train),
    }

    def setup(config: Config):
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
        n = len(manager.block_ids)
        n_test = int(np.ceil(n*(1-config.split)))
        n_train = n-n_test
        _idxs = np.random.default_rng(config.seed).choice(n, n, replace=False)
        manager.idxs = {'train': _idxs[:n_train], 'test': _idxs[n_train:]}
        return float('inf')
    def reset():
        manager.belief_vae = manager.model.create_belief_vae(
            z_dim=manager.z_dim, **manager.vae_kw,
        )
        manager.losses = {'train': [], 'val': [], 'test': []}
        manager.min_loss, manager.best_state = float('inf'), None
    def step():
        manager.belief_vae.rng = np.random.default_rng(manager.seed) # reset train/val split
        beliefs_train = torch.cat([manager.beliefs[i] for i in manager.idxs['train']])
        beliefs_test = torch.cat([manager.beliefs[i] for i in manager.idxs['test']])
        stats = manager.belief_vae.regress(
            beliefs_train, beliefs_train,
            num_epochs=1, pbar_kw={'disable': True}, **manager.regress_kw,
        )
        loss = (stats['losses_val'][-1]*stats['alphas']).sum()
        if loss<manager.min_loss:
            manager.min_loss = loss
            manager.best_state = manager.belief_vae.state_dict()
        manager.losses['train'].append(stats['losses_train'][-1, 0])
        manager.losses['val'].append(stats['losses_val'][-1, 0])
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

def main(
    data_dir: Path|str, store_dir: Path|str,
    subject: str, kappa: float, num_samples: int,
    patience: float = 12,
    choices: dict|Path|str|None = None,
    num_epochs: int = 80,
    num_works: int|None = None,
):
    r"""

    Uses a manager to compress beliefs of all blocks given subject and cue
    reliability.

    Args
    ----
    data_dir, store_dir, subject, kappa, num_samples, patience:
        Arguments of `create_manager`.
    choices:
        Job specifications which is a dict containing possible values.
    num_epochs:
        Number of epochs of training belief VAE models.
    num_works:
        Number of works to process, see `Manager.batch` for more details.

    """
    data_dir, store_dir = Path(data_dir), Path(store_dir)
    manager = create_manager(data_dir, store_dir, subject, kappa, num_samples, patience)
    if choices is None or isinstance(choices, dict):
        choices = Config(choices).fill({
            'z_dim': list(range(12)),
            'seed': list(range(6)),
            'regress_kw.z_reg': [1., 0.1, 0.01],
        })
    configs = choices2configs(choices)
    manager.batch(configs, num_epochs, num_works)

if __name__=='__main__':
    main(**from_cli().fill({
        'data_dir': DATA_DIR, 'store_dir': STORE_DIR,
        'subject': 'marco', 'kappa': 0.1, 'num_samples': 1000,
    }))
