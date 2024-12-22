from pathlib import Path
from jarvis.config import from_cli, choices2configs, Config
from jarvis.manager import Manager
from jarvis.utils import tqdm, tensor2array, array2tensor

from hexarena import DATA_DIR, STORE_DIR

from compute_beliefs import prepare_blocks, create_model, fetch_beliefs


def create_manager(
    data_dir: Path, store_dir: Path, subject: str, kappa: float,
    num_samples: int = 1000, patience: float = 12.,
) -> Manager:
    r"""Creates a manager to train networks for belief dynamics.

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
        store_dir=store_dir/'belief_nets'/subject, patience=patience,
    )
    block_ids = prepare_blocks(data_dir, subject, kappa, verbose=False)
    manager.observations, manager.actions, _, manager.beliefs = zip(*[
        fetch_beliefs(
            data_dir, store_dir, subject, session_id, block_idx, num_samples,
        ) for session_id, block_idx in tqdm(block_ids, unit='block', leave=False)
    ])
    manager.default = {
        'seed': 0, 'model_kw': {},
        'update_kw': {'z_reg': 1e-4, 'num_epochs': 300},
        'init_kw': {'z_reg': 1e-4, 'num_epochs': 200},
    }

    def setup(config: Config):
        r"""
        config:
          - subject: str        # subject name
          - kappa: float        # stimulus reliability
          - num_samples: int    # number of samples used in belief computation
        """
        manager.seed = config.seed
        manager.model_kw = config.model_kw
        manager.train_kw = {
            'update_kw': config.update_kw, 'init_kw': config.init_kw,
        }
        return 1 # one epoch for manager
    def reset():
        _, manager.model = create_model(subject, kappa, model_kw=manager.model_kw)
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

def main(
    data_dir: Path|str, store_dir: Path|str,
    subject: str, kappa: float, num_samples: int,
    choices: dict|Path|str|None = None,
    patience: float = 12., num_works: int|None = None,
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
    data_dir, store_dir = Path(data_dir), Path(store_dir)
    manager = create_manager(
        data_dir, store_dir, subject, kappa, num_samples, patience,
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
        'data_dir': DATA_DIR, 'store_dir': STORE_DIR,
        'subject': 'marco', 'kappa': 0.1, 'num_samples': 1000,
    }))
