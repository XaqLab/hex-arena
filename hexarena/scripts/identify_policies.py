from pathlib import Path
import pickle
import torch
import numpy as np
from jarvis.config import from_cli, choices2configs, Config
from jarvis.manager import Manager
from jarvis.utils import tqdm, tensor2array, array2tensor
from irc.hmp import HiddenMarkovPolicy

from .. import STORE_DIR


def create_manager(
    subject: str, data_pth: Path|None = None, kappas: list[float]|None = None,
    save_interval: int = 1, patience: float = 1.,
) -> Manager:
    r"""Creates a manager for policy identification.

    Args
    ----
    subject, data_pth, kappas:
        Subject name, the data file path and stimulus reliability parameters.
        See `main` for more details.
    save_interval, patience:
        Arguments of the manager, see `Manager` for more details.

    """
    if data_pth is None:
        data_pth = STORE_DIR/f'{subject}.mean.beliefs.pkl'
    if kappas is None:
        kappas = [0.01]
    manager = Manager(
        STORE_DIR/'policies'/subject, save_interval=save_interval, patience=patience,
    )
    with open(data_pth, 'rb') as f:
        saved = array2tensor(pickle.load(f))
    manager.block_ids = []
    for block_id, block_info in saved['block_infos'].items():
        if block_info['kappa'] in kappas:
            manager.block_ids.append(block_id)
    manager.block_ids = sorted(manager.block_ids)
    n_blocks = len(manager.block_ids)
    manager.inputs = [saved['inputs'][block_id] for block_id in manager.block_ids]
    input_dim = np.unique([v.shape[1] for v in manager.inputs]).item()
    manager.actions = [saved['actions'][block_id] for block_id in manager.block_ids]
    n_actions = saved['n_actions']

    manager.default = {
        'data_pth': data_pth.name,
        'block_ids': manager.block_ids,
        'seed': 0, 'split': 0.9, 'n_policies': 2,
        'policy': {'num_features': [], 'nonlinearity': 'Softplus'},
        'reg_coefs': {
            'alpha_A': 100., 'off_ratio': 0.1, 'l2_reg': 1e-3, 'ent_reg': 1e-4,
        },
    }

    def setup(config: Config):
        r"""
        config:
          - data_pth: str
          - block_ids: list[tuple[str, int]]
          - seed: int           # random seed for HMP initialization
          - split: float        # portion of training time steps in each block
          - n_policies: int     # number of policies
          - policy: dict        # config of policy network
            - num_features: list[int]   # hidden layer sizes
            - nonlinearity: str         # nonlinearity
          - reg_coefs: dict     # regularization coefficients, see `m_step`
            - alpha_A: float
            - off_ratio: float
            - l2_reg: float
            - ent_reg: float

        """
        assert config.data_pth==data_pth.name, (
            "Data file inconsistent with current manager"
        )
        assert config.block_ids==manager.block_ids, (
            "Block IDs inconsistent with current manager"
        )
        manager.config = config
        manager.hmp = HiddenMarkovPolicy(
            config.n_policies, input_dim, n_actions, config.policy,
        )
        manager.hmp.reset(config.seed)
        return float('inf') # no limit on EM iterations
    def training_tensors():
        split = manager.config.split
        inputs, actions = [], []
        for i in range(n_blocks):
            t_train = int(len(manager.actions[i])*split)
            inputs.append(manager.inputs[i][:t_train])
            actions.append(manager.actions[i][:t_train])
        return inputs, actions
    def reset():
        manager.hmp.reset(manager.config.seed)
        manager.Ps, manager.As, manager.losses = [], [], []
        manager.gammas = [[] for _ in range(n_blocks)]
        manager.log_Zs = []
    def step():
        # training set only
        inputs, actions = training_tensors()
        hmp, reg_coefs = manager.hmp, manager.config.reg_coefs
        _, _, log_gammas, log_xis, log_Z = hmp.e_step(inputs, actions)
        for i in range(n_blocks):
            manager.gammas[i].append(log_gammas[i].data.exp().cpu())
        manager.log_Zs.append(log_Z)
        stats = hmp.m_step(
            inputs, actions, log_gammas, log_xis,  **reg_coefs,
        )
        manager.Ps.append(hmp.log_P.data.exp())
        manager.As.append(hmp.log_A.data.exp())
        manager.losses.append(stats['losses_val'][stats['best_epoch']])
        return 'LL {:.3f}'.format(-manager.losses[-1][0])
    def get_ckpt():
        return tensor2array({
            'state_dict': manager.hmp.state_dict(),
            'Ps': manager.Ps, 'As': manager.As, 'losses': manager.losses,
            'gammas': manager.gammas, 'log_Zs': manager.log_Zs,
        })
    def load_ckpt(ckpt):
        ckpt = array2tensor(ckpt, manager.hmp.device)
        manager.hmp.load_state_dict(ckpt['state_dict'])
        for key in ['Ps', 'As', 'losses', 'gammas', 'log_Zs']:
            setattr(manager, key, ckpt[key])
        return len(manager.losses)
    manager.setup = setup
    manager.reset = reset
    manager.step = step
    manager.get_ckpt = get_ckpt
    manager.load_ckpt = load_ckpt
    return manager


def main(
    data_pth: Path|str, kappas: list[float]|float,
    choices: dict|Path|str|None = None,
    num_epochs: int = 50,
    num_works: int|None = None,
    **kwargs,
):
    r"""Train hidden Markov models to identify distinct policies.

    Current version supports only the 3-dimensional belief of Poisson boxes in
    the bandit environment.

    Args
    ----
    data_pth:
        The data file containing meta information, policy inputs and agent
        actions of different blocks. The file name should be formatted as
        '[subject].*.pkl'. It contains the following dicts,
        - 'block_infos': containing meta information such as `kappa`, `gamma`.
        - 'inputs': containing tensors of shape `(n_steps, input_dim)` for
        policy inputs.
        - 'actions': containing tensors of shape `(n_steps,)` for agent actions.
    kappas:
        Cue reliability parameters to specify blocks to process. Higher value
        means more reliable color cue.
    choices:
        Job specifications, corresponding to a dictionary with keys of 'config'.
        See `setup` in `create_manager` for more details.
    num_epochs, num_works:
        Arguments for batch processing, see `Manager.batch` for more details.
        `num_epochs` is the number of EM iterations in HMP learning.
    kwargs:
        Keyword arguments of the manager, see `Manager` for more details.

    """
    data_pth = Path(data_pth)
    if not data_pth.is_absolute():
        data_pth = STORE_DIR/data_pth
    subject = data_pth.name.split('.')[0]
    assert subject in ['marco', 'dylan', 'viktor'], (
        "Only 'marco', 'dylan' and 'viktor' are supported."
    )
    if isinstance(kappas, float):
        kappas = [kappas]
    manager = create_manager(subject, data_pth, kappas, **kwargs)
    if choices is None or isinstance(choices, dict):
        choices = Config(choices).fill({
            'seed': list(range(6)),
            'n_policies': [1, 2, 3, 4, 5, 6],
            'policy.num_features': [[], [16]],
            'reg_coefs': {
                'alpha_A': [1., 1e2, 1e4],
                'off_ratio': [0.9, 0.3, 0.1],
                'l2_reg': [1e-2, 1e-3, 1e-4],
                'ent_reg': [1e-2, 1e-3, 1e-4]
            },
        })
    configs = choices2configs(choices)
    manager.batch(
        configs, num_epochs, num_works, process_kw={'pbar_kw.unit': 'iter'},
    )


if __name__=='__main__':
    main(**from_cli().fill({
        'data_pth': 'marco.mean.beliefs.pkl',
        'kappas': [0.01, 0.1],
    }))
