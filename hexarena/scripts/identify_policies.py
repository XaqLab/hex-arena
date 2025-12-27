from pathlib import Path
import pickle
import numpy as np
import torch
from jarvis.config import from_cli, choices2configs, Config
from jarvis.manager import Manager
from jarvis.utils import tensor2array, array2tensor
from irc.hmm import HiddenMarkovPolicy

from .. import STORE_DIR


def create_manager(
    subject: str, data_pth: Path|None = None, kappas: list[float]|None = None,
    device: str = 'cuda', save_interval: int = 10, patience: float = 1.,
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
        if subject in ['marco', 'dylan']:
            kappas = [0.01, 0.1]
        if subject in ['viktor']:
            kappas = [0.01, 0.2]
    device = device if torch.cuda.is_available() else 'cpu'
    manager = Manager(
        STORE_DIR/'policies'/subject, save_interval=save_interval, patience=patience,
    )

    # split blocks to train/val
    with open(data_pth, 'rb') as f:
        saved = array2tensor(pickle.load(f))
    block_ids = []
    for block_id, block_info in saved['block_infos'].items():
        if block_info['kappa'] in kappas:
            block_ids.append(block_id)
    block_ids = sorted(block_ids)
    n_blocks = len(block_ids)
    idxs = np.random.default_rng(0).permutation(n_blocks)
    _n = int(n_blocks*0.1)
    idxs = {'train': idxs[_n:], 'val': idxs[:_n]}
    block_ids = {tag: [block_ids[i] for i in idxs[tag]] for tag in idxs}
    manager.block_ids = block_ids

    # prepare datasets
    manager.raw_dsets = {tag: [
        (saved['inputs'][block_id], saved['actions'][block_id])
        for block_id in block_ids[tag]
    ] for tag in block_ids}
    n_actions = saved['n_actions']

    manager.default = {
        'data_pth': data_pth.name, 'block_ids': block_ids,
        'belief_aware': True, 'n_policies': 2,
        'policy': {'num_features': [], 'nonlinearity': 'Softplus'},
        'lr': 0.01,
        'reset': {'seed': 0, 'alpha_A': 5.},
        'learn': {
            'max_steps': 800, 'batch_size': 32,
            'l2_reg': 1., 'jsd_reg': 0., 'switch_reg': 0.,
        },
    }

    def setup(config: Config):
        r"""
        config:
          - data_pth: str
          - block_ids: dict[str, list[tuple[str, int]]]
          - belief_aware: bool  # whether to use belief as inputs
          - n_policies: int     # number of policies
          - policy: dict    # config of policy network
            - num_features: list[int]   # hidden layer sizes
            - nonlinearity: str         # nonlinearity
          - lr: float       # learning rate of Adam optimizer
          - reset: dict     # arguments of `HiddenMarkovPolicy.reset`
            - seed: int         # random seed for HMP initialization
            - alpha_A: float    # diagonal prior of transition matrix
          - learn: dict     # arguments of `HiddenMarkovPolicy.learn`
            - l2_reg: float
            - jsd_reg: float
            - switch_reg: float
            - max_steps: int
            - batch_size: int

        """
        assert config.data_pth==data_pth.name, (
            "Data file inconsistent with current manager"
        )
        assert config.block_ids==manager.block_ids, (
            "Block IDs inconsistent with current manager"
        )
        manager.config = config
        manager.dsets = {}
        for tag in manager.raw_dsets:
            manager.dsets[tag] = []
            for inputs, actions in manager.raw_dsets[tag]:
                if config.belief_aware:
                    manager.dsets[tag].append((torch.atanh(inputs), actions))
                else:
                    manager.dsets[tag].append((torch.zeros((len(inputs), 0)), actions))
        _, input_dim = manager.dsets['train'][0][0].shape
        manager.hmp = HiddenMarkovPolicy(
            config.n_policies, input_dim, n_actions, config.policy,
        )
        manager.optimizer = manager.hmp.default_optimizer(config.lr)
        return float('inf')
    def reset():
        manager.hmp.reset(**manager.config.reset)
        manager.losses, manager.last_state = None, None
        manager.min_loss, manager.best_epoch, manager.best_state = float('inf'), None, None
    def step():
        losses, states, loss, _, manager.optimizer = manager.hmp.learn(
            manager.dsets, manager.optimizer, n_epochs=1,
            pbar_kw={'disable': True}, **manager.config.learn,
        )
        if manager.losses is None:
            manager.losses = {tag: [losses[tag]] for tag in losses}
        else:
            for tag in manager.losses:
                manager.losses[tag].append(losses[tag])
        manager.last_state = states[-1]
        if loss<manager.min_loss:
            manager.min_loss = loss
            manager.best_epoch = len(manager.losses['val'])-1
            manager.best_state = states[-1]
        return 'LL {:.3f}'.format(losses['val'][-1][0])
    def get_ckpt():
        ckpt = {
            k: getattr(manager, k) for k in [
                'losses', 'last_state', 'min_loss', 'best_epoch', 'best_state',
            ]
        }
        ckpt['optimizer'] = manager.optimizer.state_dict()
        return tensor2array(ckpt)
    def load_ckpt(ckpt):
        ckpt = array2tensor(ckpt)
        manager.hmp.load_state_dict(ckpt['last_state'])
        for key in ['losses', 'min_loss', 'best_epoch', 'best_state']:
            setattr(manager, key, ckpt[key])
        manager.optimizer.load_state_dict(ckpt['optimizer'])
        return len(manager.losses)
    manager.setup = setup
    manager.reset = reset
    manager.step = step
    manager.get_ckpt = get_ckpt
    manager.load_ckpt = load_ckpt
    return manager


def main(
    data_pth: Path|str, kappas: list[float]|float|None = None,
    choices: dict|Path|str|None = None,
    n_epochs: int = 500,
    n_works: int|None = None,
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
            'belief_aware': [False, True],
            'n_policies': [2, 3, 4],
            'policy.num_features': [[], [16]],
            'reset': {
                'seed': list(range(6)),
                'alpha_A': [1., 3., 5.],
            },
            'learn': {
                'jsd_reg': [0., 10., 20., 50.],
                'switch_reg': [0., 10., 20., 50.],
            }
        })
    configs = choices2configs(choices)
    manager.batch(configs, n_epochs, n_works)


if __name__=='__main__':
    main(**from_cli().fill({
        'data_pth': 'marco.mean.beliefs.pkl',
    }))
