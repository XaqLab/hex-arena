from pathlib import Path
import pickle
import numpy as np
import torch
from jarvis.config import from_cli, choices2configs, Config
from jarvis.manager import Manager
from jarvis.utils import tensor2array, array2tensor, get_defaults
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
        'seed': 0,
        'learn': get_defaults(HiddenMarkovPolicy.learn, ['l2_reg', 'A_reg']) \
            |get_defaults(HiddenMarkovPolicy.train_one_batch, ['max_steps', 'batch_size']),
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
          - seed: int       # random seed for HMP initialization
          - learn: dict     # arguments of `HiddenMarkovPolicy.learn`
            - l2_reg: float
            - A_reg: float
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
                    manager.dsets[tag].append((torch.atanh(inputs*2-1), actions))
                else:
                    manager.dsets[tag].append((torch.zeros((len(inputs), 0)), actions))
        _, input_dim = manager.dsets['train'][0][0].shape
        manager.hmp = HiddenMarkovPolicy(
            config.n_policies, input_dim, n_actions, config.policy,
        )
        manager.optimizer = manager.hmp.default_optimizer(config.lr)
        return float('inf')
    def reset():
        manager.hmp.reset(seed=manager.config.seed)
        manager.losses, manager.last_state = None, None
        manager.min_loss, manager.best_epoch, manager.best_state = float('inf'), None, None
        manager.log_gammas, manager.log_xis = None, None
    def step():
        losses, log_gammas, log_xis, states, loss, _, manager.optimizer = manager.hmp.learn(
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
            manager.log_gammas = log_gammas['val'][-1]
            manager.log_xis = log_xis['val'][-1]
        return 'LL {:.3f}'.format(losses['val'][-1][0])
    attr_names = [
        'losses', 'last_state', 'min_loss', 'best_epoch', 'best_state', 'log_gammas', 'log_xis',
    ]
    def get_ckpt():
        ckpt = {k: getattr(manager, k) for k in attr_names}
        ckpt['optimizer'] = manager.optimizer.state_dict()
        return tensor2array(ckpt)
    def load_ckpt(ckpt):
        ckpt = array2tensor(ckpt)
        for key in attr_names:
            if key in ckpt:
                setattr(manager, key, ckpt[key])
        manager.hmp.load_state_dict(manager.last_state)
        manager.optimizer.load_state_dict(ckpt['optimizer'])
        return len(manager.losses['val'])
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
    n_epochs, n_works:
        Arguments for batch processing, see `Manager.batch` for more details.
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
            'n_policies': [1, 2, 3, 4, 5],
            'policy.num_features': [[16]],
            'seed': list(range(6)),
            'learn': {
                'A_reg': [1e-4, 1e-3, 1e-2, 0.1, 1., 10.],
            },
        })
    configs = []
    for config in choices2configs(choices):
        if config.n_policies==1:
            config.update({'learn': {'A_reg': 0}})
        if not config.belief_aware:
            config.policy = {}
        configs.append(config)
    manager.batch(configs, n_epochs, n_works)


if __name__=='__main__':
    main(**from_cli().fill({
        'data_pth': 'marco.mean.beliefs.pkl',
    }))
