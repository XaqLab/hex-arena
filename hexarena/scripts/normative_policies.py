from pathlib import Path
import pickle
import numpy as np
import torch
import torch.nn as nn
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO

from irc.hmm import HiddenMarkovPolicy, HiddenMarkovNormativePolicy
from jarvis.config import from_cli, Config, choices2configs
from jarvis.utils import create_mlp_layers, tensor2array, array2tensor, get_defaults
from jarvis.manager import Manager

from .. import STORE_DIR
from .common import create_env


class BeliefModel(nn.Module):
    r"""Class for belief dynamics

    Args
    ----
    n_boxes:
        Number of food boxes.
    encoder:
        Variational encoder that outputs both mean and variance in z-space.
    decoder:
        Decoder that converts samples in z-space to unbounded belief representation.

    """

    def __init__(self,
        n_boxes: int = 3,
        z_dim: int = 8,
        encoder: dict|None = None,
        decoder: dict|None = None,
    ):
        super().__init__()
        self.n_boxes = n_boxes
        self.z_dim = z_dim
        self.encoder = nn.Sequential(*create_mlp_layers(
            2*n_boxes+1, 2*z_dim, **Config(encoder),
        ))
        self.decoder = nn.Sequential(*create_mlp_layers(
            z_dim, n_boxes, **Config(decoder),
        ))

        self.rng = np.random.default_rng()

    def encode(self, beliefs, actions):
        inputs = torch.cat([beliefs, nn.functional.one_hot(actions, self.n_boxes+1)], dim=1)
        outs = self.encoder(inputs)
        mus = outs[:, :self.z_dim]
        log_sigmas = outs[:, self.z_dim:]
        return mus, log_sigmas

    def sample(self, mus, log_sigmas, n_scale = 1.):
        ns = torch.tensor(self.rng.normal(loc=0, scale=n_scale, size=mus.shape)).to(mus)
        zs = mus+ns*torch.exp(log_sigmas)
        z_losses = (-0.5*ns.pow(2)-log_sigmas+0.5*zs.pow(2)).sum(dim=1)
        return zs, z_losses

    def decode(self, zs):
        next_beliefs = self.decoder(zs)
        return next_beliefs

    @classmethod
    def get_kl_losses(cls, preds, targets):
        ps = (torch.tanh(targets)+1)/2
        qs = (torch.tanh(preds)+1)/2
        kl_losses = (ps*(torch.log(ps)-torch.log(qs))+(1-ps)*(torch.log(1-ps)-torch.log(1-qs))).sum(dim=1)
        return kl_losses

    def forward(self, beliefs, actions, n_scale = 1.):
        mus, log_sigmas = self.encode(beliefs, actions)
        zs, z_losses = self.sample(mus, log_sigmas, n_scale)
        next_beliefs = self.decode(zs)
        return zs, z_losses, next_beliefs


class BeliefMDP(Env):

    def __init__(self, model: BeliefModel, reward: float = 10.):
        self.model = model
        self.reward = reward
        self.observation_space = Box(shape=(4,), low=-float('inf'), high=float('inf'))
        self.action_space = Discrete(4)

    def get_observation(self):
        theta = np.arctanh(self.push_cost/self.reward*2-1)
        observation = np.array([theta, *self.belief[0].numpy()])
        return observation

    def reset(self, seed: int|None = None, options: dict|None=None):
        if seed is not None:
            self.model.rng = np.random.default_rng(seed)
        self.belief = torch.tensor(self.model.rng.normal(-1.48, 0.01, (1, 3)), dtype=torch.float)
        if isinstance(options, dict) and 'push_cost' in options:
            assert 0<=options['push_cost']<=self.reward
            self.push_cost = options['push_cost']
        else:
            self.push_cost = self.model.rng.uniform(1e-5, self.reward)
        self.env = create_env(no_arena=True, env_kw={'monkey': {'push_cost': self.push_cost}})
        _, info = self.env.reset(seed=seed)
        return self.get_observation(), info

    def step(self, action: int):
        with torch.no_grad():
            _, _, self.belief = self.model(self.belief, torch.tensor([action], dtype=torch.long), n_scale=1.)
        _, reward, terminated, truncated, info = self.env.step(action)
        return self.get_observation(), reward, terminated, truncated, info


def create_manager(
    subject_id: str,
    **kwargs,
):
    r"""Creates manager for normative policy identification.

    Args
    ----
    subject_id:
        Subject name.
    kwargs:
        Keyword arguments of the manager, see `Manager` for more details.

    """
    if subject_id in ['marco', 'dylan']:
        kappas = [0.01, 0.1]
    if subject_id in ['viktor']:
        kappas = [0.01, 0.2]
    manager = Manager(STORE_DIR/'normative.policies'/subject_id, **kwargs)

    # split blocks to train/val
    with open(STORE_DIR/f'{subject_id}.mean.beliefs.pkl', 'rb') as f:
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

    # prepare datasets, transforming box beliefs from [0, 1] to (-inf, inf)
    manager.dsets = {tag: [
        (torch.atanh(2*saved['inputs'][block_id]-1), saved['actions'][block_id])
        for block_id in block_ids[tag]
    ] for tag in block_ids}
    n_actions = saved['n_actions']

    # load belief model trained by `distinct-policies.ipynb`
    manager.model = BeliefModel()
    model_pth = STORE_DIR/f'{subject_id}.belief.model.pkl'
    with open(model_pth, 'rb') as f:
        manager.model.load_state_dict(array2tensor(pickle.load(f)))

    # load trained policy trained by `bandit-policy.ipynb`
    manager.algo = PPO(
        policy='MlpPolicy', env=TimeLimit(BeliefMDP(manager.model), 1000),
        n_steps=5000, batch_size=100, learning_rate=1e-5,
        gamma=0.99, ent_coef=0.01, device='cpu',
    )
    policy_pth = STORE_DIR/f'{subject_id}.policy.pkl'
    with open(policy_pth, 'rb') as f:
        manager.algo.policy.load_state_dict(array2tensor(pickle.load(f)))

    manager.default = {
        'n_policies': 2, 'lr': 0.01, 'seed': 0,
        'learn': get_defaults(HiddenMarkovPolicy.learn, ['l2_reg', 'A_reg']) \
            |get_defaults(HiddenMarkovPolicy.train_one_batch, ['max_steps', 'batch_size']),
    }

    def setup(config: Config):
        r"""
        config:
          - n_policies: int     # number of policies
          - lr: float           # learning rate of Adam optimizer
          - seed: int           # random seed
          - learn: dict         # arguments of `HiddenMarkovNormativePolicy.learn`
            - l2_reg: float
            - A_reg: float
            - max_steps: int
            - batch_size: int

        """
        manager.config = config
        _, input_dim = manager.dsets['train'][0][0].shape
        manager.hmp = HiddenMarkovNormativePolicy(
            config.n_policies, input_dim, n_actions, manager.algo.policy,
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
    subject_id: str,
    choices: dict|Path|str|None = None,
    n_epochs: int = 500,
    n_works: int|None = None,
    **kwargs,
):
    r"""Train hidden Markov models to identify distinct normative policies.

    Current version supports only the 3-dimensional belief of Poisson boxes in
    the bandit environment. Normative policies are from parameterized optimal
    policies trained by `bandit-policy.ipynb`.

    Args
    ----
    subject_id:
        Subject name.
    kwargs:
        Keyword arguments of the manager, see `create_manager` for more details.

    """
    assert subject_id in ['marco', 'dylan', 'viktor'], (
        "Only 'marco', 'dylan' and 'viktor' are supported."
    )
    manager = create_manager(subject_id, **kwargs)
    if choices is None or isinstance(choices, dict):
        choices = Config(choices).fill({
            'n_policies': [1, 2, 3, 4, 5],
            'seed': list(range(6)),
            'learn': {
                'A_reg': [1e-4, 1e-3, 1e-2, 0.1, 1., 10.],
            },
        })
    configs = []
    for config in choices2configs(choices):
        if config.n_policies==1:
            config.update({'learn': {'A_reg': 0}})
        configs.append(config)
    manager.batch(configs, n_epochs, n_works)


if __name__=='__main__':
    main(**from_cli().fill({
        'subject_id': 'marco',
    }))
