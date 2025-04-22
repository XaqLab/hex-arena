from pathlib import Path
import torch
from jarvis.config import from_cli, choices2configs, Config
from jarvis.manager import Manager
from jarvis.utils import tqdm, tensor2array, array2tensor
from irc.hmp import HiddenMarkovPolicy

from .. import STORE_DIR
from ..alias import Tensor
from .common import get_block_ids, create_env_and_model
from .compute_beliefs import fetch_beliefs
from .compress_beliefs import fetch_best_vae


def prepare_data(
    subject: str, kappa: float, num_samples: int, num_macros: int,
    block_ids: list[tuple[str, int]],
) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
    r"""Collect data from blocks.

    Known states and beliefs along with merged actions are collected for given
    blocks.

    Args
    ----
    subject, kappa, num_samples:
        Subject name, cue reliability and number of state samples, see `main`
        for more details.
    num_macros:
        Number of macro actions for merging actions.
    block_ids:
        Block IDs to collect data for.

    Returns
    -------
    knowns, beliefs: [(num_steps, .)]
        Known states and beliefs of each block.
    actions: [(num_steps,)]
        Macro actions of each block.

    """
    env, _ = create_env_and_model(subject, kappa)
    knowns, beliefs, actions = [], [], []
    for session_id, block_idx in tqdm(block_ids, desc='Collect data', unit='block', leave=False):
        _, _actions, _knowns, _beliefs = fetch_beliefs(
            subject, kappa, num_samples, session_id, block_idx,
        )
        _actions = env.monkey.merge_actions(_actions, num_macros)
        knowns.append(torch.tensor(_knowns, dtype=torch.float)[:-1])
        beliefs.append(_beliefs[:-1])
        actions.append(torch.tensor(_actions, dtype=torch.long))
    return knowns, beliefs, actions

def e_step(
    hmp: HiddenMarkovPolicy,
    knowns: list[Tensor], beliefs: list[Tensor], actions: list[Tensor],
) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
    r"""Performs one expectation step.

    Args
    ----
    hmp:
        A hidden Markov policy.
    knowns: [(num_steps, .)]
        Known states of each block.
    beliefs: [(num_steps, .)]
        Model-based beliefs of each block.
    actions: [(num_steps,)]
        Actions taken by the agent in each block.

    Returns
    -------
    log_gammas: [(num_steps, num_policies)]
        Posterior of policy variable at each time step.
    log_xis: [(num_steps, num_policies, num_policies)]
        Pseudocounts of transitions at each time step.
    log_Zs: [scalar]
        Data likelihood of each block.

    """
    log_gammas, log_xis, log_Zs = [], [], []
    for _knowns, _beliefs, _actions in zip(knowns, beliefs, actions):
        _, _, _log_gammas, _log_xis, _log_Z = hmp.e_step(_knowns, _beliefs, _actions)
        log_gammas.append(_log_gammas)
        log_xis.append(_log_xis)
        log_Zs.append(_log_Z)
    return log_gammas, log_xis, log_Zs

def m_step(
    hmp: HiddenMarkovPolicy,
    knowns: list[Tensor], beliefs: list[Tensor], actions: list[Tensor],
    log_gammas: list[Tensor], log_xis: list[Tensor],
    alpha_A: float, off_ratio: float, l2_reg: float, ent_reg: float,
) -> dict:
    r"""Performs one maximization step.

    Args
    ----
    hmp:
        A hidden Markov policy.
    knowns, beliefs, actions:
        Sequential data, see `e_step` for more details.
    log_gammas, log_xis:
        Output of the expectation step, see `e_step` for more details.
    alpha_A, off_ratio:
        Dirichlet priors for estimating the transition matrix.
    l2_reg, ent_reg:
        Regularization coefficients used in policy learning, see
        `HiddenMarkovPolicy.m_step` for more details.

    Returns
    -------
    stats:
        Statistics for training policy networks, see `HiddenMarkovPolicy.m_step`
        for more details.

    """
    hmp.update_pi(log_gammas)
    hmp.update_A(log_xis, alpha_A, off_ratio)
    stats = hmp.m_step(
        torch.cat(knowns), torch.cat(beliefs), torch.cat(actions), torch.cat(log_gammas),
        l2_reg=l2_reg, ent_reg=ent_reg,
    )
    return stats

def create_manager(
    subject: str, kappa: float, num_samples: int,
    z_dim: int = 5, num_macros: int = 10,
    read_only: bool = False,
    save_interval: int = 1, patience: float = 1.,
) -> Manager:
    r"""Creates a manager for policy identification.

    Args
    ----
    subject, kappa, num_samples:
        Subject name, cue reliability and number of state samples, see `main`
        for more details.
    z_dim:
        Dimension of compressed beliefs, used for fetching pretrained belief VAE
        model.
    num_marcos:
        Number of macro actions, see `prepare_data` for more details.
    save_interval, patience:
        Arguments of the manager, see `Manager` for more details.

    """
    manager = Manager(
        STORE_DIR/'policies'/subject, save_interval=save_interval, patience=patience,
    )
    manager.env, manager.model = create_env_and_model(subject, kappa)
    manager.block_ids = get_block_ids(subject, kappa)
    manager.num_blocks = len(manager.block_ids)
    bv_config, manager.belief_vae = fetch_best_vae(subject, kappa, num_samples, z_dim)
    if not read_only:
        manager.knowns, manager.beliefs, manager.actions = prepare_data(
            subject, kappa, num_samples, num_macros, manager.block_ids,
        )
    manager.default = {
        'subject': subject, 'kappa': kappa, 'num_samples': num_samples,
        'z_dim': z_dim, 'num_macros': num_macros, 'num_policies': 2,
        'seed': 0, 'split': 0.9, 'bv_config': bv_config,
        'policy': {'num_features': [], 'nonlinearity': 'Softplus'},
        'reg_coefs': {
            'alpha_A': 100., 'off_ratio': 0.1, 'l2_reg': 1e-3, 'ent_reg': 1e-4,
        },
    }

    def setup(config: Config):
        r"""
        config:
          - subject: str        # subject
          - kappa: float        # stimulus reliability
          - num_samples: int    # number of states in belief update
          - z_dim: int          # compressed belief dimension
          - num_macros: int     # number of macro actions
          - seed: int           # random seed for HMP initialization
          - split: float        # portion of training time steps in each block
          - bv_config: dict     # config of belief_vae
          - num_policies: int   # number of policies
          - policy: dict        # config of policy network
            - num_features: list[int]   # hidden layer sizes
            - nonlinearity: str         # nonlinearity
          - reg_coefs: dict     # regularization coefficients, see `m_step`
            - alpha_A: float
            - off_ratio: float
            - l2_reg: float
            - ent_reg: float

        """
        assert config.subject==subject
        assert config.kappa==kappa
        assert config.num_samples==num_samples
        assert config.z_dim==z_dim
        assert config.num_macros==num_macros
        manager.split = config.split
        manager.reg_coefs = config.reg_coefs
        manager.hmp = HiddenMarkovPolicy(
            manager.model.p_s, z_dim, num_macros,
            num_policies=config.num_policies, policy=config.policy,
            ebd_k=manager.model.ebd_k, ebd_b=manager.model.ebd_b,
        )
        manager.hmp.reset(config.seed)
        manager.hmp.belief_vae = manager.belief_vae
        manager.pis, manager.As, manager.lls = [], [], []
        manager.gammas, manager.log_Zs = [[] for _ in range(manager.num_blocks)], []
        return float('inf') # no limit on EM iterations
    def training_tensors():
        split = manager.split
        knowns, beliefs, actions = [], [], []
        for i in range(manager.num_blocks):
            t_train = int(len(manager.actions[i])*split)
            knowns.append(manager.knowns[i][:t_train])
            beliefs.append(manager.beliefs[i][:t_train])
            actions.append(manager.actions[i][:t_train])
        return knowns, beliefs, actions
    def reset():
        knowns, beliefs, actions = training_tensors()
        manager.log_gammas, manager.log_xis, _ = e_step(manager.hmp, knowns, beliefs, actions)
    def step():
        knowns, beliefs, actions = training_tensors()
        hmp, reg_coefs = manager.hmp, manager.reg_coefs
        stats = m_step(
            manager.hmp, knowns, beliefs, actions,
            manager.log_gammas, manager.log_xis, **reg_coefs,
        )
        manager.pis.append(hmp.log_pi.exp())
        manager.As.append(hmp.log_A.exp())
        manager.lls.append(-stats['losses_val'][-1][0])

        manager.log_gammas, manager.log_xis, log_Zs = e_step(hmp, knowns, beliefs, actions)
        for i in range(manager.num_blocks):
            manager.gammas[i].append(manager.log_gammas[i].exp())
        manager.log_Zs.append(log_Zs)
    def get_ckpt():
        hmp = manager.hmp
        knowns, beliefs, actions = manager.knowns, manager.beliefs, manager.actions
        log_gammas, _, _ = e_step(hmp, knowns, beliefs, actions)
        log_gammas_test, lls_train, lls_test = [], [], []
        for i in range(manager.num_blocks):
            t_train = int(len(actions[i])*manager.split)
            log_gammas_test.append(log_gammas[i][t_train:])
            gammas = log_gammas[i].exp()
            with torch.no_grad():
                inputs = hmp.policy_inputs(knowns[i].to(hmp.device), beliefs[i].to(hmp.device))
                _, logps = hmp.action_probs(inputs)
                lls = (hmp.emission_probs(logps, actions[i])*gammas).sum(dim=1).cpu()
                lls_train.append(lls[:t_train])
                lls_test.append(lls[t_train:])
        ll_train = torch.cat(lls_train).mean().item()
        ll_test = torch.cat(lls_test).mean().item()
        return tensor2array({
            'workspace': {k: getattr(manager, k) for k in [
                'pis', 'As', 'lls', 'gammas', 'log_Zs', 'log_gammas', 'log_xis',
            ]},
            'log_pi': hmp.log_pi, 'log_A': hmp.log_A,
            'policies': [policy.state_dict() for policy in hmp.policies],
            'log_gammas': log_gammas_test, 'll_train': ll_train, 'll_test': ll_test,
        })
    def load_ckpt(ckpt):
        ckpt = array2tensor(ckpt, manager.model.device)
        for k in ckpt['workspace']:
            setattr(manager, k, ckpt['workspace'][k])
        hmp = manager.hmp
        hmp.log_pi = ckpt['log_pi']
        hmp.log_A = ckpt['log_A']
        for i, policy in enumerate(hmp.policies):
            policy.load_state_dict(ckpt['policies'][i])
        return len(manager.lls)
    manager.setup = setup
    manager.reset = reset
    manager.step = step
    manager.get_ckpt = get_ckpt
    manager.load_ckpt = load_ckpt
    return manager


def main(
    subject: str, kappa: float, num_samples: int,
    choices: dict|Path|str|None = None,
    num_epochs: int = 50,
    num_works: int|None = None,
    **kwargs,
):
    r"""Train hidden Markov models to identify distinct policies.

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
    num_epochs, num_works:
        Arguments for batch processing, see `Manager.batch` for more details.
        `num_epochs` is the number of EM iterations in HMP learning.
    kwargs:
        Keyword arguments of the manager, see `Manager` for more details.

    """
    manager = create_manager(subject, kappa, num_samples, **kwargs)
    if choices is None or isinstance(choices, dict):
        choices = Config(choices).fill({
            'seed': list(range(6)),
            'num_policies': [1, 2, 3, 4, 5, 6],
            'policy.num_features': [
                [], [64], [16, 32],
            ],
            'reg_coefs': {
                'alpha_A': [1., 1e2, 1e4],
                'off_ratio': [0.9, 0.3, 0.1],
            },
        })
    configs = choices2configs(choices)
    manager.batch(
        configs, num_epochs, num_works, process_kw={'pbar_kw.unit': 'iter'},
    )


if __name__=='__main__':
    main(**from_cli().fill({
        'subject': 'marco',
        'kappa': 0.1,
        'num_samples': 1000,
    }))
