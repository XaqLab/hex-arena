import pickle, torch, yaml
from pathlib import Path
import numpy as np
from jarvis.config import from_cli, choices2configs, Config
from jarvis.utils import tqdm, tensor2array, array2tensor
from jarvis.manager import Manager
from irc.model import SamplingBeliefModel
from irc.hmp import HiddenMarkovPolicy

from hexarena import DATA_DIR, STORE_DIR
from hexarena.alias import Tensor

from compute_beliefs import prepare_blocks, create_model, fetch_beliefs


def collect_data(
    data_dir: Path, store_dir: Path, subject: str, block_ids: list[tuple[str, int]],
    num_samples: int, num_macros: int,
) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
    r"""Collect data from blocks.

    Args
    ----
    data_dir, store_dir:
        Directory of data and storage respectively.
    subject_dir:
        Subject name.
    num_samples:
        Number of samples in each step of belief update, see `fetch_beliefs`
        for more details.
    num_macros:
        Number of macro actions for action merging.

    Returns
    -------
    knowns, beliefs: [(num_steps, .)]
        Known states and beliefs of each block.
    actions: [(num_steps,)]
        Macro actions of each block.

    """
    env, _ = create_model(subject)
    knowns, beliefs, actions = [], [], []
    for session_id, block_idx in tqdm(block_ids, desc='Collect data', unit='block', leave=False):
        _, _actions, _knowns, _beliefs = fetch_beliefs(
            data_dir, store_dir, subject, session_id, block_idx, num_samples,
        )
        _actions = env.monkey.merge_actions(_actions, num_macros)
        actions.append(torch.tensor(_actions, dtype=torch.long))

        T = len(_actions)
        knowns.append(torch.tensor(_knowns, dtype=torch.float)[:T])
        beliefs.append(torch.tensor(_beliefs, dtype=torch.float)[:T])
    return knowns, beliefs, actions

def init_hmp(
    model: SamplingBeliefModel,
    z_dim: int, num_macros: int, num_policies: int, policy: dict,
    store_dir: Path, seed: int|None = None,
) -> HiddenMarkovPolicy:
    r"""Initializes a hidden Markov policy object.

    The pretrained belief encoder is loaded, expected to be found in the folder
    `store_dir/'belief_vaes'`.

    Args
    ---
    model:
        Belief model of the foraging environment.
    z_dim, num_macros, num_policies:
        Arguments of the hidden Markov policy.
    store_dir:
        Directory of storage, where pretrained belief encoder should be found.

    Returns
    -------
    hmp:
        The HiddenMarkovPolicy for policy identification.

    """
    hmp = HiddenMarkovPolicy(
        model.p_s, z_dim, num_macros, num_policies=num_policies,
        ebd_k=model.ebd_k, ebd_b=model.ebd_b, policy=policy,
    )
    if seed is not None:
        hmp.reset(seed)
    vae_path = store_dir/'belief_vaes/belief.vae_[Dz{:02d}].pkl'.format(z_dim)
    with open(vae_path, 'rb') as f:
        hmp.belief_vae.load_state_dict(pickle.load(f)['state_dict'])
    return hmp

def e_step(
    hmp: HiddenMarkovPolicy,
    knowns: list[Tensor], beliefs: list[Tensor], actions: list[Tensor],
) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
    r"""Performs one expectation step.

    Args
    ----
    hmp:
        Hidden Markov policy object, see `init_hmp` for more details.
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
        Hidden Markov policy object, see `init_hmp` for more details.
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
    data_dir: Path, store_dir: Path, subject: str,
    block_ids: list[tuple[str, int]],
    num_samples: int = 1000, num_macros: int = 10,
    patience: float = 4.,
) -> Manager:
    r"""Creates a manger for hidden Markov policy learning.

    Args
    ----
    data_dir, store_dir:
        Directory of data and storage respectively.
    subject_dir:
        Subject name.
    block_ids:
        The block ID `(session_id, block_idx)` of all blocks to be processed.
    num_samples, num_macros:
        See `collect_data` for more details.

    Returns
    -------
    manager:
        A manager that performs hidden Markov policy learning with different
        hyperparameters.

    """
    manager = Manager(store_dir=Path(store_dir)/'policies'/subject, patience=patience)
    manager.env, manager.model = create_model(subject)
    manager.num_blocks = len(block_ids)
    manager.knowns, manager.beliefs, manager.actions = collect_data(
        data_dir, store_dir, subject, block_ids, num_samples, num_macros,
    )
    manager.default = {
        'seed': 0, 'block_ids': block_ids,
        'split': 0.9, # portion of training length of each block
        'num_samples': num_samples, # number of samples in belief update
        'z_dim': 3, # compressed belief dimension
        'num_macros': num_macros, # number of macro actionds
        'policy': { # MLP parameters of policy network
            'num_features': [], # hidden layer sizes
            'nonlinearity': 'Softplus',
        },
        'reg_coefs': {  # regularization coefficients, see `HiddenMarkovPolicy.m_step`
            'alpha_A': 100.,
            'off_ratio': 0.1,
            'l2_reg': 1e-3,
            'ent_reg': 1e-4,
        },
    }

    def setup(config: Config):
        r"""
        config:
          - seed: int
          - split: float        # portion of training length of each block
          - num_samples: int    # number of samples used in belief update
          - z_dim: int          # compressed belief dimension
          - num_policies: int   # number of candidate policies
          - policy: dict        # MLP parameters of policy network
          - num_macros: int     # number of macro actions
          - reg_coefs:          # regularization coefficients, see `HiddenMarkovPolicy.m_step`
            - alpha_A: float
            - off_ratio: float
            - l2_reg: float
            - ent_reg: float

        """
        assert config.num_samples==num_samples
        assert config.num_macros==num_macros
        manager.config = config
        manager.hmp = init_hmp(
            manager.model, config.z_dim, config.num_macros, config.num_policies,
            config.policy, store_dir, config.seed,
        )
        manager.pis, manager.As, manager.lls = [], [], []
        manager.gammas, manager.log_Zs = [[] for _ in range(manager.num_blocks)], []
        return float('inf') # no limit on EM iterations
    def get_tensors():
        split = manager.config.split
        knowns, beliefs, actions = [], [], []
        for i in range(manager.num_blocks):
            t_train = int(len(manager.actions[i])*split)
            knowns.append(manager.knowns[i][:t_train])
            beliefs.append(manager.beliefs[i][:t_train])
            actions.append(manager.actions[i][:t_train])
        return knowns, beliefs, actions
    def reset():
        knowns, beliefs, actions = get_tensors()
        manager.log_gammas, manager.log_xis, _ = e_step(manager.hmp, knowns, beliefs, actions)
    def step():
        knowns, beliefs, actions = get_tensors()
        hmp, reg_coefs = manager.hmp, manager.config.reg_coefs
        stats = m_step(
            hmp, knowns, beliefs, actions,
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
        for i in range(len(actions)):
            t_train = int(len(actions[i])*manager.config.split)
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

def fetch_results(
    manager: Manager, config: Config, key: str,
):
    r"""Fetch computed results.

    Args
    ----
    manager:
        The Manager object returned by `create_manager`, containing the full
        progress of hidden Markov policy learning.
    config:
        The computation config, see `create_manager.setup` for more details.
    key:
        Key for the field of interest in result, see comments for more details.

    """
    manager.setup(config)
    ckpt = manager.ckpts[manager.configs.get_key(config)]
    manager.load_ckpt(ckpt)
    hmp: HiddenMarkovPolicy = manager.ws['hmp']
    if key=='log_pi': # learned initial distribution
        return hmp.log_pi
    elif key=='log_A': # learned transition matrix
        return hmp.log_A
    elif key=='policies': # state dict of learned policies
        return hmp.policies
    elif key=='pis': # history of initial distribution
        return torch.stack(manager.ws['pis']) # (num_iters, num_policies)
    elif key=='As': # history of transition matrix
        return torch.stack(manager.ws['As']) # (num_iters, num_policies, num_policies)
    elif key=='lls': # history of log likelihoods on validation set
        return np.array(manager.ws['lls']) # (num_iters,)
    elif key=='gammas': # history of posteriors
        gammas = manager.ws['gammas']
        return [torch.stack(gammas[i]) for i in range(len(gammas))] # (num_iters, num_steps, num_policies)
    elif key=='log_Zs': # history of data likelihood
        total_steps = sum([len(a) for a in manager.ws['actions']])
        log_Zs = np.array(manager.ws['log_Zs']).sum(axis=1)/total_steps
        return log_Zs # (num_iters,)
    elif key=='ent': # entropy of marginal policy distribution
        gammas = manager.ws['gammas']
        probs = torch.cat([gammas[i][-1] for i in range(len(gammas))]).mean(dim=0)
        ent = -(probs*torch.log(probs)).sum()
        return ent
    elif key in ['ll_train', 'll_test']: # log likelihood on training/testing segments
        return ckpt[key]
    else:
        raise RuntimeError(f"Key '{key}' not recognized")

def main(
    data_dir: Path|str,
    store_dir: Path|str,
    subject: str,
    choices: Path|str = 'hmp_spec_A10.yaml',
    num_iters: int = 50,
    num_works: int|None = None,
    patience: float = 12.,
):
    data_dir, store_dir = Path(data_dir), Path(store_dir)
    block_ids = prepare_blocks(data_dir, subject)
    with open(store_dir/choices, 'r') as f:
        choices = yaml.safe_load(f)
    assert len(choices['num_samples'])==1
    num_samples = choices['num_samples'][0]
    assert len(choices['num_macros'])==1
    num_macros = choices['num_macros'][0]

    manager = create_manager(
        data_dir, store_dir, subject, block_ids, num_samples, num_macros, patience,
    )
    configs = choices2configs(choices)
    print('{} configs generated.'.format(len(configs)))
    manager.batch(
        configs, num_iters, num_works, process_kw={'pbar_kw.unit': 'iter'},
    )


if __name__=='__main__':
    main(**from_cli().fill({
        'data_dir': DATA_DIR,
        'store_dir': STORE_DIR,
        'subject': 'marco',
    }))
