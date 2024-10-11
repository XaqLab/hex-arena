import pickle, torch
from pathlib import Path
import numpy as np
from jarvis.config import from_cli, choices2configs, Config
from jarvis.utils import tqdm, tensor2array, array2tensor
from jarvis.manager import Manager
from irc.model import SamplingBeliefModel
from irc.hmp import HiddenMarkovPolicy

from hexarena.alias import Tensor, Array

from compute_beliefs import prepare_blocks, create_model, fetch_beliefs


DATA_DIR = Path(__file__).parent/'data'
STORE_DIR = Path(__file__).parent/'store'

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
    z_dim: int, num_macros: int, num_policies: int,
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
        ebd_k=model.ebd_k, ebd_b=model.ebd_b,
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
    alpha_A: float, l2_reg: float, ent_reg: float, kl_reg: float,
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
    alpha_A:
        Diagonal prior for estimating the transition matrix. `alpha_A>0` prefers
        stable dynamics, while `alpha_A<0` prefers switching between policies.
    l2_reg, ent_reg, kl_reg:
        Regularization coefficients used in policy learning.

    Returns
    -------
    stats:
        Statistics for training policy networks, see `HiddenMarkovPolicy.m_step`
        for more details.

    """
    hmp.update_pi(log_gammas)
    hmp.update_A(log_xis, alpha_A)
    stats = hmp.m_step(
        torch.cat(knowns), torch.cat(beliefs), torch.cat(actions), torch.cat(log_gammas),
        l2_reg=l2_reg, ent_reg=ent_reg, kl_reg=kl_reg,
    )
    return stats

def create_manager(
    data_dir: Path, store_dir: Path, subject: str,
    block_ids: list[tuple[str, int]],
    num_samples: int = 1000, num_macros: int = 10,
    patience: float = 12.,
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
    _, model = create_model(subject)
    num_blocks = len(block_ids)
    knowns, beliefs, actions = collect_data(
        data_dir, store_dir, subject, block_ids, num_samples, num_macros,
    )
    ws = { # workspace
        'knowns': knowns, 'beliefs': beliefs, 'actions': actions,
    }
    def setup(config: Config):
        r"""
        config:
          - seed: int
          - split: float        # portion of training length of each block
          - num_samples: int    # number of samples used in belief update
          - z_dim: int          # compressed belief dimension
          - num_policies: int   # number of candidate policies
          - num_macros: int     # number of macro actions
          - reg_coefs:          # regularization coefficients, see `HiddenMarkovPolicy.m_step`
            - alpha_A: float
            - l2_reg: float
            - ent_reg: float
            - kl_reg: float

        """
        assert config.num_samples==num_samples
        assert config.num_macros==num_macros
        hmp = init_hmp(
            model, config.z_dim, config.num_macros, config.num_policies, store_dir, config.seed,
        )
        pis, As, lls = [], [], []
        gammas, log_Zs = [[] for _ in range(num_blocks)], []
        ws.update({
            'config': config, 'hmp': hmp, 'pis': pis, 'As': As, 'lls': lls,
            'gammas': gammas, 'log_Zs': log_Zs,
        })
        return float('inf') # no limit on EM iterations
    def get_tensors(train: bool = True):
        split = ws['config'].split
        knowns, beliefs, actions = [], [], []
        for i in range(num_blocks):
            t_train = int(len(ws['actions'][i])*split)
            if train:
                knowns.append(ws['knowns'][i][:t_train])
                beliefs.append(ws['beliefs'][i][:t_train])
                actions.append(ws['actions'][i][:t_train])
            else:
                knowns.append(ws['knowns'][i][t_train:])
                beliefs.append(ws['beliefs'][i][t_train:])
                actions.append(ws['actions'][i][t_train:])
        return knowns, beliefs, actions
    def reset():
        knowns, beliefs, actions = get_tensors()
        log_gammas, log_xis, _ = e_step(ws['hmp'], knowns, beliefs, actions)
        ws.update({'log_gammas': log_gammas, 'log_xis': log_xis})
    def step():
        knowns, beliefs, actions = get_tensors()
        hmp, reg_coefs = ws['hmp'], ws['config'].reg_coefs
        stats = m_step(
            hmp, knowns, beliefs, actions,
            ws['log_gammas'], ws['log_xis'], **reg_coefs,
        )
        ws['pis'].append(hmp.log_pi.exp())
        ws['As'].append(hmp.log_A.exp())
        ws['lls'].append(-stats['losses_val'][-1][0])

        ws['log_gammas'], ws['log_xis'], log_Zs = e_step(hmp, knowns, beliefs, actions)
        for i in range(num_blocks):
            ws['gammas'][i].append(ws['log_gammas'][i].exp())
        ws['log_Zs'].append(log_Zs)
    def get_ckpt():
        hmp = ws['hmp']
        return tensor2array({
            'workspace': {k: ws[k] for k in [
                'pis', 'As', 'lls', 'gammas', 'log_Zs', 'log_gammas', 'log_xis',
            ]},
            'log_pi': hmp.log_pi, 'log_A': hmp.log_A,
            'policies': [policy.state_dict() for policy in hmp.policies],
        })
    def load_ckpt(ckpt):
        ckpt = array2tensor(ckpt, model.device)
        ws.update(ckpt['workspace'])
        hmp = ws['hmp']
        hmp.log_pi = ckpt['log_pi']
        hmp.log_A = ckpt['log_A']
        for i, policy in enumerate(hmp.policies):
            policy.load_state_dict(ckpt['policies'][i])
        return len(ws['lls'])

    manager = Manager(store_dir=Path(store_dir)/'policies'/subject, patience=patience)
    manager.setup = setup
    manager.reset = reset
    manager.step = step
    manager.get_ckpt = get_ckpt
    manager.load_ckpt = load_ckpt
    manager.ws = ws
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
    manager.load_ckpt(manager.ckpts[manager.configs.get_key(config)])
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
    elif key=='ll_val': # log likelihoods on validation blocks
        knowns = manager.ws['knowns']
        beliefs = manager.ws['beliefs']
        actions = manager.ws['actions']
        log_gammas, _, _ = e_step(hmp, knowns, beliefs, actions)
        lls = []
        for i in range(len(actions)):
            t_train = int(len(actions[i])*config.split)
            gammas = log_gammas[i][t_train:].exp()
            with torch.no_grad():
                inputs = hmp.policy_inputs(knowns[i][t_train:], beliefs[i][t_train:])
                _, logps = hmp.action_probs(inputs)
                lls.append((hmp.emission_probs(logps, actions[i][t_train:])*gammas).sum(dim=1))
        lls = torch.cat(lls)
        return lls.mean().item()
    else:
        raise RuntimeError(f"Key '{key}' not recognized")

def main(
    data_dir: Path|str,
    store_dir: Path|str,
    subject: str,
    num_samples: int = 1000,
    num_macros: int = 10,
    choices: Path|str = 'hmp_spec_A10.yaml',
    num_iters: int = 50,
    num_works: int|None = None,
    patience: float = 12.,
):
    data_dir, store_dir = Path(data_dir), Path(store_dir)
    block_ids = prepare_blocks(data_dir, subject)
    manager = create_manager(
        data_dir, store_dir, subject, block_ids, num_samples, num_macros, patience,
    )
    configs = choices2configs(store_dir/choices, num_works)
    manager.batch(
        configs, num_iters, num_works,
        process_kw={'pbar_kw.unit': 'iter'},
    )

''
if __name__=='__main__':
    main(**from_cli().fill({
        'data_dir': DATA_DIR,
        'store_dir': STORE_DIR,
        'subject': 'marco',
    }))
