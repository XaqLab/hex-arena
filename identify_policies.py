import os, pickle, torch
from pathlib import Path
import numpy as np
from jarvis.config import from_cli, Config
from jarvis.utils import tqdm
from jarvis.manager import Manager
from irc.model import SamplingBeliefModel
from irc.hmp import HiddenMarkovPolicy

from hexarena.env import SimilarBoxForagingEnv
from hexarena.utils import get_valid_blocks, load_monkey_data, align_monkey_data

from compute_beliefs import prepare_blocks, create_default_model, create_manager


DATA_DIR = Path(__file__).parent/'data'
STORE_DIR = Path(__file__).parent/'store'

def fetch_beliefs(manager, session_id, block_idx, num_samples=1000):
    ckpt = manager.process({
        'session_id': session_id, 'block_idx': block_idx, 'num_samples': num_samples,
    })
    knowns = np.array(ckpt['knowns'])
    beliefs = np.array(ckpt['beliefs'])
    return knowns, beliefs

def collect_data(block_ids, data_path, env, manager, num_samples, num_macros):
    knowns, beliefs, actions = [], [], []
    for session_id, block_idx in tqdm(block_ids, desc='Collect data', unit='block', leave=False):
        block_data = load_monkey_data(data_path, session_id, block_idx)
        block_data = align_monkey_data(block_data)
        env_data = env.convert_experiment_data(block_data)
        _, _actions, _ = env.extract_observation_action_reward(env_data)
        _actions = env.monkey.merge_actions(_actions, num_macros)
        actions.append(torch.tensor(_actions, dtype=torch.long))

        T = len(_actions)
        _knowns, _beliefs = fetch_beliefs(manager, session_id, block_idx, num_samples)
        knowns.append(torch.tensor(_knowns, dtype=torch.float)[:T])
        beliefs.append(torch.tensor(_beliefs, dtype=torch.float)[:T])
    return knowns, beliefs, actions

def init_hmp(model, z_dim, num_macros, num_policies, store_dir):
    hmp = HiddenMarkovPolicy(
        model.p_s, z_dim, num_macros, num_policies=num_policies,
        ebd_k=model.ebd_k, ebd_b=model.ebd_b,
    )
    vae_path = store_dir/'belief_vaes/belief.vae_[Dz{:02d}].pkl'.format(z_dim)
    with open(vae_path, 'rb') as f:
        hmp.belief_vae.load_state_dict(pickle.load(f)['state_dict'])
    return hmp

def e_step(hmp, knowns, beliefs, actions):
    log_gammas, log_xis, log_Zs = [], [], []
    for _knowns, _beliefs, _actions in zip(knowns, beliefs, actions):
        _, _, _log_gammas, _log_xis, _log_Z = hmp.e_step(_knowns, _beliefs, _actions)
        log_gammas.append(_log_gammas)
        log_xis.append(_log_xis)
        log_Zs.append(_log_Z)
    return log_gammas, log_xis, log_Zs

def m_step(hmp, knowns, beliefs, actions, log_gammas, log_xis, **kwargs):
    hmp.update_pi(log_gammas)
    hmp.update_A(log_xis)
    stats = hmp.m_step(
        torch.cat(knowns), torch.cat(beliefs), torch.cat(actions),
        torch.cat(log_gammas), **kwargs
    )
    return stats

def main(
    data_dir: Path|str,
    store_dir: Path|str,
    subject: str,
    num_samples: int = 1000,
    z_dim: int = 3,
    num_policies: int = 3,
    num_macros: int = 10,
    num_iters: int = 50,
    m_step_kw: dict|None = None
):
    m_step_kw = Config(m_step_kw).fill({
        'l2_reg': 1e-3, 'ent_reg': 1e-4, 'kl_reg': 1e-3,
    })

    data_path = Path(data_dir)/f'data_{subject}.mat'
    block_ids = prepare_blocks(data_path, subject)
    num_blocks = len(block_ids)
    env, model = create_default_model(subject)

    store_dir = Path(store_dir)
    manager = create_manager(data_path, env, model, store_dir/'beliefs'/subject)
    knowns, beliefs, actions = collect_data(
        block_ids, data_path, env, manager, num_samples, num_macros,
    )
    num_steps = sum([len(a) for a in actions])
    hmp = init_hmp(model, z_dim, num_macros, num_policies, store_dir)

    pis, As, lls = [], [], []
    gammas, log_Zs = [[] for _ in range(num_blocks)], []
    log_gammas, log_xis, _ = e_step(hmp, knowns, beliefs, actions)
    with tqdm(total=num_iters, unit='iter') as pbar:
        for _ in range(num_iters):
            stats = m_step(hmp, knowns, beliefs, actions, log_gammas, log_xis, **m_step_kw)
            pis.append(hmp.log_pi.exp())
            As.append(hmp.log_A.exp())
            lls.append(stats['lls'][-1])

            log_gammas, log_xis, _log_Zs = e_step(hmp, knowns, beliefs, actions)
            for i in range(num_blocks):
                gammas[i].append(log_gammas[i].exp())
            log_Zs.append(_log_Zs)

            pbar.set_description('log(Z): {:.3f}'.format(np.sum(_log_Zs)/num_steps))
            pbar.update()
    pis = torch.stack(pis).numpy()
    As = torch.stack(As).numpy()
    lls = np.array(lls)
    for i in range(num_blocks):
        gammas[i] = torch.stack(gammas[i]).numpy()
    log_Zs = np.array(log_Zs)

    save_path = store_dir/'policies_{}_[Dz{:02d}][Np{:d}][Na{:d}].pkl'.format(
        subject, z_dim, num_policies, num_macros,
    )
    with open(save_path, 'wb') as f:
        pickle.dump({
            'num_samples': num_samples, 'z_dim': z_dim,
            'num_policies': num_policies,
            'num_macros': num_macros,
            'num_iters': num_iters,
            'm_step_kw': m_step_kw,
            'pis': pis, 'As': As, 'lls': lls,
            'gammas': gammas, 'log_Zs': log_Zs,
            'policies': [
                {k: v.clone() for k, v in policy.state_dict().items()}
                for policy in hmp.policies
            ],
        }, f)


if __name__=='__main__':
    main(**from_cli().fill({
        'data_dir': DATA_DIR,
        'store_dir': STORE_DIR,
        'subject': 'marco',
    }))
