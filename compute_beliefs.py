import os, torch
from pathlib import Path
import numpy as np
from jarvis.config import from_cli
from jarvis.manager import Manager
from irc.model import SamplingBeliefModel

from hexarena.env import SimilarBoxForagingEnv
from hexarena.utils import get_valid_blocks, load_monkey_data, align_monkey_data


DATA_DIR = Path(__file__).parent/'data'
STORE_DIR = Path(__file__).parent/'store'

def create_manager(
    data_path: Path, env: SimilarBoxForagingEnv, model: SamplingBeliefModel,
    store_dir: Path, save_interval: int, patience: float,
) -> Manager:
    workspace = {}
    def setup(config: dict):
        session_id, block_idx = config['session_id'], config['block_idx']
        block_data = load_monkey_data(data_path, session_id, block_idx)
        block_data = align_monkey_data(block_data)
        env_data = env.convert_experiment_data(block_data)
        observations, actions, _ = env.extract_observation_action_reward(env_data)
        workspace['observations'] = observations
        workspace['actions'] = actions
        return len(actions)
    def reset():
        observations = workspace['observations']
        known, belief, info = model.init_belief(observations[0])
        workspace.update({
            'knowns': [known], 'beliefs': [belief], 'infos': [info],
        })
    def step():
        t = len(workspace['knowns'])-1
        known, belief, info = model.update_belief(
            workspace['knowns'][t], workspace['beliefs'][t],
            workspace['actions'][t], workspace['observations'][t+1],
        )
        workspace['knowns'].append(known)
        workspace['beliefs'].append(belief)
        workspace['infos'].append(info)
    def get_ckpt():
        return {
            'knowns': workspace['knowns'], 'infos': workspace['infos'],
            'beliefs': [b.data.cpu().numpy() for b in workspace['beliefs']],
        }
    def load_ckpt(ckpt):
        workspace.update({
            'knowns': ckpt['knowns'], 'infos': ckpt['infos'],
            'beliefs': [torch.tensor(b, device=model.device) for b in ckpt['beliefs']],
        })

    manager = Manager(
        store_dir=store_dir, save_interval=save_interval, patience=patience,
    )
    manager.setup = setup
    manager.reset = reset
    manager.step = step
    manager.get_ckpt = get_ckpt
    manager.load_ckpt = load_ckpt
    return manager

def main(
    data_dir: Path|str,
    store_dir: Path|str,
    subject: str,
    num_samples: int,
    num_works: int|None = None,
    save_interval: int = 10,
    patience: float = 12.,
):
    data_path = Path(data_dir)/f'data_{subject}.mat'
    assert os.path.exists(data_path), f"{data_path} not found"
    assert subject in ['marco', 'viktor'], (
        "Only 'marco' and 'viktor' can be processed now."
    )
    block_infos = get_valid_blocks(data_path, min_pos_ratio=0.5, min_gaze_ratio=0.1)

    block_ids = []
    for session_id in block_infos:
        for block_idx in block_infos[session_id]:
            block_data = load_monkey_data(data_path, session_id, block_idx)
            to_process = False
            if subject=='marco':
                if np.all(block_data['kappas']==0.01) and set(block_data['taus'])==set([15., 21., 35.]):
                    to_process = True
            if subject=='viktor':
                if set(block_data['taus'])==set([7., 14., 21.]):
                    to_process = True
            if to_process:
                block_ids.append((session_id, block_idx))
    print(f'{len(block_ids)} valid blocks found.')

    if subject=='marco':
        env = SimilarBoxForagingEnv(
            box={
                '_target_': 'hexarena.box.StationaryBox',
                'num_patches': 1, 'num_levels': 10, 'num_grades': 8,
            },
            boxes=[{'tau': tau} for tau in [35, 21, 15]],
        )
        model = SamplingBeliefModel(env,
            s_idcs=[[0], [1], [2, 3], [4, 5], [6, 7]],
        )
    if subject=='viktor':
        env = SimilarBoxForagingEnv(
            box={
                '_target_': 'hexarena.box.GammaLinearBox', 'num_patches': 1, 'num_levels': 40,
            },
            boxes=[{'tau': tau} for tau in [21, 14, 7]],
        )
        phi = {
            'embedder._target_': 'hexarena.box.LinearBoxStateEmbedder',
            'mlp_features': [6],
        }
        model = SamplingBeliefModel(
            env, p_s={'phis': [phi]*3},
        )
    model.use_sample = True
    model.num_samples = num_samples

    store_dir = Path(store_dir)/'beliefs'/subject
    manager = create_manager(
        data_path, env, model, store_dir, save_interval, patience,
    )
    configs = [{
        'session_id': session_id,
        'block_idx': block_idx,
        'num_samples': num_samples,
    } for session_id, block_idx in block_ids]
    np.random.default_rng().shuffle(configs)
    manager.batch(
        configs, num_works=num_works, pbar_kw={'unit': 'block', 'leave': True},
    )

if __name__=='__main__':
    main(**from_cli().fill({
        'data_dir': DATA_DIR,
        'store_dir': STORE_DIR,
        'subject': 'marco',
        'num_samples': 1000,
    }))
