import os, yaml, pickle, torch
from pathlib import Path
import numpy as np
from jarvis.config import from_cli
from jarvis.manager import Manager
from irc.model import SamplingBeliefModel

from hexarena.env import SimilarBoxForagingEnv
from hexarena.utils import get_valid_blocks, load_monkey_data, align_monkey_data


DATA_DIR = Path(__file__).parent/'data'
STORE_DIR = Path(__file__).parent/'store'

def compute_beliefs(
    session_id: str, block_idx: int,
    data_path: Path, env: SimilarBoxForagingEnv, model: SamplingBeliefModel,
    ckpt_dir: Path|None = None,
):
    if ckpt_dir is None:
        ckpt_path = None
    else:
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = ckpt_dir/f'{session_id}_{block_idx}.pkl'
    block_data = load_monkey_data(data_path, session_id, block_idx)
    block_data = align_monkey_data(block_data)
    env_data = env.convert_experiment_data(block_data)
    observations, actions, _ = env.extract_observation_action_reward(env_data)
    knowns, beliefs, _ = model.compute_beliefs(observations, actions, ckpt_path, pbar_kw={'leave': False})
    return knowns, beliefs.data.cpu().numpy()

def main(
    data_dir: Path|str,
    store_dir: Path|str,
    subject: str,
    num_samples: int,
    num_works: int|None = None,
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
    np.random.default_rng().shuffle(block_ids)
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
    os.makedirs(store_dir, exist_ok=True)
    meta = {
        'subject': subject, 'env_spec': env.spec, 'num_samples': num_samples,
    }
    meta_path = store_dir/'meta.yaml'
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            assert yaml.safe_load(f)==meta
    else:
        with open(meta_path, 'w') as f:
            yaml.dump(meta, f)

    manager = Manager(store_dir, patience=6.)
    manager.main = lambda config: compute_beliefs(**config,
        data_path=data_path, env=env, model=model, ckpt_dir=store_dir/'cache',
    )
    manager.batch(
        [{'session_id': session_id, 'block_idx': block_idx} for session_id, block_idx in block_ids],
        total=num_works, pbar_kw={'unit': 'block', 'leave': True},
    )

if __name__=='__main__':
    main(**from_cli().fill({
        'data_dir': DATA_DIR,
        'store_dir': STORE_DIR,
        'subject': 'marco',
        'num_samples': 1000,
    }))
