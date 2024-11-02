import os, pickle, torch
from pathlib import Path
from jarvis.config import from_cli, Config
from jarvis.utils import tqdm

from hexarena import DATA_DIR, STORE_DIR

from compute_beliefs import prepare_blocks, create_model, fetch_beliefs


def main(
    data_dir: Path|str,
    store_dir: Path|str,
    subject: str,
    num_samples: int,
    seeds: list[int],
    update_kw: dict|None = None,
    init_kw: dict|None = None,
):
    r"""Trains belief networks for a belief model.

    Args
    ----
    data_dir:
        Directory that contains experiment data file, e.g. 'data_marco.mat'.
    store_dir:
        Directory for storage, containing precomputed beliefs using sampling
        methods, see `compute_beliefs.py` for more details.
    subject:
        Subject name.
    num_samples:
        Number of samples used in estimating beliefs, used to identify the saved
        belief data.

    """
    data_dir = Path(data_dir)
    store_dir = Path(store_dir)
    update_kw = Config(update_kw).fill({
        'z_reg': 1e-4, 'num_epochs': 300,
    })
    init_kw = Config(init_kw).fill({
        'z_reg': 1e-4, 'num_epochs': 200,
    })

    # collect experiment data along with precomputed beliefs
    block_ids = prepare_blocks(data_dir, subject)
    observations, actions, beliefs = [], [], []
    for session_id, block_idx in tqdm(block_ids, unit='block'):
        _observations, _actions, _, _beliefs = fetch_beliefs(
            data_dir, store_dir, subject, session_id, block_idx, num_samples,
        )
        observations.append(_observations)
        actions.append(_actions)
        beliefs.append(torch.tensor(_beliefs))

    _, model = create_model(subject)
    for seed in tqdm(seeds, unit='seed'):
        save_pth = store_dir/'belief_nets/{}_[seed{:02d}].pkl'.format(subject, seed)
        meta = Config({
            'subject': subject, 'seed': seed,
            'block_ids': block_ids, 'num_samples': num_samples,
            'update_kw': update_kw, 'init_kw': init_kw,
        })

        # check if the model instance is trained
        to_skip = False
        if os.path.exists(save_pth):
            with open(save_pth, 'rb') as f:
                saved = pickle.load(f)
            to_skip = Config(saved['meta'])==meta
        if to_skip:
            continue

        # train update_net and init_net of the belief model
        model.init_net.reset(seed)
        model.update_net.reset(seed)
        stats_u, stats_i = model.train_nets(
            observations, actions, beliefs, strict_init=True,
            update_kw=update_kw, init_kw=init_kw,
        )
        os.makedirs(save_pth, exist_ok=True)
        with open(save_pth, 'wb') as f:
            pickle.dump({
                'meta': meta.asdict(), 'state_dict': model.state_dict(),
                'stats_u': stats_u, 'stats_i': stats_i,
            }, f)


if __name__=='__main__':
    main(**from_cli().fill({
        'data_dir': DATA_DIR, 'store_dir': STORE_DIR,
        'subject': 'marco', 'num_samples': 1000,
        'seeds': list(range(12)),
    }))
