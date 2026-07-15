from jarvis.config import from_cli, Config
from jarvis.utils import tqdm
from irc.manager import BayesianBeliefManager

from .. import STORE_DIR
from ..utils import get_valid_blocks, load_monkey_data, align_monkey_data
from .common import create_env


def main(
    subject: str = 'marco',
    gamma: float = 1.0,
    no_arena: bool = True,
    bmdp_kw: dict|None = None,
):
    r"""Computes beliefs for multiple blocks.

    Args
    ----
    subject:
        Subject name.
    gamma:
        The shape parameter of Gamma distribution for food schedule.
    no_arena:
        If ``True``, `BanditForagingEnv` will be used and the observation only
        includes action outcome. If ``False``, `ArenaForagingEnv` will be used,
        with both spatial information and color cues.
    bmdp_kw:
        Keyword arguments of `BaseBeliefMDP`.

    """
    bmdp_kw = Config(bmdp_kw).fill({
        'train_kw.n_epochs': 40,
        'estimate_kw.sga_kw.n_epochs': 60,
    }).asdict()
    if no_arena:
        block_infos = get_valid_blocks(subject, min_pos_ratio=0, min_gaze_ratio=0)
        block_ids = [
            block_id for block_id in block_infos if block_infos[block_id]['gamma']==gamma
        ]
    else:
        raise NotImplementedError
    print(f'{len(block_ids)} blocks found for {subject}')

    env = create_env(gamma=gamma, no_arena=no_arena)
    ckpt_dir = STORE_DIR/'beliefs'/subject
    ckpt_dir /= 'bandit' if no_arena else 'arena'
    ckpt_dir /= 'Poisson' if gamma==1 else 'Gamma'
    manager = BayesianBeliefManager(env, ckpt_dir)

    configs = []
    for session_id, block_idx in tqdm(block_ids, desc='Load data', unit='block'):
        data_id = f'{session_id}-{block_idx}'
        if manager.data_ids.get_key(data_id) is None:
            block_data = load_monkey_data(subject, session_id, block_idx)
            align_monkey_data(block_data)
            env_data = env.convert_experiment_data(block_data)
            agt_states, _, obss, actions = env.extract_episode(env_data)
            manager.add_episode(data_id, actions, obss, agt_states)
        configs.append({
            'param': env.get_param(), 'data_id': data_id,
        })
    manager.batch(configs, pbar_kw={'desc': 'Compute beliefs'})


if __name__=='__main__':
    main(**from_cli())
