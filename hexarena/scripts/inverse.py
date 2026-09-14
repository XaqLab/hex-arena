import numpy as np
from jarvis.config import from_cli, Config
from jarvis.utils import tqdm
from irc.behavior import Episode

from .. import STORE_DIR
from ..utils import get_valid_blocks, load_monkey_data, align_monkey_data
from .common import create_env


def main(
    subject: str = 'marco',
    gamma: float = 1.0,
    no_arena: bool = True,
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
    if no_arena:
        block_infos = get_valid_blocks(subject, min_pos_ratio=0, min_gaze_ratio=0)
        block_ids = [
            block_id for block_id in block_infos if block_infos[block_id]['gamma']==gamma
        ]
    else:
        raise NotImplementedError
    print(f'{len(block_ids)} blocks found for {subject}')

    env = create_env(gamma=gamma, no_arena=no_arena)
    episodes = []
    for session_id, block_idx in tqdm(block_ids, unit='block'):
        block_data = load_monkey_data(subject, session_id, block_idx)
        for aligned in [False, True]:
            if aligned:
                align_monkey_data(block_data)
            data_id = '{}_{}-{:02d}_{}'.format(subject, session_id, block_idx, 'A' if aligned else 'R')
            env_data = env.convert_experiment_data(block_data)
            agt_states, _, obss, actions = env.extract_episode(env_data)
            episodes.append(Episode(
                data_id=data_id, actions=actions, obss=obss,
                agt_states=None if no_arena else agt_states,
            ))


if __name__=='__main__':
    main(**from_cli())
