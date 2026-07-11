import random
from jarvis.config import from_cli, Config
from jarvis.utils import tqdm, safe_write
import irc

from .. import STORE_DIR
from ..utils import get_valid_blocks, load_monkey_data, align_monkey_data
from .common import create_env


def main(
    subject: str = 'marco',
    no_arena: bool = True,
    bmdp_kw: dict|None = None,
):
    r"""Computes beliefs for multiple blocks.

    Args
    ----
    subject:
        Subject name.
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
        block_ids = sorted(block_infos.keys())
    else:
        raise NotImplementedError
    print(f'{len(block_ids)} blocks found for {subject}')
    random.shuffle(block_ids)
    for session_id, block_idx in tqdm(block_ids, unit='block'):
        ckpt_pth = STORE_DIR/'beliefs'/subject/'{}.beliefs_{}[block{}].pkl'.format(
            'bandit' if no_arena else 'arena', session_id, block_idx,
        )
        if not ckpt_pth.exists():
            safe_write({'bmdp_kw': bmdp_kw}, ckpt_pth)
        gamma = block_infos[(session_id, block_idx)]['gamma']
        kappa = block_infos[(session_id, block_idx)]['kappa']

        env = create_env(gamma, kappa, no_arena=no_arena)
        block_data = load_monkey_data(subject, session_id, block_idx)
        align_monkey_data(block_data)
        env_data = env.convert_experiment_data(block_data)
        agt_states, _, obss, actions = env.extract_episode(env_data)

        irc.bayesian(
            env, actions, obss, agt_states,
            ckpt_pth=ckpt_pth, bmdp_kw=bmdp_kw, allow_interrupt=False,
            pbar_kw={'desc': f'{session_id}-{block_idx}', 'leave': False},
        )


if __name__=='__main__':
    main(**from_cli())
