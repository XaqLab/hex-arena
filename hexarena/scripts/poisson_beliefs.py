from irc.model import SamplingBeliefModel
from jarvis.utils import tqdm
from jarvis.config import from_cli

from .. import STORE_DIR
from ..utils import get_valid_blocks, load_monkey_data, align_monkey_data
from .common import create_env


def main(
    subject: str = 'marco',
):
    r"""Computes beliefs for Poisson boxes.

    Args
    ----
    subject:
        Monkey name, can be 'marco', 'dylan' or 'viktor'.

    """
    block_infos = get_valid_blocks(subject, min_pos_ratio=0, min_gaze_ratio=0, min_push=10)
    session_ids = [key for key in block_infos if block_infos[key]['gamma']==1]

    # Bandit foraging problem
    env = create_env(no_arena=True)
    for session_id, block_idx in tqdm(session_ids, desc=f'Analyzing data of {subject}', unit='session'):
        block_data = load_monkey_data(subject, session_id, block_idx)
        align_monkey_data(block_data)
        env_data = env.convert_experiment_data(block_data)
        agt_states, _, obss, actions = env.extract_episode(env_data)

        model = SamplingBeliefModel(env)
        model.use_sample = True
        model.estimate_kw.update({'sga_kw.pbar_kw': {'disable': False, 'leave': False}})

        ckpt_pth = STORE_DIR/subject/f'poisson.beliefs/{session_id}-{block_idx}.pkl'
        model.compute_beliefs(
            agt_states, obss, actions, ckpt_pth, pbar_kw={'leave': False},
        )


if __name__=='__main__':
    main(**from_cli())
