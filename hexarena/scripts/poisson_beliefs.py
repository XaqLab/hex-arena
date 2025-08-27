import numpy as np
from irc.model import SamplingBeliefModel
from jarvis.utils import tensor2array, array2tensor
from jarvis.config import from_cli, Config
from jarvis.manager import Manager

from .. import STORE_DIR
from ..utils import get_valid_blocks, load_monkey_data, align_monkey_data
from .common import create_env


def create_manager(
    subject: str,
    save_interval: int = 10, patience: float = 2.,
) -> Manager:
    r"""Creates manager to compute beliefs of each session.

    Args
    ----
    subject:
        Monkey name, see `main` for more details.
    save_interval, patience:
        Arguments of the manager, see `Manager` for more details.

    """
    manager = Manager(
        STORE_DIR/'beliefs'/subject, save_interval=save_interval, patience=patience,
    )
    manager.default = {
        'tau_in_state': False, 'n_samples': 2000,
    }

    def setup(config: Config) -> int:
        r"""
        config:
          - session_id: str
          - block_idx: int
          - tau_in_state: bool
          - n_samples: int

        """
        manager.config = config
        manager.env = create_env(no_arena=True, tau_in_state=config.tau_in_state)
        block_data = load_monkey_data(subject, config.session_id, config.block_idx)
        if not config.tau_in_state:
            align_monkey_data(block_data)
        env_data = manager.env.convert_experiment_data(block_data)
        manager.agt_states, _, manager.obss, manager.actions = \
            manager.env.extract_episode(env_data)

        if config.tau_in_state:
            cliques = {
                'env': [
                    [('box_0', 'food'), ('box_0', 'tau', (0,))],
                    [('box_1', 'food'), ('box_1', 'tau', (0,))],
                    [('box_2', 'food'), ('box_2', 'tau', (0,))],
                ],
            }
            p_s = {
                'phis': [{'embedder': {'_target_': 'irc.dist.embedder.ConcatEmbedder'}}]*3,
            }
        else:
            cliques, p_s = None, None
        manager.model = SamplingBeliefModel(
            manager.env, p_s=p_s, cliques=cliques, n_samples=config.n_samples,
        )
        manager.model.update_spaces()
        manager.model.use_sample = True
        manager.model.estimate_kw.update({
            'sga_kw.pbar_kw': {'disable': False, 'leave': False},
        })
        return len(manager.actions)
    def reset():
        belief, info = manager.model.init_belief(
            manager.agt_states[0], manager.obss[0],
        )
        manager.beliefs = [belief]
        manager.infos = [info]
    def step():
        t = len(manager.beliefs)
        belief, info = manager.model.update_belief(
            manager.agt_states[t-1], manager.beliefs[-1], manager.actions[t-1],
            manager.agt_states[t], manager.obss[t],
        )
        manager.beliefs.append(belief)
        manager.infos.append(info)
    def get_ckpt():
        return tensor2array({
            'beliefs': manager.beliefs,
            'infos': manager.infos,
        })
    def load_ckpt(ckpt):
        ckpt = array2tensor(ckpt)
        manager.beliefs = ckpt['beliefs']
        manager.infos = ckpt['infos']
        return len(manager.beliefs)-1
    def pbar_desc(config):
        return f'{config.session_id}-B{config.block_idx}'

    manager.setup = setup
    manager.reset = reset
    manager.step = step
    manager.get_ckpt = get_ckpt
    manager.load_ckpt = load_ckpt
    manager.pbar_desc = pbar_desc
    return manager


def main(
    subject: str = 'marco',
    tau_in_state: bool = False,
    n_samples: int = 2000,
    num_works: int|None = None,
    **kwargs,
):
    r"""Computes beliefs for Poisson boxes.

    Args
    ----
    subject:
        Monkey name, can be 'marco', 'dylan' or 'viktor'.
    tau_in_state:
        Whether to include box quality 'tau' in the box state.
    n_samples:
        Number of samples used in belief estimation.
    kwargs:
        Keyword arguments of the manager, see `Manager` for more details.

    """
    manager = create_manager(subject, **kwargs)

    block_infos = get_valid_blocks(subject, min_pos_ratio=0, min_gaze_ratio=0, min_push=10)
    block_ids = [key for key in block_infos if block_infos[key]['gamma']==1]
    configs = [Config({
        'session_id': session_id, 'block_idx': block_idx,
        'tau_in_state': tau_in_state, 'n_samples': n_samples,
    }) for session_id, block_idx in block_ids]
    np.random.default_rng().shuffle(configs)
    manager.batch(
        configs, num_works=num_works,
        pbar_kw={'unit': 'block', 'leave': True},
        process_kw={'pbar_kw': {'unit': 'steps'}},
    )


if __name__=='__main__':
    main(**from_cli())
