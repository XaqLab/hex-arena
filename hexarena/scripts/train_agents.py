
from pathlib import Path
from jarvis.config import from_cli, choices2configs, Config
from irc.agent.train import create_manager

from .. import STORE_DIR


def main(
    reward: float = 10., kappa: float = 0.1, schedule: str = 'Gamma',
    choices: dict|Path|str|None = None,
    num_epochs: int = 10,
    num_works: int|None = None,
    max_errors: int = 10,
):
    r"""Trains RL agents for different reward parameters.

    Args
    ----
    subject_name:
        Subject name, can be 'marco', 'dylan' or 'viktor'.
    kappa:
        Cue reliability, higher value means more reliable color cue.
    num_samples:
        Number of state samples used in estimating new belief at each time step.
    choices:
        Job specifications, corresponding to a dictionary with keys of 'config'.
        See `setup` in `create_manager` for more details.
    num_epochs, num_works, max_errors:
        Arguments for batch processing, see `Manager.batch` for more details.
    kwargs:
        Keyword arguments of the manager, see `create_manager` for more details.

    """
    env = {'_target_': 'hexarena.env.ForagingEnv'}
    box_kw = {'reward': reward, 'kappa': kappa}
    if schedule=='Expo':
        box_kw.update({'_target_': 'hexarena.box.PoissonBox'})
        taus = [35, 21, 15]
    if schedule=='Gamma':
        box_kw.update({'_target_': 'hexarena.box.GammaBox'})
        taus = [21, 14, 7]
    env['boxes'] = [{'tau': tau, **box_kw} for tau in taus]
    if choices is None or isinstance(choices, dict):
        choices = Config(choices).fill({
            'env.monkey': {
                'push_cost': [0., 1., 2.],
                'center_cost': [0., 2., 4.],
            },
            'algo': {
                '_target_': ['sb3_contrib.RecurrentPPO', 'stable_baselines3.PPO'],
                'gamma': [0.99, 0.9, 0.6],
                'ent_coef': [0.1, 0.05, 0.01],
            },
            'learn.seed': list(range(8)),
        })
    configs = choices2configs(choices)

    manager = create_manager(env, STORE_DIR/'agents/ppo')
    manager.batch(
        configs, num_epochs, num_works, max_errors,
        pbar_kw={'unit': 'agent'},
    )


if __name__=='__main__':
    main(**from_cli())
