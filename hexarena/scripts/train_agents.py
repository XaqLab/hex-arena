
from pathlib import Path
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
import numpy as np
from jarvis.config import from_cli, choices2configs, Config
from jarvis.manager import Manager
from jarvis.utils import tqdm, get_defaults, tensor2array, array2tensor
from irc.utils.train import exp_fit

from .. import STORE_DIR
from ..env import ForagingEnv
from ..monkey import Monkey
from ..box import BaseFoodBox
from ..utils import ProgressBarCallback
from .train_nets import fetch_best_net


TIME_LIMIT = 1200   # episode time limit in PPO
NUM_STEPS = 2000    # number of time steps per update in PPO
BATCH_SIZE = 100    # batch size in PPO
NUM_UPDATES = 10    # number of policy updates per epoch


def create_manager(
    subject: str, kappa: float, num_samples: int,
    save_interval: int = 1, patience: float = 1.,
) -> Manager:
    r"""Creates a manager for training RL agents.

    Args
    ----
    subject, kappa, num_samples:
        Subject name, cue reliability and number of state samples, see `main`
        for more details.
    save_interval, patience:
        Arguments of the manager, see `Manager` for more details.

    """
    manager = Manager(
        STORE_DIR/'agents'/subject, save_interval=save_interval, patience=patience,
    )
    manager.default = Config({'seed': 0, 'gamma': 0.99, 'ent_coef': 0.01})
    manager.default.fill(get_defaults(ForagingEnv, ['time_cost']))
    manager.default.fill(get_defaults(
        Monkey, ['push_cost', 'turn_price', 'move_price', 'look_price', 'center_cost'],
    ))
    manager.default.fill(get_defaults(BaseFoodBox, ['reward']))

    manager.default['model'], manager.model = fetch_best_net(subject, kappa, num_samples)
    manager.env = manager.model.env
    manager.model.enable_info = False
    manager.model.keep_grad = False

    def setup(config: Config) -> int:
        r"""
        config:
          - seed: int           # seed for belief nets
          - time_cost: float    # time cost
          - push_cost: float    # push cost
          - turn_price: float   # turn price, 1/deg
          - move_price: float   # move price, 1/(1^2)
          - look_price: float   # look price, 1/deg
          - center_cost: float  # center cost
          - reward: float       # box reward
          - gamma: float        # reward decay
          - ent_coef: float     # entropy coefficient

        """
        manager.env.time_cost = config.time_cost
        manager.env.monkey.set_param([
            config.push_cost, config.turn_price, config.move_price,
            config.look_price, config.center_cost,
        ])
        for box in manager.env.boxes:
            box.reward = config.reward
        manager.algo = PPO(
            env=TimeLimit(manager.model, TIME_LIMIT),
            policy='MlpPolicy', device='cpu',
            n_steps=NUM_STEPS, batch_size=BATCH_SIZE,
            gamma=config.gamma, ent_coef=config.ent_coef,
        )
        return float('inf')
    def reset():
        manager.logs = []
        manager.model.reset()
    def step():
        manager.algo.policy.set_training_mode(True)
        with tqdm(total=NUM_UPDATES, unit='update', leave=False) as pbar:
            pbar_cb = ProgressBarCallback(pbar, disp_freq=NUM_STEPS)
            pbar_cb.logs = manager.logs
            if len(manager.logs)>0:
                pbar_cb.reward, pbar_cb.food = manager.logs[-1]
            manager.algo.env.envs[0].needs_reset = False
            manager.algo.learn(
                total_timesteps=NUM_UPDATES*NUM_STEPS,
                callback=pbar_cb, reset_num_timesteps=False,
            )
        manager.algo.policy.set_training_mode(False)
    def get_ckpt():
        if len(manager.logs)>=NUM_UPDATES:
            rewards, _ = np.array(manager.logs).T
            _, optimality, _ = exp_fit(
                np.arange(len(rewards), dtype=float), rewards, ascending=True,
            )
        else:
            optimality = np.nan
        return tensor2array({
            'policy': manager.algo.policy.state_dict(),
            'logs': manager.logs,
            'last_obs': manager.algo._last_obs,
            'elapsed_steps': manager.algo.env.envs[0].env._elapsed_steps,
            'env_state': manager.env.get_state(),
            'known': manager.model.known,
            'belief': manager.model.belief,
            'optimality': optimality,
        })
    def load_ckpt(ckpt) -> int:
        ckpt = array2tensor(ckpt, manager.algo.device)
        manager.algo.policy.load_state_dict(ckpt['policy'])
        manager.logs = ckpt['logs']
        manager.algo._last_obs = ckpt['last_obs']
        manager.algo.env.envs[0].env._elapsed_steps = ckpt['elapsed_steps']
        manager.env.set_state(ckpt['env_state'])
        manager.model.known = ckpt['known']
        manager.model.belief = array2tensor(ckpt['belief'])
        return len(manager.logs)//NUM_UPDATES

    manager.setup = setup
    manager.reset = reset
    manager.step = step
    manager.get_ckpt = get_ckpt
    manager.load_ckpt = load_ckpt
    return manager


def main(
    subject: str, kappa: float, num_samples: int,
    choices: dict|Path|str|None = None,
    num_epochs: int = 20,
    num_works: int|None = None,
    max_errors: int = 10,
    **kwargs,
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
        Keyword arguments of the manager, see `Manager` for more details.

    """
    manager = create_manager(subject, kappa, num_samples, **kwargs)
    if choices is None or isinstance(choices, dict):
        choices = Config(choices).fill({
            'seed': list(range(12)),
            'push_cost': [0., 1., 2.],
            'reward': [5., 10.],
            'turn_price': [0.005],
            'move_price': [0., 1.],
            'center_cost': [0., 2., 4.],
            'gamma': [0.99, 0.9],
            'ent_coef': [0.1, 0.05],
        })
    configs = choices2configs(choices)
    manager.batch(
        configs, num_epochs, num_works, max_errors,
        pbar_kw={'unit': 'agent'},
    )


if __name__=='__main__':
    main(**from_cli().fill({
        'subject': 'marco',
        'kappa': 0.1,
        'num_samples': 1000,
    }))
