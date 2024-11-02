import yaml, pickle, inspect
from pathlib import Path
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO

from jarvis.config import from_cli, Config, choices2configs
from jarvis.manager import Manager
from jarvis.utils import tqdm, tensor2array, array2tensor

from hexarena import STORE_DIR
from hexarena.monkey import Monkey
from hexarena.box import BaseFoodBox
from hexarena.env import ForagingEnv
from hexarena.utils import ProgressBarCallback

from compute_beliefs import create_model


def create_manager(
    store_dir: Path, subject: str, patience: float = 2.,
) -> Manager:
    r"""Creates manager for training RL agents.

    Args
    ----
    store_dir:
        Directory for storage, see `main` for more details.
    subject:
        Subject name.

    """
    manager = Manager(
        store_dir=store_dir/f'agents/{subject}', patience=patience,
    )
    manager.env, manager.model = create_model(subject)
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
        save_pth = store_dir/'belief_nets/{}_[seed{:02d}].pkl'.format(subject, config.seed)
        with open(save_pth, 'rb') as f:
            saved = pickle.load(f)
        manager.model.load_state_dict(saved['state_dict'])
        manager.env.time_cost = config.time_cost
        manager.env.monkey.set_param([
            config.push_cost, config.turn_price, config.move_price,
            config.look_price, config.center_cost,
        ])
        for box in manager.env.boxes:
            box.reward = config.reward
        manager.algo = PPO(
            env=TimeLimit(manager.model, 1000), policy='MlpPolicy',
            gamma=config.gamma, ent_coef=config.ent_coef,
        )
        return 0
    def reset():
        n_steps = 2000 # number of steps per update
        num_updates = 100 # number of policy updates

        manager.algo.policy.set_training_mode(True)
        with tqdm(total=num_updates, unit='update', leave=False) as pbar:
            pbar_cb = ProgressBarCallback(pbar, disp_freq=n_steps)
            manager.algo.learn(
                total_timesteps=num_updates*n_steps, callback=pbar_cb,
            )
        manager.algo.policy.set_training_mode(False)
        manager.logs = pbar_cb.logs
    def step():
        ...
    def get_ckpt():
        ckpt = {
            'policy': tensor2array(manager.algo.policy.state_dict()),
            'logs': manager.logs,
        }
        return ckpt
    def load_ckpt(ckpt) -> int:
        manager.algo.policy.load_state_dict(
            array2tensor(ckpt['policy'], manager.algo.device),
        )
        manager.logs = ckpt['logs']
        return 0

    manager.setup = setup
    manager.reset = reset
    manager.step = step
    manager.get_ckpt = get_ckpt
    manager.load_ckpt = load_ckpt
    return manager

def main(
    store_dir: Path|str,
    subject: str,
    choices: Path|str = 'agents_spec.yaml',
    patience: float = 2.,
    num_agents: int|None = None,
):
    r"""Trains RL agents for different reward parameters.

    Args
    ----
    store_dir:
        Directory where trained belief models and RL agents are saved.
    subject:
        Subject name.
    choices:
        Choices for specifying agents.
    patience:
        Arguments for RL agent manager, see `Manager` for more details.
    num_agents:
        Number of agents to train, ``None`` for all specified parameters.

    """
    store_dir = Path(store_dir)
    with open(store_dir/choices, 'r') as f:
        choices = yaml.safe_load(f)

    default = Config({'seed': 0, 'gamma': 0.99, 'ent_coef': 0.01})
    sig = inspect.signature(ForagingEnv).parameters
    default.fill({k: sig[k].default for k in ['time_cost']})
    sig = inspect.signature(Monkey).parameters
    default.fill({k: sig[k].default for k in ['push_cost', 'turn_price', 'move_price', 'look_price', 'center_cost']})
    sig = inspect.signature(BaseFoodBox).parameters
    default.fill({k: sig[k].default for k in ['reward']})

    configs = choices2configs(choices)
    for config in configs:
        config.fill(default)

    manager = create_manager(store_dir, subject, patience)
    manager.batch(configs, 0, num_agents)

if __name__=='__main__':
    main(**from_cli().fill({
        'store_dir': STORE_DIR,
        'subject': 'marco',
    }))
