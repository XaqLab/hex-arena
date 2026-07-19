import yaml
from itertools import product
from jarvis.config import from_cli
from irc.manager import RationalPolicyManager
from irc.example.env import FoodBoxEnv

from .. import STORE_DIR


def main(
    spec_pth: str = 'irc.example.spec.yaml',
    n_works: int|None = None,
    **kwargs,
):
    r"""Trains rational policy by manager.

    Args
    ----
    spec_pth:
        Path to the yaml file that specifies the parameter grid.
    n_works:
        Number of rational policies to be trained in total.
    kwargs:
        Additional keyword arguments to specify rational policy configuration,
        see `RationalPolicyManager.setup` for more details.

    """
    env = FoodBoxEnv()
    with open(STORE_DIR/spec_pth, 'r') as f:
        spec = yaml.safe_load(f)
    configs = []
    for seed, gamma, ent_coef, p_appear, p_cue in product(
        spec['seed'], spec['gamma'], spec['ent_coef'],
        spec['p_appear'], spec['p_cue'],
    ):
        env.p_appear = p_appear
        env.p_cue = p_cue
        configs.append({
            'param': env.get_param(), 'seed': seed,
            'gamma': gamma, 'ent_coef': ent_coef,
            **kwargs,
        })
    manager = RationalPolicyManager(env, STORE_DIR/'irc.example/rational')
    manager.batch(configs, n_works=n_works)


if __name__=='__main__':
    main(**from_cli())
