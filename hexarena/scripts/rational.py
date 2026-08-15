import yaml
from itertools import product, permutations
from jarvis.config import from_cli
from irc.manager import RationalPolicyManager

from .. import STORE_DIR
from .common import create_env


def main(
    bandit: bool = True,
    spec_pth: str = 'rational.spec.yaml',
    n_works: int|None = None,
    **kwargs,
):
    r"""Trains rational policy by manager.

    Args
    ----
    bandit:
        Whether to train for bandit environment or not. If ``True``, bandit
        environment with Poisson boxes is used.
    spec_pth:
        Path to the yaml file that specifies the parameter grid.
    n_works:
        Number of rational policies to be trained in total.
    kwargs:
        Additional keyword arguments to specify rational policy configuration,
        see `RationalPolicyManager.setup` for more details.

    """
    if not bandit:
        raise NotImplementedError
    env = create_env(gamma=(1 if bandit else 10), no_arena=bandit)
    with open(STORE_DIR/spec_pth, 'r') as f:
        spec = yaml.safe_load(f)
    taus = spec.pop('taus', [15., 21., 35.])
    assert len(taus)==env.n_boxes, f"Expect {env.n_boxes} different tau values."
    configs = []
    for seed, gamma, ent_coef, push_cost in product(
        spec['seed'], spec['gamma'], spec['ent_coef'], spec['push_cost'],
    ):
        env.monkey.push_cost = push_cost
        for perm in permutations(range(env.n_boxes)):
            for i in range(env.n_boxes):
                env.boxes[i].tau = taus[perm[i]]
            configs.append({
                'param': env.get_param(), 'seed': seed,
                'gamma': gamma, 'ent_coef': ent_coef,
                **kwargs,
            })
    manager = RationalPolicyManager(env, STORE_DIR/'rational')
    manager.batch(configs, n_works=n_works)


if __name__=='__main__':
    main(**from_cli())
