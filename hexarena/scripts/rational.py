import yaml
from itertools import product, permutations
from jarvis.config import from_cli
from irc.manager import RationalPolicyManager

from .. import STORE_DIR
from .common import create_env


def main(
    gamma: float = 1.,
    no_arena: bool = False,
    cue_in_state: bool = False,
    spec_pth: str = 'rational.spec.yaml',
    n_epochs: int|None = None,
    n_works: int|None = None,
    **kwargs,
):
    r"""Trains rational policy by manager.

    Args
    ----
    gamma, no_arena, cue_in_state:
        Arguments of `create_env`.
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
    if gamma!=1:
        raise NotImplementedError("Only exponential schedule is supported.")
    env = create_env(gamma=gamma, no_arena=no_arena, cue_in_state=cue_in_state)
    with open(STORE_DIR/spec_pth, 'r') as f:
        spec = yaml.safe_load(f)
    taus = spec.pop('taus', [15., 21., 35.])
    assert len(taus)==env.n_boxes, f"Expect {env.n_boxes} different tau values."
    configs = []
    for perm in permutations(range(env.n_boxes)):
        for i in range(env.n_boxes):
            env.boxes[i].tau = taus[perm[i]]
        if no_arena:
            for seed, gamma, ent_coef, push_cost in product(
                spec['seed'], spec['gamma'], spec['ent_coef'], spec['push_cost'],
            ):
                env.monkey.push_cost = push_cost
                configs.append({
                    'param': env.get_param(), 'seed': seed,
                    'gamma': gamma, 'ent_coef': ent_coef,
                    **kwargs,
                })
        else:
            for seed, gamma, ent_coef, push_cost, turn_price, move_price, center_cost in product(
                spec['seed'], spec['gamma'], spec['ent_coef'], spec['push_cost'],
                spec['turn_price'], spec['move_price'], spec['center_cost'],
            ):
                env.monkey.push_cost = push_cost
                env.monkey.turn_price = turn_price
                env.monkey.move_price = move_price
                env.center_cost = center_cost
                configs.append({
                    'param': env.get_param(), 'seed': seed,
                    'gamma': gamma, 'ent_coef': ent_coef,
                    **kwargs,
                })
    store_dir = STORE_DIR/'rational'/'[{}][{}][gamma{}]'.format(
        'bandit' if no_arena else 'arena',
        'visual' if cue_in_state else 'blind',
        str(int(gamma)),
    )
    manager = RationalPolicyManager(env, store_dir)
    manager.batch(configs, n_epochs=n_epochs, n_works=n_works)


if __name__=='__main__':
    main(**from_cli())
