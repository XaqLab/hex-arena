import numpy as np

from ..env import BaseForagingEnv, BanditForagingEnv, ArenaForagingEnv
from ..utils import get_valid_blocks


GAMMAS = { # default shape parameter of schedule distribution
    'marco': 1., 'dylan': 1., 'viktor': 10.,
}

def get_block_ids(
    subject: str, kappa: float,
    gamma: float|None = None,
    min_pos_ratio: float = 0.5,
    min_gaze_ratio: float = 0.1,
    **kwargs,
) -> list[tuple[str, int]]:
    r"""Returns common block IDs for batch processing.

    Args
    ----
    subject, kappa:
        Subject name and cue reliability.
    gamma:
        Shape parameter of reward interval distribution.
    min_pos_ratio, min_gaze_ratio:
        Minimum ratio value of position data and gaze data respectively.
    kwargs:
        Additional keyword arguments for `get_valid_blocks`.

    Returns
    -------
    block_ids:
        Sorted block IDs as `(session_id, block_idx)`.

    """
    if gamma is None:
        gamma = GAMMAS[subject]
    block_infos = get_valid_blocks(
        subject, min_pos_ratio=min_pos_ratio, min_gaze_ratio=min_gaze_ratio, **kwargs,
    )
    block_ids = []
    for (session_id, block_idx), meta in block_infos.items():
        if meta['kappa']!=kappa or meta['gamma']!=gamma:
            continue
        block_ids.append((session_id, block_idx))
    return sorted(block_ids)


def create_env(
    gamma: float = 1., kappa: float = 0.1,
    taus: list[float]|None = None,
    no_arena: bool = False,
    env_kw: dict|None = None,
) -> BaseForagingEnv:
    r"""Creates default foraging environment.

    Args
    ----
    gamma:
        Shape parameter of the Gamma distribution of reward interval. Only '1'
        and '10' are supported. When `gamma==1`, the food box uses exponential
        schedule with 'tau' as 35, 21 and 15 secs. When `gamma==10`, the food
        boxes uses Gamma schedule and linear cue with 'tau' as 21, 14 and 7 secs.
    kappa:
        Cue reliability.
    taus:
        Schedule parameter of all three boxes. If provided, it needs to be a
        permutation of the default values.
    no_arena:
        Whether the arena is disabled or not. When ``True``, `BanditForagingEnv`
        will be used.
    env:
        Additional keyword arguments of a `BaseForagingEnv` object. It won't
        overwrite the default specification set by other arguments.

    Returns
    -------
    env_kw:
        A default environment with boxes sorted from worst to best.

    """
    assert gamma in [1, 10], "Unsupported gamma distribution shape"
    assert kappa>0, "Cue reliability needs to be positive"
    if gamma==1:
        box = 'hexarena.box.PoissonBox'
        _taus = [35, 21, 15]
    if gamma==10:
        box = 'hexarena.box.GammaBox'
        _taus = [21, 14, 7]
    if taus is None:
        taus = _taus
    else:
        assert np.all(np.sort(taus)==np.sort(_taus))
    if no_arena:
        env_cls = BanditForagingEnv
    else:
        env_cls = ArenaForagingEnv
    env = env_cls(
        boxes=[{
            '_target_': box, 'kappa': kappa, 'tau': tau,
        } for tau in taus], **({} if env_kw is None else env_kw),
    )
    return env
