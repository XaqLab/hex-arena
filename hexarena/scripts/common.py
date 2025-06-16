from jarvis.config import Config
from irc.model import SamplingBeliefModel

from ..env import SimilarBoxForagingEnv
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
    env_kw: dict|None = None,
) -> SimilarBoxForagingEnv:
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
    env_kw:
        Additional keyword arguments of environment, see `SimilarBoxForagingEnv`
        for more details.

    Returns
    -------
    env:
        A default environment with boxes sorted from worst to best.

    """
    assert gamma in [1, 10], "Unsupported gamma distribution shape"
    assert kappa>0, "Cue reliability needs to be positive"
    if gamma==1:
        env_kw = Config(env_kw).fill({
            'box': {
                '_target_': 'hexarena.box.PoissonBox',
                'kappa': kappa,
            },
            'boxes': [{'tau': tau} for tau in [35, 21, 15]],
        })
    if gamma==10:
        env_kw = Config(env_kw).fill({
            'box': {
                '_target_': 'hexarena.box.GammaBox',
                'kappa': kappa,
            },
            'boxes': [{'tau': tau} for tau in [21, 14, 7]],
        })
    env = SimilarBoxForagingEnv(**env_kw)
    return env


def create_env_and_model(
    subject: str, kappa: float = 0.,
    env_kw: dict|None = None,
    model_kw: dict|None = None,
) -> tuple[SimilarBoxForagingEnv, SamplingBeliefModel]:
    r"""Creates default belief model.

    Args
    ----
    subject, kappa:
        Subject name and cue reliability.
    env_kw:
        Keyword arguments of the environment.
    model_kw:
        Keyword arguments of the belief model.

    Returns
    -------
    env:
        A foraging environment. For 'marco', it is the exponential schedule. For
        'viktor' it is the Gamma distribution schedule. Color cues are also
        presented in different ways.
    model:
        A sampling-based belief model, with state independence specified.

    """
    if subject not in ['marco', 'viktor']:
        raise NotImplementedError("'subject' can only be 'marco' or 'viktor'")
    if subject=='marco':
        env_kw = Config(env_kw).fill({
            'box': {
                '_target_': 'hexarena.box.StationaryBox',
                'num_levels': 10,
            },
            'boxes': [{'tau': tau} for tau in [35, 21, 15]],
        })
        model_kw = Config(model_kw).fill({
            's_idcs': [[0], [1], [2, 3], [4, 5], [6, 7]],
        })
    if subject=='viktor':
        env_kw = Config(env_kw).fill({
            'box': {
                '_target_': 'hexarena.box.GammaLinearBox',
                'max_interval': 36,
            },
            'boxes': [{'tau': tau} for tau in [21, 14, 7]],
            'monkey': {'num_grades': 48},
        })
        model_kw = Config(model_kw).fill({
            'p_s.phis': [{
                'embedder._target_': 'hexarena.box.LinearBoxStateEmbedder',
                'mlp_features': [16, 4],
            }]*3,
            'ebd_s': {
                'ebds': [{'_target_': 'hexarena.box.LinearBoxStateEmbedder'}]*3,
                'idcs': [[0], [1], [2]],
            },
        })
    env_kw.update({'box.kappa': kappa})
    env = SimilarBoxForagingEnv(**env_kw)
    model = SamplingBeliefModel(env, **model_kw)
    if subject=='viktor':
        model.train_kw.update({'num_epochs': 60})
        model.estimate_kw.update({'sga_kw.num_epochs': 200})
    return env, model
