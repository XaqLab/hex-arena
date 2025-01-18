from jarvis.config import Config
from irc.model import SamplingBeliefModel

from ..env import SimilarBoxForagingEnv
from ..utils import get_valid_blocks


def get_block_ids(
    subject: str, kappa: float,
    min_pos_ratio: float = 0.5,
    min_gaze_ratio: float = 0.1,
    **kwargs,
) -> list[tuple[str, int]]:
    r"""Returns common block IDs for batch processing.

    Args
    ----
    subject, kappa:
        Subject name and cue reliability.
    min_pos_ratio, min_gaze_ratio:
        Minimum ratio value of position data and gaze data respectively.
    kwargs:
        Additional keyword arguments for `get_valid_blocks`.

    """
    block_infos = get_valid_blocks(
        subject, min_pos_ratio=min_pos_ratio, min_gaze_ratio=min_gaze_ratio, **kwargs,
    )
    block_ids = []
    for (session_id, block_idx), meta in block_infos.items():
        if meta['kappa']!=kappa:
            continue
        if subject=='marco' and meta['gamma']!=1:
            continue
        if subject=='viktor' and meta['gamma']!=10:
            continue
        block_ids.append((session_id, block_idx))
    return sorted(block_ids)


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
                'max_interval': 40,
            },
            'boxes': [{'tau': tau} for tau in [21, 14, 7]],
        })
        model_kw = Config(model_kw).fill({
            'p_s.phis': [{
                'embedder._target_': 'hexarena.box.LinearBoxStateEmbedder',
                'mlp_features': [16, 8],
            }]*3,
        })
    env_kw.update({'box.kappa': kappa})
    env = SimilarBoxForagingEnv(**env_kw)
    model = SamplingBeliefModel(env, **model_kw)
    return env, model
