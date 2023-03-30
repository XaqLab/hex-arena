from jarvis.config import Config

from typing import Optional

from . import rcParams

class Box:
    r"""Class for a food box with 2D color cue."""

    def __init__(self,
        rate: Optional[float] = None,
        step_size: Optional[float] = None,
        sigma_c: Optional[float] = None,
        num_grades: Optional[int] = None,
        resol: Optional[int] = None,
    ):
        r"""
        Args
        ----
        rate:
            Poisson rate of food appears, in counts per second.
        step_size:
            Time step size for temporal discretization, in seconds.
        sigma_c:
            Parameter governing the noise of color cues, should be non-negative.
        num_grades:
            Number of distinct colors on a color map, used for discretizing the
            color cue.
        resol:
            Spatial resolution of color cues on a screen. For example, if
            `resol=4`, a 4*4 grid of integers will be used to represent the
            colored pattern on the screen.

        """
        _rcParams = Config(rcParams.get('box.Box._init_'))
        self.rate = rate or _rcParams.rate
        self.step_size = step_size or _rcParams.step_size
        self.sigma_c = sigma_c or _rcParams.sigma_c
        self.num_grades = num_grades or _rcParams.num_grades
        self.resol = resol or _rcParams.resol
