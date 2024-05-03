from numpy import ndarray as Array
from numpy.random import Generator as RandGen

from torch import Tensor

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.artist import Artist
from matplotlib.animation import Animation

from collections.abc import Sequence

EnvParam = Sequence[float]
BoxObservation = Sequence[Sequence[int]]
BoxState = tuple[int, int]
MonkeyState = tuple[int, int]
State = Sequence[int]
Observation = Sequence[int]
