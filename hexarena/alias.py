from numpy import ndarray as Array
from numpy.random import Generator as RandGen

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.artist import Artist
from matplotlib.animation import Animation

from collections.abc import Collection

from irc.alias import EnvParam
BoxObservation = Collection[Collection[int]]
BoxState = tuple[int, int]
MonkeyState = tuple[int, int]
State = Collection[int]
Observation = Collection[int]
