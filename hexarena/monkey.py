
from gym.spaces import MultiDiscrete
from jarvis.config import Config

from typing import Union

from .arena import Arena

class Monkey:
    r"""Class for the monkey in an arena."""

    def __init__(self,
        arena: Union[Arena, dict, None] = None,
    ):
        if arena is None or isinstance(arena, dict):
            arena = Config(arena)
            arena._target_ = 'hexarena.arena.Arena'
            arena: Arena = arena.instantiate()
        self.arena = arena

        # state (pos, gaze)
        self.state_space = MultiDiscrete([self.arena.num_tiles]*2)
