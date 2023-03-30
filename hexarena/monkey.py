import numpy as np
from gym.spaces import MultiDiscrete
from jarvis.config import Config

from typing import Optional, Union

from .arena import Arena
from .alias import MonkeyState

class Monkey:
    r"""Class for the monkey in an arena."""

    def __init__(self,
        arena: Union[Arena, dict, None] = None,
    ):
        r"""
        Args
        ----
        arena:
            The arena in which the monkey plays in.

        """
        if arena is None or isinstance(arena, dict):
            arena = Config(arena)
            arena._target_ = 'hexarena.arena.Arena'
            arena: Arena = arena.instantiate()
        self.arena = arena

        # state: (pos, gaze)
        self.state_space = MultiDiscrete([self.arena.num_tiles]*2)

        self.rng = np.random.default_rng()

    def reset(self, seed: Optional[int] = None) -> None:
        r"""Resets the monkey state.

        Put the monkey on the outer region randomly, and sets up the gaze
        position to a random tile in inner region.

        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.pos = self.rng.choice(self.arena.outers)
        self.gaze = self.rng.choice(self.arena.inners)

    def get_state(self) -> MonkeyState:
        r"""Returns monkey state.

        Tile index of the current position and gaze position are returned.

        """
        state = (self.pos, self.gaze)
        return state

    def set_state(self, state: MonkeyState) -> None:
        r"""Sets the monkey state."""
        self.pos, self.gaze = state
