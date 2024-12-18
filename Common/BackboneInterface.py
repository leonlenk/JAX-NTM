from abc import ABC, abstractmethod
from typing import Sequence

import jax
from flax import linen as nn
from jax import Array

from Common.ControllerInterface import ControllerInterface
from Common.globals import METADATA
from Common.MemoryInterface import MemoryInterface


class BackboneInterface(nn.Module, ABC):
    # TODO make more general for e.g. transformers (and override get_metadata with custom info)
    layers: int
    num_outputs: int
    # TODO allow arbitrary numbers of controllers. And make naming consistent between head and controller?
    read_heads: Sequence[ControllerInterface]
    write_heads: Sequence[ControllerInterface]

    @abstractmethod
    def setup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def __call__(
        self,
        input: Array,
        memory_weights: Array,
        read_previous: Array,
        write_previous: Array,
        memory_model: MemoryInterface,
    ) -> tuple[Array, Array, Array, Array, Array]:
        raise NotImplementedError

    def parameter_count(self, params):
        return sum(x.size for x in jax.tree_util.tree_leaves(params))

    def get_metadata(self) -> dict:
        return {
            METADATA.NAME: self.__class__.__name__,
            # TODO accept a seed and convert it into a prng_key so we can add METADATA.SEED
        }
