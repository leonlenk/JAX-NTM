from abc import ABC, abstractmethod

from flax import linen as nn
from jax import Array

from Common.ControllerInterface import ControllerInterface
from Common.globals import METADATA
from Common.MemoryInterface import MemoryInterface


class BackboneInterface(nn.Module, ABC):
    prng_key: Array
    # TODO make more general for e.g. transformers (and override get_metadata with custom info)
    features: int
    layers: int
    num_outputs: int
    # TODO allow arbitrary numbers of controllers. And make naming consistent between head and controller?
    read_head: ControllerInterface
    write_head: ControllerInterface

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

    def get_metadata(self) -> dict:
        return {
            METADATA.NAME: self.__class__.__name__,
            # TODO accept a seed and convert it into a prng_key so we can add METADATA.SEED
        }
