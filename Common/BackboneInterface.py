from abc import ABC, abstractmethod

from flax import linen as nn
from jax import Array

from Common.ControllerInterface import ControllerInterface
from Common.MemoryInterface import MemoryInterface


class BackboneInterface(nn.Module, ABC):
    prng_key: Array
    features: int
    layers: int
    num_outputs: int
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
