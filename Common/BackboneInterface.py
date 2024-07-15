from abc import ABC, abstractmethod

from flax import linen as nn
from jax import Array

from Common.ControllerInterface import ControllerInterface
from Common.MemoryInterface import MemoryInterface


class PreviousState:
    def __init__(self, memory_weights, read_previous, write_previous):
        self.memory_weights = memory_weights
        self.read_previous = read_previous
        self.write_previous = write_previous


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
        self, input: Array, memory_model: MemoryInterface, previous_state: PreviousState
    ) -> tuple[Array, Array, PreviousState]:
        raise NotImplementedError
