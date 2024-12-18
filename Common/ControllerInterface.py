from abc import ABC, abstractmethod

from flax import linen as nn
from jax import Array

from Common.globals import METADATA
from Common.MemoryInterface import MemoryInterface


class ControllerInterface(ABC, nn.Module):
    N_dim_memory: int
    M_dim_memory: int

    @abstractmethod
    def setup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _address_memory(
        self,
        memory_weights: Array,
        k: Array,
        β: Array,
        g: Array,
        s: Array,
        y: Array,
        w_prev: Array,
        memory_model: MemoryInterface,
    ) -> Array:
        raise NotImplementedError

    @abstractmethod
    def is_read_controller(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def __call__(
        self,
        embeddings: Array,
        w_prev: Array,
        memory_weights: Array,
        memory_model: MemoryInterface,
    ) -> tuple[Array, Array]:
        raise NotImplementedError

    def get_metadata(self) -> dict:
        return {
            METADATA.NAME: self.__class__.__name__,
        }
