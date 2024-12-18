from abc import ABC, abstractmethod

import jax.numpy as jnp
from flax import linen as nn
from jax import Array

from Common.globals import METADATA


class MemoryInterface(ABC, nn.Module):
    def update_step(self, *args, **kwargs) -> None:
        return

    def add_output(self, *args, **kwargs) -> None:
        return

    @abstractmethod
    def read(self, memory_weights: Array, read_weights: Array) -> Array:
        raise NotImplementedError

    @abstractmethod
    def write(
        self,
        memory_weights: Array,
        write_weights: Array,
        erase_vector: Array,
        add_vector: Array,
    ) -> Array:
        raise NotImplementedError

    @abstractmethod
    def address(
        self,
        memory_weights: Array,
        key_vector: Array,
        key_strength: Array,
        interp_gate_scalar: Array,
        shift_weights: Array,
        sharpen_scalar: Array,
        previous_weights: Array,
    ) -> Array:
        raise NotImplementedError

    def get_metadata(self) -> dict:
        return {
            METADATA.NAME: self.__class__.__name__,
        }


class MemoryStub(MemoryInterface):
    def __init__(self, N, M):
        self.N = N
        self.M = M

    def size(self):
        return self.N, self.M

    def read(self, memory_weights: Array, read_weights: Array) -> Array:
        return read_weights

    def write(
        self,
        memory_weights: Array,
        write_weights: Array,
        erase_vector: Array,
        add_vector: Array,
    ) -> Array:
        return write_weights

    def address(
        self,
        key_vector,
        key_strength,
        interp_gate_scalar,
        shift_weights,
        sharpen_scalar,
        previous_weights,
    ):
        """Return key_vector (length M) padded to be of length N"""
        return jnp.pad(
            key_vector, self.N - self.M
        )  # note that jnp.pad allows negative padding. pad value defaults to 0s
