from abc import ABC, abstractmethod

import jax.numpy as jnp
from jax import Array


class MemoryInterface(ABC):
    @abstractmethod
    def size(self) -> tuple[int, int]:
        pass

    @abstractmethod
    def read(self, read_weights: Array) -> Array:
        pass

    @abstractmethod
    def write(
        self, read_weights: Array, erase_vector: Array, add_vector: Array
    ) -> None:
        pass

    @abstractmethod
    def address(
        self,
        key_vector: Array,
        key_strength: Array,
        interp_gate_scalar: Array,
        shift_weights: Array,
        sharpen_scalar: Array,
        previous_weights: Array,
    ) -> Array:
        pass


class MemoryStub(MemoryInterface):
    def __init__(self, N, M):
        self.N = N
        self.M = M

    def size(self):
        return self.N, self.M

    def read(self, read_weights):
        """Return the input (length N)"""
        return read_weights

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
