from functools import partial
from typing import List

import jax
import jax.numpy as jnp
from flax import linen as nn

from Common import globals
from Common.ControllerInterface import ControllerInterface
from Common.MemoryInterface import MemoryInterface


def _split_cols(matrix: jax.Array, length_indices: tuple) -> List[jax.Array]:
    """Split a 2D matrix to variable length columns."""
    return jnp.split(matrix, length_indices, axis=-1)


class NTMControllerTemplate(ControllerInterface, nn.Module):
    """An NTM Read/Write Controller."""

    N_dim_memory: int
    M_dim_memory: int

    def setup(self):
        """
        Initilize the read/write controller.
        """
        # xavier uniform initialization but with a gain of 1.4
        self.weight_initializer = nn.initializers.variance_scaling(
            scale=1.4, mode="fan_avg", distribution="uniform"
        )
        self.bias_initializer = nn.initializers.normal(stddev=0.01)

    # TODO: give better variable names and add type annotations
    def _address_memory(self, memory_weights, k, β, g, s, y, w_prev, memory_model):
        # Handle Activations
        k = k.copy()
        β = nn.softplus(β)
        g = nn.sigmoid(g)
        s = nn.softmax(s, axis=-1)
        y = 1 + nn.softplus(y)

        w = memory_model.address(
            memory_weights,
            k,
            β,
            g,
            s,
            y,
            w_prev,
        )

        return w


class NTMReadController(NTMControllerTemplate):
    def setup(self):
        super().setup()

        # Corresponding to k, β, g, s, γ sizes from the paper
        self.read_lengths = (self.M_dim_memory, 1, 1, 3, 1)
        self.length_indices = (
            self.M_dim_memory,
            self.M_dim_memory + 1,
            self.M_dim_memory + 2,
            self.M_dim_memory + 5,
        )

        self._split_cols = jax.jit(
            partial(_split_cols, length_indices=self.length_indices)
        )

    def is_read_controller(self) -> bool:
        return True

    # TODO: figure out type annotations
    @nn.compact
    def __call__(self, embeddings, w_prev, memory_weights, memory_model):
        """NTMReadController forward function.

        :param embeddings: input representation of the model.
        :param w_prev: [m] previous step state
        """
        memory_addresses = nn.Dense(
            sum(self.read_lengths),
            kernel_init=self.weight_initializer,
            bias_init=self.bias_initializer,
        )(embeddings)
        k, β, g, s, y = self._split_cols(memory_addresses)

        # Read from memory
        memory_locations = self._address_memory(
            memory_weights, k, β, g, s, y, w_prev, memory_model
        )

        memory_data = memory_model.read(
            memory_weights,
            memory_locations,
        )

        return memory_data, memory_locations


class NTMWriteController(NTMControllerTemplate):
    def setup(self):
        super().setup()
        # Corresponding to k, β, g, s, γ, e, a sizes from the paper
        self.write_lengths = (
            self.M_dim_memory,
            1,
            1,
            3,
            1,
            self.M_dim_memory,
            self.M_dim_memory,
        )

        self.length_indices = (
            self.M_dim_memory,
            self.M_dim_memory + 1,
            self.M_dim_memory + 2,
            self.M_dim_memory + 5,
            self.M_dim_memory + 6,
            2 * self.M_dim_memory + 6,
        )

        self._split_cols = jax.jit(
            partial(_split_cols, length_indices=self.length_indices)
        )

    def is_read_controller(self) -> bool:
        return False

    # TODO: figure out type annotations
    @nn.compact
    def __call__(
        self,
        embeddings: jax.Array,
        w_prev: jax.Array,
        memory_weights: jax.Array,
        memory_model: MemoryInterface,
    ):
        """NTMWriteController forward function.

        :param embeddings: input representation of the model.
        :param w_prev: [m] previous step state
        """
        memory_components = nn.Dense(
            sum(self.write_lengths),
            kernel_init=self.weight_initializer,
            bias_init=self.bias_initializer,
        )(embeddings)
        k, β, g, s, y, erase, add_weight = self._split_cols(memory_components)

        # e should be in [0, 1]
        erase_weight = nn.sigmoid(erase)

        # Write to memory
        memory_addresses = self._address_memory(
            memory_weights, k, β, g, s, y, w_prev, memory_model
        )

        memory_weights = memory_model.write(
            memory_weights,
            memory_addresses,
            erase_weight,
            add_weight,
        )

        return memory_weights, memory_addresses


# TODO: add test cases
if __name__ == "__main__":
    from Memories.NTM_graves2014 import NTMMemory

    test_n = 8
    test_m = 9
    test_model_feature_size = 10
    learning_rate = 5e-3

    memory_model = NTMMemory()
    memory_weights = jnp.zeros((test_n, test_m))
    read_controller = NTMReadController(test_n, test_m)
    write_controller = NTMWriteController(test_n, test_m)

    rng_key = jax.random.key(globals.JAX.RANDOM_SEED)
    key1, key2 = jax.random.split(rng_key)

    read_controller_variables = read_controller.init(
        key1,
        jnp.ones((test_model_feature_size,)),
        jnp.ones((test_n,)),
        memory_weights,
        memory_model,
    )
    print("Initialized read controller")
    write_controller_variables = write_controller.init(
        key2,
        jnp.ones((test_model_feature_size,)),
        jnp.ones((test_n,)),
        memory_weights,
        memory_model,
    )
    print("Initialized write controller")

    print(read_controller.is_read_controller())
    assert read_controller.is_read_controller()
    assert not write_controller.is_read_controller()

    print("passed all tests")
