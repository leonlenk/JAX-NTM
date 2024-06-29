from typing import List, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state

from Common import globals
from Common.ControllerInterface import ControllerInterface
from Common.MemoryInterface import MemoryInterface


def _split_cols(matrix: jax.Array, lengths: Tuple) -> List[jax.Array]:
    """Split a 2D matrix to variable length columns."""
    assert jnp.size(matrix, axis=-1) == sum(
        lengths
    ), "Lengths must be summed to num columns"
    length_indices = jnp.cumsum(jnp.asarray(lengths))[:-1]

    return jnp.split(matrix, length_indices, axis=-1)


@ControllerInterface.register
class NTMControllerTemplate(nn.Module):
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
    def _address_memory(self, k, β, g, s, y, w_prev, memory_state, memory_model):
        # Handle Activations
        k = k.copy()
        β = nn.softplus(β)
        g = nn.sigmoid(g)
        s = nn.softmax(s, axis=-1)
        y = 1 + nn.softplus(y)

        w = memory_state.apply_fn(
            {globals.JAX.PARAMS: memory_state.params},
            k,
            β,
            g,
            s,
            y,
            w_prev,
            method=memory_model.address,
        )

        return w


class NTMReadController(NTMControllerTemplate):
    def setup(self):
        super().setup()

        # Corresponding to k, β, g, s, γ sizes from the paper
        self.read_lengths = (self.M_dim_memory, 1, 1, 3, 1)

    def create_new_state(self, batch_size: int) -> jax.Array:
        # The state holds the previous time step address weightings
        return jnp.zeros((batch_size, self.N_dim_memory))

    def is_read_controller(self) -> bool:
        return True

    # TODO: figure out type annotations
    @nn.compact
    def __call__(
        self,
        embeddings: jax.Array,
        w_prev: jax.Array,
        memory_state: train_state.TrainState,
        memory_model: MemoryInterface,
    ):
        """NTMReadController forward function.

        :param embeddings: input representation of the model.
        :param w_prev: [1xm] previous step state
        """
        memory_addresses = nn.Dense(
            sum(self.read_lengths),
            kernel_init=self.weight_initializer,
            bias_init=self.bias_initializer,
        )(embeddings)
        k, β, g, s, y = _split_cols(memory_addresses, self.read_lengths)

        # Read from memory
        memory_locations = self._address_memory(
            k, β, g, s, y, w_prev, memory_state, memory_model
        )

        memory_data = memory_state.apply_fn(
            {globals.JAX.PARAMS: memory_state.params},
            memory_locations,
            method=memory_model.read,
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

    def create_new_state(self, batch_size: int) -> jax.Array:
        return jnp.zeros((batch_size, self.N_dim_memory))

    def is_read_controller(self) -> bool:
        return False

    # TODO: figure out type annotations
    @nn.compact
    def __call__(
        self,
        embeddings: jax.Array,
        w_prev: jax.Array,
        memory_state: train_state.TrainState,
        memory_model: MemoryInterface,
    ):
        """NTMWriteController forward function.

        :param embeddings: input representation of the model.
        :param w_prev: [1xm] previous step state
        """
        memory_components = nn.Dense(
            sum(self.write_lengths),
            kernel_init=self.weight_initializer,
            bias_init=self.bias_initializer,
        )(embeddings)
        k, β, g, s, y, erase, add_weight = _split_cols(
            memory_components, self.write_lengths
        )

        # e should be in [0, 1]
        erase_weight = nn.sigmoid(erase)

        # Write to memory
        memory_addresses = self._address_memory(
            k, β, g, s, y, w_prev, memory_state, memory_model
        )

        memory_state.apply_fn(
            {globals.JAX.PARAMS: memory_state.params},
            memory_addresses,
            erase_weight,
            add_weight,
            method=memory_model.write,
            mutable=[globals.JAX.PARAMS],
        )

        return memory_addresses


# TODO: add test cases
if __name__ == "__main__":
    from Common import globals
    from Memories.NTM_graves2014 import Memory

    test_n = 8
    test_m = 9
    test_model_feature_size = 10
    memory_model = Memory(test_n, test_m)
    read_controller = NTMReadController(*memory_model.size())
    write_controller = NTMWriteController(*memory_model.size())

    rng_key = jax.random.key(globals.JAX.RANDOM_SEED)
    key1, key2 = jax.random.split(rng_key)

    read_controller_variables = read_controller.init(
        key1,
        jnp.ones((1, test_model_feature_size)),
        jnp.ones((1, test_n)),
        memory_model,
    )
    print("Initialized read controller")
    write_controller_variables = write_controller.init(
        key2,
        jnp.ones((1, test_model_feature_size)),
        jnp.ones((1, test_n)),
        memory_model,
    )
    print("Initialized write controller")

    assert read_controller.is_read_controller()
    assert not write_controller.is_read_controller()
