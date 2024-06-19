from Common import common

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Any, Tuple, List


def _split_cols(matrix: jax.Array, lengths: Tuple) -> List[jax.Array]:
    """Split a 2D matrix to variable length columns."""
    assert jnp.size(matrix, axis=1) == sum(
        lengths
    ), "Lengths must be summed to num columns"
    length_indices = jnp.cumsum(jnp.asarray((0,) + lengths))
    results = []
    for start, end in zip(length_indices[:-1], length_indices[1:]):
        results += [matrix[:, start:end]]
    return results


class NTMControllerTemplate(nn.Module):
    """An NTM Read/Write Controller."""

    memory: Any

    def setup(self):
        # TODO: figure out memory typing
        """Initilize the read/write controller.

        :param memory: The memory object to be addressed by the controller.
        """
        self.N_dim_memory, self.M_dim_memory = self.memory.size()

        # xavier uniform initialization but with a gain of 1.4
        self.weight_initializer = nn.initializers.variance_scaling(
            scale=1.4, mode="fan_avg", distribution="uniform"
        )
        self.bias_initializer = nn.initializers.normal(stddev=0.01)

    def create_new_state(self, batch_size: int):
        raise NotImplementedError

    def register_parameters(self):
        raise NotImplementedError

    def is_read_controller(self):
        return NotImplementedError

    # TODO: give better variable names and add type annotations
    def _address_memory(self, k, β, g, s, y, w_prev):
        # Handle Activations
        k = k.clone()
        β = nn.softplus(β)
        g = nn.sigmoid(g)
        s = nn.softmax(s, axis=1)
        y = 1 + nn.softplus(y)

        w = self.memory.address(k, β, g, s, y, w_prev)

        return w


class NTMReadController(NTMControllerTemplate):
    def setup(self):
        super().setup()

        # Corresponding to k, β, g, s, γ sizes from the paper
        self.read_lengths = (self.M_dim_memory, 1, 1, 3, 1)

    def create_new_state(self, batch_size: int) -> jax.Array:
        # The state holds the previous time step address weightings
        return jnp.zeros(batch_size, self.N_dim_memory)

    def is_read_controller(self) -> bool:
        return True

    # TODO: figure out type annotations
    @nn.compact
    def __call__(self, embeddings: jax.Array, w_prev):
        """NTMReadController forward function.

        :param embeddings: input representation of the controller.
        :param w_prev: previous step state
        """
        memory_addresses = nn.Dense(
            sum(self.read_lengths),
            kernel_init=self.weight_initializer,
            bias_init=self.bias_initializer,
        )(embeddings)
        k, β, g, s, y = _split_cols(memory_addresses, self.read_lengths)

        # Read from memory
        memory_locations = self._address_memory(k, β, g, s, y, w_prev)
        memory_data = self.memory.read(memory_locations)

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
        return jnp.zeros(batch_size, self.N_dim_memory)

    def is_read_controller(self) -> bool:
        return False

    # TODO: figure out type annotations
    @nn.compact
    def __call__(self, embeddings: jax.Array, w_prev):
        """NTMWriteController forward function.

        :param embeddings: input representation of the controller.
        :param w_prev: previous step state
        """
        memory_addresses = nn.Dense(
            sum(self.write_lengths),
            kernel_init=self.weight_initializer,
            bias_init=self.bias_initializer,
        )(embeddings)
        k, β, g, s, y, e, a = _split_cols(memory_addresses, self.write_lengths)

        # TODO: what is e?
        # e should be in [0, 1]
        e = nn.sigmoid(e)

        # TODO: what is a?
        # Write to memory
        memory_addresses = self._address_memory(k, β, g, s, y, w_prev)
        self.memory.write(memory_addresses, e, a)

        return memory_addresses


# TODO: add test cases
if __name__ == "__main__":
    from Memories import NTM_graves2014

    memory_model = NTM_graves2014.Memory(10, 10, 10)
    read_controller = NTMReadController(memory_model)
    write_controller = NTMWriteController(memory_model)

    rng_key = jax.random.key(common.RANDOM_SEED)
    key1, key2 = jax.random.split(rng_key)

    # TODO: fix dim mismatch when read controller calls memory.address
    read_controller_variables = read_controller.init(
        key1, jnp.ones((1, 10)), jnp.ones((1, 10))
    )
    write_controller_variables = write_controller.init(
        key2, jnp.ones((1, 10)), jnp.ones((1, 10))
    )

    assert read_controller.is_read_controller()
    assert not write_controller.is_read_controller()

    print("NTM graves 2014 controller passed all tests")