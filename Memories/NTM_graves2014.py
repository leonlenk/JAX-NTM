from typing import cast

import jax.numpy as jnp
import optax
from flax import linen as nn

from Common import common
from Common.MemoryInterface import MemoryInterface


class Memory(MemoryInterface, nn.Module):
    """Memory interface for NTM from Graves 2014 (arXiv:1410.5401).
    Memory has a size of: N x M.
    N = number of memory locations
    N = size of vector at each memory location
    """

    N: int
    M: int

    def setup(self):
        memory_bias_initializer = (
            nn.initializers.uniform()
        )  # TODO test different memory bias initializers
        self.memory = self.variable(
            common.JAX.PARAMS,
            common.MACHINES.GRAVES2014.MEMORY_BIAS,
            (
                lambda s, d: memory_bias_initializer(
                    self.make_rng(common.JAX.PARAMS), s, d
                )
            ),
            (1, self.N, self.M),
            jnp.float_,
        )

    def size(self):
        return self.N, self.M

    def read(self, read_weights):
        """arXiv:1410.5401 section 3.1"""
        return jnp.matmul(read_weights, self.memory.value).squeeze(0)

    def write(self, write_weights, erase_vector, add_vector):
        """arXiv:1410.5401 section 3.2"""

        # calculate erase and add vectors
        write_weights = jnp.expand_dims(write_weights.squeeze(0), axis=1)
        erase = jnp.matmul(write_weights, erase_vector)
        add = jnp.matmul(write_weights, add_vector)
        # update memory
        self.memory.value = jnp.multiply(self.memory.value, 1 - erase)
        self.memory.value = jnp.add(self.memory.value, add)

    def address(
        self,
        key_vector,
        key_strength,
        interp_gate_scalar,
        shift_weights,
        sharpen_scalar,
        previous_weights,
    ):
        """arXiv:1410.5401 section 3.3"""
        # content addressing
        content_weights = self._similarity(key_vector, key_strength)

        # location addressing
        gate_weights = self._interpolate(
            content_weights, previous_weights, interp_gate_scalar
        )

        # calculate and sharpen weights
        weights = self._shift(gate_weights, shift_weights)
        sharpened_weights = self._sharpen(weights, sharpen_scalar)

        return sharpened_weights

    def _similarity(self, key_vector, key_strength):
        """arXiv:1410.5401 equations 5-6"""
        w = nn.softmax(
            key_strength
            * optax.cosine_similarity(self.memory.value, key_vector, epsilon=1e-16),
        )
        return w

    def _interpolate(self, weights, previous_weights, interp_gate_scalar):
        """arXiv:1410.5401 equation 7"""
        return (
            interp_gate_scalar * weights + (1 - interp_gate_scalar) * previous_weights
        )

    def _shift(self, gated_weighting, shift_weighting):
        """arXiv:1410.5401 equation 8"""
        return circular_convolution_1d(
            gated_weighting.squeeze(0), shift_weighting.squeeze(0)
        )

    def _sharpen(self, weights, sharpen_scalar):
        """arXiv:1410.5401 equation 9"""
        w = weights**sharpen_scalar
        w = jnp.divide(w, jnp.sum(w) + 1e-16)
        return w


def circular_convolution_1d(array, kernel):
    assert kernel.shape[0] % 2 == 1
    pad_length = kernel.shape[0] // 2
    padded_array = jnp.pad(array, pad_length, "wrap")
    return jnp.correlate(padded_array, kernel, mode="valid")


# basic test cases
if __name__ == "__main__":
    import jax
    from jax import Array

    N = 10
    M = 4
    memory = Memory(N, M)

    read_weights = jnp.divide(jnp.ones(N), N)
    memory_variables = memory.init(
        jax.random.key(common.JAX.RANDOM_SEED), read_weights, method=Memory.read
    )
    # print(memory_variables)

    read_output = cast(
        Array, memory.apply(memory_variables, read_weights, method=Memory.read)
    )
    # print(f'read output: {read_output}')
    expected_read = jnp.average(
        jnp.array(
            memory_variables[common.JAX.PARAMS][common.MACHINES.GRAVES2014.MEMORY_BIAS]
        ),
        axis=1,
    ).squeeze(0)
    assert (
        jnp.sum(jnp.abs(jnp.subtract(read_output, expected_read))) < M * 1e-5
    ), "Memory read function did not return expected vector"
    print("Memory read works as expected with a uniform vector")

    write_weights = jnp.expand_dims(jnp.divide(jnp.ones(N), N), axis=0)
    erase_vector = jnp.expand_dims(jnp.multiply(jnp.ones(M), N), axis=0)
    add_vector = jnp.expand_dims(jnp.multiply(jnp.ones(M), N), axis=0)
    write_output_full = memory.apply(
        memory_variables,
        write_weights,
        erase_vector,
        add_vector,
        method=Memory.write,
        mutable=[common.JAX.PARAMS],
    )
    # print(write_output_full)
    write_output = cast(
        Array,
        write_output_full[1][common.JAX.PARAMS][common.MACHINES.GRAVES2014.MEMORY_BIAS],
    )
    expected_write = jnp.ones((1, N, M))
    assert (
        jnp.sum(jnp.abs(jnp.subtract(write_output, expected_write))) < M * N * 1e-5
    ), "Memory write function did not update memory as expected"
    print("Memory write works as expected with a uniform vector")

    # TODO test memory addressing

    # TODO test memory grad calculation
    def loss(memory_variables, read_weights):
        read_output = cast(
            Array, memory.apply(memory_variables, read_weights, method=Memory.read)
        )
        likelihoods = read_output * expected_read + (1 - read_output) * (
            1 - expected_read
        )
        return -jnp.sum(jnp.log(likelihoods))

    mem_grad, read_grad = jax.grad(loss, (0, 1))(memory_variables, read_weights)
    # print(f'{mem_grad=}')
    # print(f'{read_grad=}')
    print("Jax Grad on memory write is not erroring")
