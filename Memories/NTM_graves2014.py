import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

from Common import globals
from Common.MemoryInterface import MemoryInterface


class Memory(MemoryInterface):
    """Memory interface for NTM from Graves 2014 (arXiv:1410.5401).
    Memory has a size of: N x M.
    N = number of memory locations
    N = size of vector at each memory location
    """

    def read(self, memory_weights, read_weights):
        """arXiv:1410.5401 section 3.1"""
        return jnp.matmul(read_weights, memory_weights)

    def write(self, memory_weights, write_weights, erase_vector, add_vector):
        """arXiv:1410.5401 section 3.2"""

        # calculate erase and add vectors
        write_weights = jnp.expand_dims(write_weights, axis=1)
        erase_vector = jnp.expand_dims(erase_vector, axis=0)
        erase = jnp.matmul(write_weights, erase_vector)
        add_vector = jnp.expand_dims(add_vector, axis=0)
        add = jnp.matmul(write_weights, add_vector)
        # update memory
        memory_weights = jnp.multiply(memory_weights, 1 - erase)
        memory_weights = jnp.add(memory_weights, add)

        return memory_weights

    def address(
        self,
        memory_weights,
        key_vector,
        key_strength,
        interp_gate_scalar,
        shift_weights,
        sharpen_scalar,
        previous_weights,
    ):
        """arXiv:1410.5401 section 3.3"""
        # content addressing
        content_weights = self._similarity(memory_weights, key_vector, key_strength)

        # location addressing
        gate_weights = self._interpolate(
            content_weights, previous_weights, interp_gate_scalar
        )

        # calculate and sharpen weights
        weights = self._shift(gate_weights, shift_weights)
        sharpened_weights = self._sharpen(weights, sharpen_scalar)

        return sharpened_weights

    def _similarity(self, memory_weights, key_vector, key_strength):
        """arXiv:1410.5401 equations 5-6"""
        w = nn.softmax(
            key_strength
            * optax.cosine_similarity(memory_weights, key_vector, epsilon=1e-16),
        )
        return w

    def _interpolate(self, weights, previous_weights, interp_gate_scalar):
        """arXiv:1410.5401 equation 7"""
        return (
            interp_gate_scalar * weights + (1 - interp_gate_scalar) * previous_weights
        )

    def _shift(self, gated_weighting, shift_weighting):
        """arXiv:1410.5401 equation 8"""
        return circular_convolution_1d(gated_weighting, shift_weighting)

    def _sharpen(self, weights, sharpen_scalar):
        """arXiv:1410.5401 equation 9"""
        w = weights**sharpen_scalar
        w = jnp.divide(w, jnp.sum(w) + 1e-16)
        return w


@jax.jit
def circular_convolution_1d(array, kernel):
    assert kernel.shape[0] % 2 == 1
    pad_length = kernel.shape[0] // 2
    padded_array = jnp.pad(array, pad_length, "wrap")
    return jnp.correlate(padded_array, kernel, mode="valid")


# basic test cases
if __name__ == "__main__":
    import jax

    test_n = 10
    test_m = 4
    learning_rate = 5e-3
    memory = Memory()
    weights = jnp.zeros((test_n, test_m))

    read_weights = jnp.divide(jnp.ones(test_n), test_n)

    rng_key = jax.random.key(globals.JAX.RANDOM_SEED)
    # print(memory_variables)

    read_output = memory.read(weights, read_weights)
    # print(f'read output: {read_output}')
    expected_read = jnp.average(weights, axis=0)

    assert (
        jnp.sum(jnp.abs(jnp.subtract(read_output, expected_read))) < test_m * 1e-5
    ), "Memory read function did not return expected vector"
    print("Memory read works as expected with a uniform vector")

    write_weights = jnp.divide(jnp.ones(test_n), test_n)
    erase_vector = jnp.expand_dims(jnp.multiply(jnp.ones(test_m), test_n), axis=0)
    add_vector = jnp.expand_dims(jnp.multiply(jnp.ones(test_m), test_n), axis=0)
    write_output = memory.write(
        weights,
        write_weights,
        erase_vector,
        add_vector,
    )
    # print(write_output)

    expected_write = jnp.ones((test_n, test_m))
    assert (
        jnp.sum(jnp.abs(jnp.subtract(write_output, expected_write)))
        < test_m * test_n * 1e-5
    ), "Memory write function did not update memory as expected"
    print("Memory write works as expected with a uniform vector")

    # TODO test memory addressing

    # TODO test memory grad calculation
    def loss(memory_weights, read_weights):
        read_output = memory.read(memory_weights, read_weights)
        likelihoods = read_output * expected_read + (1 - read_output) * (
            1 - expected_read
        )
        return -jnp.sum(jnp.log(likelihoods))

    mem_grad, read_grad = jax.grad(loss, (0, 1))(weights, read_weights)
    # print(f'{mem_grad=}')
    # print(f'{read_grad=}')
    print("Jax Grad on memory write is not erroring")

    print("passed all tests")
