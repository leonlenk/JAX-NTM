import jax.numpy as jnp
import optax
from flax import linen as nn

from Common import common
from Memories.MemoryInterface import MemoryInterface


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
        self.memory_bias = self.param(
            common.GRAVES2014_MEMORY_BIAS,
            memory_bias_initializer,
            (1, self.N, self.M),
        )

        # TODO do we need this copy?
        # TODO memory is frozen outside of setup - can't write
        self.memory = self.memory_bias.copy()

    def size(self):
        return self.N, self.M

    def read(self, read_weights):
        """arXiv:1410.5401 section 3.1"""
        return jnp.matmul(read_weights, self.memory).squeeze(1)

    def write(self, read_weights, erase_vector, add_vector):
        """arXiv:1410.5401 section 3.2"""

        # calculate erase and add vectors
        read_weights = jnp.expand_dims(read_weights.squeeze(0), axis=1)
        erase = jnp.matmul(read_weights, erase_vector)
        add = jnp.matmul(read_weights, add_vector)
        # update memory
        self.memory = jnp.multiply(self.memory, 1 - erase)
        self.memory = jnp.add(self.memory, add)

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
            * optax.cosine_similarity(self.memory, key_vector, epsilon=1e-16),
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
