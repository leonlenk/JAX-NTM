from Common import common

import jax.numpy as jnp
from flax import linen as nn
import optax


class Memory(nn.Module):
    """Memory interface for NTM from Graves 2014 (arXiv:1410.5401).
    Memory has a size of: batch size x N x M.
    N = number of memory locations
    N = size of vector at each memory location
    """

    # TODO batch size logic may be unnecessary, due to JAX batch handling?
    batch_size: int
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

        self.memory = self.memory_bias.copy().repeat(self.batch_size, axis=1)

    def size(self):
        return self.N, self.M

    def read(
        self, read_weights
    ):  # TODO - figure out all the necessary squeezing/unsqueezing
        """arXiv:1410.5401 section 3.1"""
        return jnp.matmul(read_weights.unsqueeze(1), self.memory).squeeze(1)

    def write(
        self, read_weights, erase_vector, add_vector
    ):  # TODO - figure out all the necessary squeezing/unsqueezing
        """arXiv:1410.5401 section 3.2"""

        # calculate erase and add vectors
        erase = jnp.matmul(read_weights.unsqueeze(-1), erase_vector.unsqueeze(1))
        add = jnp.matmul(read_weights.unsqueeze(-1), add_vector.unsqueeze(1))
        # update memory
        self.memory = jnp.multiply(self.memory, (1 - erase) + add)

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
        """arXiv:1410.5401 equation 6"""
        key_vector = key_vector.view(self.batch_size, 1, -1)
        w = nn.softmax(
            key_strength
            * optax.cosine_similarity(self.memory, key_vector, epsilon=1e-16),
            axis=1,
        )
        return w

    def _interpolate(self, weights, previous_weights, interp_gate_scalar):
        """arXiv:1410.5401 equation 7"""
        return (
            interp_gate_scalar * weights + (1 - interp_gate_scalar) * previous_weights
        )

    def _shift(self, gated_weighting, shift_weighting):
        """arXiv:1410.5401 equation 8"""
        result = jnp.zeros(gated_weighting.size())
        for b in range(self.batch_size):
            result[b] = circular_convolution_1d(gated_weighting[b], shift_weighting[b])
        return result

    def _sharpen(self, weights, sharpen_scalar):
        """arXiv:1410.5401 equation 9"""
        w = weights**sharpen_scalar
        w = jnp.divide(w, jnp.sum(w, axis=1).view(-1, 1) + 1e-16)
        return w


def circular_convolution_1d(array, kernel):
    assert kernel.shape[0] % 2 == 1
    pad_length = kernel.shape[0] // 2
    padded_array = jnp.pad(array, pad_length, "wrap")
    return jnp.correlate(padded_array, kernel, mode="valid")
