from Common import common

import jax.numpy as jnp
from flax import linen as nn
import optax


class Memory(nn.Module):
    """Memory interface for NTM (Graves 2014).
    Memory has a size of: batch size x N x M.
    N = number of memory locations
    N = size of vector at each memory location
    """

    batch_size: int
    N: int
    M: int

    def setup(self):
        memory_bias_initializer = (
            nn.initializers.uniform()
        )  # TODO test different memory bias initializers
        self.memory_bias = self.param(
            common.MEMORY_ORIGINAL_GRAVES2014_MEMORY_BIAS,
            memory_bias_initializer,
            (1, self.N, self.M),
        )

        self.memory = self.memory_bias.copy().repeat(self.batch_size, 0)

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
        """warXiv:1410.5401 section 3.2"""

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
            previous_weights, content_weights, interp_gate_scalar
        )

        # calculate and sharpen weights
        weights = self._shift(gate_weights, shift_weights)
        sharpened_weights = self._sharpen(weights, sharpen_scalar)

        return sharpened_weights

    def _similarity(self, k, beta):
        """arXiv:1410.5401 equation 6"""
        k = k.view(self.batch_size, 1, -1)
        w = nn.softmax(
            beta * optax.cosine_similarity(self.memory, k, epsilon=1e-16), axis=1
        )
        return w

    def _interpolate(self, w_prev, wc, g):
        return g * wc + (1 - g) * w_prev

    def _shift(self, wg, s):
        result = jnp.zeros(wg.size())
        for b in range(self.batch_size):
            result[b] = circular_convolution(wg[b], s[b])
        return result

    def _sharpen(self, ŵ, γ):
        w = ŵ**γ
        w = jnp.divide(w, jnp.sum(w, axis=1).view(-1, 1) + 1e-16)
        return w


def circular_convolution(w, s):
    # TODO implement circular convolution
    return
