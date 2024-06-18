from Memories.NTM_graves2014 import Memory
from Models.NTM_graves2014 import LSTMModel
# from Controllers.NTM_graves2014 import *

from flax import linen as nn
import jax.numpy as jnp


class NTM(nn.Module):
    """Implementation of Graves 2014 NTM (arXiv:1410.5401)"""

    input_size: int
    output_size: int
    memory: Memory
    # controllers:
    model: LSTMModel

    def setup(self):
        self.N, self.M = self.memory.size()

        # TODO initialize the controller

        # add a dense layer for the ultimate output
        kernel_init = nn.initializers.xavier_uniform()
        bias_init = nn.initializers.normal()
        self.output_layer = nn.Dense(
            self.num_outputs, kernel_init=kernel_init, bias_init=bias_init
        )

    # TODO how do we deal with batch size?

    def __call__(self, input, previous_state):
        previous_reads, previous_model_state, previous_controller_states = (
            previous_state
        )

        full_input = jnp.concatenate([input] + previous_reads, axis=1)
        model_output, model_state = self.model(full_input, previous_model_state)

        # TODO perform controller operations
        reads = []
        controller_states = []

        dense_input = jnp.concatenate([model_output] + reads, axis=1)
        output = nn.sigmoid(self.output_layer(dense_input))

        return output, (reads, model_state, controller_states)
