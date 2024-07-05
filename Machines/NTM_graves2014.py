import jax.numpy as jnp
from flax import linen as nn
from jax import Array

from Backbone.NTM_graves2014 import LSTMModel
from Common import globals
from Controllers.NTM_graves2014 import NTMReadController, NTMWriteController
from Memories.NTM_graves2014 import Memory


class NTM(nn.Module):
    """Implementation of Graves 2014 NTM (arXiv:1410.5401)"""

    input_size: int
    output_size: int
    memory: Memory
    read_controllers: list[NTMReadController]
    write_controllers: list[NTMWriteController]
    model: LSTMModel

    def setup(self):
        self.N_dim_memory, self.M_dim_memory = self.memory.size()

        # initialize bias for each read head
        self.read_controller_biases = []
        controller_bias_initializer = nn.initializers.normal()
        for i in range(len(self.read_controllers)):
            self.read_controller_biases.append(
                self.param(
                    f"{globals.MACHINES.GRAVES2014.READ_CONTROLLER_BIAS}{i}",
                    controller_bias_initializer,
                    (1, self.M),
                )
            )

        # add a dense layer for the ultimate output
        kernel_init = nn.initializers.xavier_uniform()
        bias_init = nn.initializers.normal()
        self.output_layer = nn.Dense(
            self.num_outputs, kernel_init=kernel_init, bias_init=bias_init
        )

    # TODO do we need functions to reinitialize the states of everything?

    # TODO how do we deal with batch size?

    def __call__(
        self,
        input: Array,
        previous_state: tuple[list[Array], list[Array], list[Array], list[Array]],
    ) -> tuple[Array, tuple[list[Array], list[Array], list[Array], list[Array]]]:
        (
            previous_reads,
            previous_model_state,
            previous_read_controller_states,
            previous_write_controller_states,
        ) = previous_state

        full_input = jnp.concatenate([input] + previous_reads, axis=1)
        model_output, model_state = self.model(full_input, previous_model_state)

        # perform controller operations
        reads = []
        read_controller_states = []
        write_controller_states = []
        for controller, prev_controller_state in zip(
            self.read_controllers, previous_read_controller_states
        ):
            read, controller_state = controller(model_output, prev_controller_state)
            reads.append(read)
            read_controller_states.append(controller_state)
        for controller, prev_controller_state in zip(
            self.write_controllers, previous_write_controller_states
        ):
            controller_state = controller(model_output, prev_controller_state)
            read_controller_states.append(controller_state)

        # run the outputs through the dense layer
        dense_input = jnp.concatenate([model_output] + reads, axis=1)
        output = nn.sigmoid(self.output_layer(dense_input))

        return output, (
            reads,
            model_state,
            read_controller_states,
            write_controller_states,
        )
