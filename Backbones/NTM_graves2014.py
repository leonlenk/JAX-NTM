from typing import Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import Array

from Common import globals
from Common.BackboneInterface import BackboneInterface
from Common.ControllerInterface import ControllerInterface
from Common.globals import METADATA, MODELS


class LSTMModel(BackboneInterface):
    """Basic stacked LSTM for controlling an NTM"""

    prng_key: Array
    layers: int
    num_outputs: int
    read_heads: Sequence[ControllerInterface]
    write_heads: Sequence[ControllerInterface]

    features: int

    def setup(self):
        # add a dense layer for the ultimate output
        self.kernel_init = nn.initializers.xavier_uniform()
        self.bias_init = nn.initializers.normal()

    @nn.compact
    def __call__(
        self, input, memory_weights, read_previous, write_previous, memory_model
    ):
        lstm_layers = [nn.OptimizedLSTMCell(self.features) for _ in range(self.layers)]
        hidden = lstm_layers[0].initialize_carry(self.prng_key, input.shape)

        for i in range(self.layers):
            hidden, input = lstm_layers[i](hidden, input)

        read_data, read_locations = self.read_heads[0](
            input,
            read_previous,
            memory_weights,
            memory_model,
        )
        memory_weights, write_locations = self.write_heads[0](
            input,
            write_previous,
            memory_weights,
            memory_model,
        )

        dense_input = jnp.concatenate([input, read_data], axis=-1)
        output = nn.sigmoid(
            nn.Dense(
                self.num_outputs, kernel_init=self.kernel_init, bias_init=self.bias_init
            )(dense_input)
        )

        return (output, memory_weights, read_locations, write_locations)

    def get_metadata(self) -> dict:
        return {
            METADATA.NAME: self.__class__.__name__,
            MODELS.LSTM.FEATURES: self.features,
            MODELS.LSTM.LAYERS: self.layers,
            MODELS.LSTM.NUM_OUTPUTS: self.num_outputs,
            # TODO accept a seed and convert it into a prng_key so we can add METADATA.SEED
        }


# basic test cases
if __name__ == "__main__":
    import optax
    from flax.training import train_state
    from jax.tree_util import tree_flatten

    from Controllers.NTM_graves2014 import NTMReadController, NTMWriteController
    from Memories.NTM_graves2014 import NTMMemory

    layers = 4
    batch_size = 8
    input_features = 20
    input_length = 1
    memory_n = 8
    memory_m = 12
    lr = 1e-3
    num_recursions = 2
    num_outputs = input_features - memory_m

    key1, key2, key3 = jax.random.split(jax.random.key(globals.JAX.RANDOM_SEED), num=3)

    memory_model = NTMMemory()
    read_head = [NTMReadController(memory_n, memory_m)]
    write_head = [NTMWriteController(memory_n, memory_m)]

    model = LSTMModel(key3, layers, num_outputs, read_head, write_head, memory_m)
    init_input = jnp.ones((input_features,))
    memory_weights = jnp.zeros((memory_n, memory_m))
    params = model.init(
        key2,
        init_input,
        memory_weights,
        jnp.ones((memory_n,)),
        jnp.ones((memory_n,)),
        memory_model,
    )
    model_state = train_state.TrainState.create(
        apply_fn=model.apply,
        tx=optax.adam(lr),
        params=params[globals.JAX.PARAMS],
    )

    def loss_fn(model_params):
        sample_batch = jnp.ones((input_features,))
        read_previous = jnp.ones((memory_n,))
        write_previous = jnp.ones((memory_n,))
        memory_weights = jnp.zeros((memory_n, memory_m))

        for _ in range(num_recursions):
            (
                sample_batch,
                read_data,
                memory_weights,
                read_previous,
                write_previous,
            ) = model_state.apply_fn(
                {globals.JAX.PARAMS: model_params},
                sample_batch,
                memory_weights,
                read_previous,
                write_previous,
                memory_model,
            )
        return jnp.sum(sample_batch)

    gradient_fn = jax.grad(loss_fn, argnums=(0))
    model_grads = gradient_fn(model_state.params)

    flat_grads, _ = tree_flatten(model_grads)
    for grad in flat_grads:
        assert jnp.nonzero(grad)

    print("passed all tests")
