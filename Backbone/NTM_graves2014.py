import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import Array

from Common import globals
from Common.BackboneInterface import BackboneInterface, PreviousState
from Common.ControllerInterface import ControllerInterface


class LSTMModel(BackboneInterface):
    """Basic stacked LSTM for controlling an NTM"""

    prng_key: Array
    features: int
    layers: int
    num_outputs: int
    read_head: ControllerInterface
    write_head: ControllerInterface

    def setup(self):
        # add a dense layer for the ultimate output
        self.kernel_init = nn.initializers.xavier_uniform()
        self.bias_init = nn.initializers.normal()

    @nn.compact
    def __call__(self, input, memory_model, previous_state):
        lstm_layers = [nn.OptimizedLSTMCell(self.features) for _ in range(self.layers)]
        hidden = lstm_layers[0].initialize_carry(self.prng_key, input.shape)

        for i in range(self.layers):
            hidden, input = lstm_layers[i](hidden, input)

        read_data, read_locations = self.read_head(
            input,
            previous_state.read_previous,
            previous_state.memory_weights,
            memory_model,
        )
        memory_weights, write_locations = self.write_head(
            input,
            previous_state.write_previous,
            previous_state.memory_weights,
            memory_model,
        )

        dense_input = jnp.concatenate([input, read_data], axis=-1)
        output = nn.sigmoid(
            nn.Dense(
                self.num_outputs, kernel_init=self.kernel_init, bias_init=self.bias_init
            )(dense_input)
        )

        return (
            output,
            read_data,
            PreviousState(memory_weights, read_locations, write_locations),
        )


# basic test cases
if __name__ == "__main__":
    import optax
    from flax.training import train_state

    from Controllers.NTM_graves2014 import NTMReadController, NTMWriteController
    from Memories.NTM_graves2014 import Memory

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

    memory_model = Memory(key1, (1, memory_n, memory_m), optax.adam(lr))
    read_head = NTMReadController(memory_n, memory_m)
    write_head = NTMWriteController(memory_n, memory_m)

    model = LSTMModel(key3, memory_m, layers, num_outputs, read_head, write_head)
    init_input = jnp.ones((input_length, input_features))
    init_previous_state = PreviousState(
        memory_model.weights, jnp.ones((1, memory_n)), jnp.ones((1, memory_n))
    )
    params = model.init(key2, init_input, memory_model, init_previous_state)
    model_state = train_state.TrainState.create(
        apply_fn=model.apply,
        tx=optax.adam(lr),
        params=params[globals.JAX.PARAMS],
    )

    def loss_fn(model_params, memory_weights):
        sample_batch = jnp.ones((input_length, input_features))
        previous_state = PreviousState(
            memory_weights, jnp.ones((1, memory_n)), jnp.ones((1, memory_n))
        )
        for _ in range(num_recursions):
            (sample_batch, read_data, previous_state), variables = model_state.apply_fn(
                {globals.JAX.PARAMS: model_params},
                sample_batch,
                memory_model,
                previous_state,
                mutable=["state"],
            )
            sample_batch = jnp.concat((sample_batch, read_data), axis=1)

        return jnp.sum(sample_batch)

    gradient_fn = jax.grad(loss_fn, argnums=(0, 1))
    model_grads, mem_grads = gradient_fn(model_state.params, memory_model.weights)

    print(model_grads)
    print(mem_grads)
